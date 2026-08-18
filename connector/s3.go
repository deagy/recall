package connector

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

// S3Connector lists and downloads objects from an S3-compatible endpoint
// (AWS S3 or MinIO) using Signature Version 4. The client is self-contained
// (no AWS SDK dependency): only ListObjectsV2 and GetObject are implemented.
type S3Connector struct {
	// Endpoint is the S3 base URL, e.g. "https://s3.us-west-2.amazonaws.com"
	// (AWS, virtual style) or "http://localhost:9000" (MinIO, path style).
	Endpoint string

	// Region for SigV4; required (use "us-east-1" or the bucket region).
	Region string

	// AccessKey / SecretKey are the static credentials. Empty keys produce
	// unsigned requests (works with public buckets).
	AccessKey string
	SecretKey string

	// PathStyle, when false (default), uses virtual-host style (bucket
	// prefixed to the host) for AWS. Set true for MinIO-style endpoints.
	PathStyle bool

	// Client is the HTTP client; default http.DefaultClient.
	Client *http.Client

	// MaxObjects caps the number of objects fetched; 0 means 100.
	MaxObjects int

	// MaxBytes caps each object body; 0 means 50 MiB.
	MaxBytes int64
}

// Name implements Connector.
func (s *S3Connector) Name() string { return "s3" }

// Fetch lists objects under ref ("bucket" or "bucket/prefix") and returns
// one document per object with the raw body as content.
func (s *S3Connector) Fetch(ctx context.Context, ref string) ([]*loader.Document, error) {
	bucket, prefix, err := parseS3Ref(ref)
	if err != nil {
		return nil, err
	}
	keys, err := s.listObjects(ctx, bucket, prefix)
	if err != nil {
		return nil, err
	}
	docs := make([]*loader.Document, 0, len(keys))
	for _, k := range keys {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		doc, err := s.getObject(ctx, bucket, k)
		if err != nil {
			return docs, fmt.Errorf("s3: %w", err)
		}
		docs = append(docs, doc)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("s3: no objects found for %q", ref)
	}
	return docs, nil
}

// parseS3Ref splits "bucket" or "bucket/prefix" into its parts.
func parseS3Ref(ref string) (bucket, prefix string, err error) {
	ref = strings.TrimPrefix(strings.TrimSpace(ref), "s3://")
	if ref == "" {
		return "", "", fmt.Errorf("s3: empty ref")
	}
	if i := strings.IndexByte(ref, '/'); i >= 0 {
		return ref[:i], ref[i+1:], nil
	}
	return ref, "", nil
}

// s3ListResult is the subset of the ListObjectsV2 response we consume.
type s3ListResult struct {
	Contents []struct {
		Key string `xml:"Key"`
	} `xml:"Contents"`
}

// listObjects returns up to MaxObjects object keys under prefix.
func (s *S3Connector) listObjects(ctx context.Context, bucket, prefix string) ([]string, error) {
	maxKeys := s.MaxObjects
	if maxKeys <= 0 || maxKeys > 1000 {
		maxKeys = 100
	}
	q := url.Values{}
	q.Set("list-type", "2")
	q.Set("max-keys", fmt.Sprintf("%d", maxKeys))
	if prefix != "" {
		q.Set("prefix", prefix)
	}
	req, err := s.newRequest(ctx, http.MethodGet, bucket, "/", q)
	if err != nil {
		return nil, err
	}
	resp, err := s.do(req)
	if err != nil {
		return nil, fmt.Errorf("s3: list %s/%s: %w", bucket, prefix, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("s3: list %s/%s returned %s", bucket, prefix, resp.Status)
	}
	var lr s3ListResult
	if err := xml.NewDecoder(resp.Body).Decode(&lr); err != nil {
		return nil, fmt.Errorf("s3: decode list: %w", err)
	}
	keys := make([]string, 0, len(lr.Contents))
	for _, c := range lr.Contents {
		keys = append(keys, c.Key)
	}
	return keys, nil
}

// getObject downloads one object and wraps it as a document.
func (s *S3Connector) getObject(ctx context.Context, bucket, key string) (*loader.Document, error) {
	// Pass the decoded key; the URL machinery and the SigV4 signer both
	// re-escape path segments identically.
	req, err := s.newRequest(ctx, http.MethodGet, bucket, "/"+key, nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.do(req)
	if err != nil {
		return nil, fmt.Errorf("s3: get %s/%s: %w", bucket, key, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("s3: get %s/%s returned %s", bucket, key, resp.Status)
	}
	maxBytes := s.MaxBytes
	if maxBytes == 0 {
		maxBytes = 50 << 20
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("s3: read %s/%s: %w", bucket, key, err)
	}
	if int64(len(body)) > maxBytes {
		return nil, fmt.Errorf("s3: %s/%s exceeds %d byte limit", bucket, key, maxBytes)
	}
	name := key
	if i := strings.LastIndexByte(key, '/'); i >= 0 {
		name = key[i+1:]
	}
	source := "s3://" + bucket + "/" + key
	d := loader.NewDocument(source, name, source, string(body))
	d.Metadata["s3_key"] = core.String{Value: key}
	if etag := resp.Header.Get("ETag"); etag != "" {
		d.Metadata["etag"] = core.String{Value: etag}
	}
	if mod := resp.Header.Get("Last-Modified"); mod != "" {
		d.Metadata["last_modified"] = core.String{Value: mod}
	}
	return d, nil
}

// newRequest builds a signed request. Virtual style rewrites the endpoint
// host to bucket.host; path style inserts /bucket into the path.
func (s *S3Connector) newRequest(ctx context.Context, method, bucket, path string, query url.Values) (*http.Request, error) {
	endpoint, err := url.Parse(s.Endpoint)
	if err != nil {
		return nil, fmt.Errorf("s3: invalid endpoint %q: %w", s.Endpoint, err)
	}
	if endpoint.Scheme == "" {
		endpoint.Scheme = "https"
	}
	host := endpoint.Host
	fullPath := path
	if !s.PathStyle && bucket != "" {
		host = bucket + "." + host
	} else if bucket != "" {
		fullPath = "/" + bucket + path
	}
	u := &url.URL{Scheme: endpoint.Scheme, Host: host, Path: fullPath}
	if len(query) > 0 {
		u.RawQuery = canonicalQueryString(query)
	}
	req, err := http.NewRequestWithContext(ctx, method, u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("s3: %w", err)
	}
	if s.AccessKey != "" && s.SecretKey != "" {
		s.sign(req)
	}
	return req, nil
}

// do executes the request with the configured client.
func (s *S3Connector) do(req *http.Request) (*http.Response, error) {
	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	return client.Do(req)
}
