package connector

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// s3Service is the AWS service identifier used in SigV4 signatures.
const s3Service = "s3"

// sign adds SigV4 Authorization headers to req.
func (s *S3Connector) sign(req *http.Request) {
	amzDate := time.Now().UTC().Format("20060102T150405Z")
	const payloadHash = "UNSIGNED-PAYLOAD"
	headers := map[string]string{
		"host":                 req.Host,
		"x-amz-date":           amzDate,
		"x-amz-content-sha256": payloadHash,
	}
	canonicalRequest := buildCanonicalRequest(
		req.Method,
		canonicalURI(req.URL.Path),
		canonicalQueryString(parseCanonicalQuery(req.URL.RawQuery)),
		headers,
		payloadHash,
	)
	scope := amzDate[:8] + "/" + s.Region + "/" + s3Service + "/aws4_request"
	signature := sigV4Signature(stringToSign(canonicalRequest, amzDate, scope), amzDate, scope, s.Region, s3Service, s.SecretKey)

	req.Header.Set("Host", req.Host)
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	req.Header.Set("Authorization", fmt.Sprintf(
		"AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		s.AccessKey, scope, "host;x-amz-content-sha256;x-amz-date", signature))
}

// buildCanonicalRequest assembles the SigV4 canonical request:
// method, URI, query, sorted canonical headers, signed header names, and
// the payload hash, newline-joined.
func buildCanonicalRequest(method, canonicalURI, canonicalQuery string, headers map[string]string, payloadHash string) string {
	keys := make([]string, 0, len(headers))
	for k := range headers {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	canonHeaders := make([]string, 0, len(keys))
	for _, k := range keys {
		canonHeaders = append(canonHeaders, k+":"+strings.TrimSpace(headers[k]))
	}
	return strings.Join([]string{
		method,
		canonicalURI,
		canonicalQuery,
		strings.Join(canonHeaders, "\n"),
		strings.Join(keys, ";"),
		payloadHash,
	}, "\n")
}

// stringToSign assembles the SigV4 string to sign from the canonical
// request, timestamp, credential scope, and scope.
func stringToSign(canonicalRequest, amzDate, scope string) string {
	return strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzDate,
		scope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")
}

// sigV4Signature derives the scoped signing key and signs stringToSign.
func sigV4Signature(sts, amzDate, scope, region, service, secretKey string) string {
	kDate := hmacSHA256([]byte("AWS4"+secretKey), []byte(amzDate[:8]))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return hex.EncodeToString(hmacSHA256(kSigning, []byte(sts)))
}

// canonicalURI URI-escapes each path segment individually.
func canonicalURI(path string) string {
	if path == "" {
		return "/"
	}
	segments := strings.Split(path, "/")
	for i, seg := range segments {
		segments[i] = url.PathEscape(seg)
	}
	out := strings.Join(segments, "/")
	if !strings.HasPrefix(out, "/") {
		out = "/" + out
	}
	return out
}

// canonicalQueryString sorts and URI-encodes query parameters per SigV4.
func canonicalQueryString(q url.Values) string {
	if len(q) == 0 {
		return ""
	}
	type kv struct{ k, v string }
	pairs := make([]kv, 0, len(q))
	for k, vs := range q {
		for _, v := range vs {
			pairs = append(pairs, kv{url.PathEscape(k), url.PathEscape(v)})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].k != pairs[j].k {
			return pairs[i].k < pairs[j].k
		}
		return pairs[i].v < pairs[j].v
	})
	out := make([]string, 0, len(pairs))
	for _, p := range pairs {
		out = append(out, p.k+"="+p.v)
	}
	return strings.Join(out, "&")
}

// parseCanonicalQuery parses a raw query string into values; the signer
// re-canonicalizes deterministically, so decoding is safe here.
func parseCanonicalQuery(raw string) url.Values {
	q := url.Values{}
	for _, part := range strings.Split(raw, "&") {
		if part == "" {
			continue
		}
		k, v, _ := strings.Cut(part, "=")
		uk, _ := url.QueryUnescape(k)
		uv, _ := url.QueryUnescape(v)
		q.Add(uk, uv)
	}
	return q
}

// hmacSHA256 returns the HMAC-SHA256 digest.
func hmacSHA256(key, msg []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(msg)
	return h.Sum(nil)
}

// sha256Hex returns the hex-encoded SHA-256 digest.
func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
