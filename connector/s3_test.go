package connector

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeS3 serves ListObjectsV2 and object GETs in path style, and records
// the Authorization header of each request.
func fakeS3(t *testing.T) (*httptest.Server, *[]string) {
	t.Helper()
	auths := &[]string{}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		*auths = append(*auths, r.Header.Get("Authorization"))
		if r.URL.Query().Get("list-type") == "2" {
			w.Header().Set("Content-Type", "application/xml")
			fmt.Fprintf(w, `<?xml version="1.0"?><ListBucketResult><Contents><Key>docs/alpha.txt</Key></Contents><Contents><Key>docs/beta report.md</Key></Contents></ListBucketResult>`)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		w.Header().Set("ETag", `"abc123"`)
		w.Header().Set("Last-Modified", "Mon, 01 Jan 2026 00:00:00 GMT")
		fmt.Fprint(w, "body of "+r.URL.Path)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, auths
}

func TestS3Connector_Fetch(t *testing.T) {
	ts, auths := fakeS3(t)
	s := &S3Connector{
		Endpoint:  ts.URL,
		Region:    "us-east-1",
		AccessKey: "AKIAIOSFODNN7EXAMPLE",
		SecretKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		PathStyle: true,
	}
	docs, err := s.Fetch(context.Background(), "mybucket/docs")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 docs, got %d", len(docs))
	}
	if docs[0].Content != "body of /mybucket/docs/alpha.txt" {
		t.Errorf("doc0 content: %q", docs[0].Content)
	}
	if docs[1].ID != "s3://mybucket/docs/beta report.md" {
		t.Errorf("doc1 id: %q", docs[1].ID)
	}
	// Every request must carry a SigV4 Authorization header.
	if len(*auths) != 3 {
		t.Fatalf("expected 3 requests, got %d", len(*auths))
	}
	for i, a := range *auths {
		if !strings.HasPrefix(a, "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/") ||
			!strings.Contains(a, "SignedHeaders=host;x-amz-content-sha256;x-amz-date") ||
			!strings.Contains(a, "Signature=") {
			t.Errorf("request %d authorization malformed: %q", i, a)
		}
	}
}

func TestS3Connector_PrefixSent(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("list-type") == "2" {
			if got := r.URL.Query().Get("prefix"); got != "docs/notes" {
				t.Errorf("prefix: got %q", got)
			}
			w.Write([]byte(`<?xml version="1.0"?><ListBucketResult/>`))
			return
		}
	}))
	defer ts.Close()
	if _, err := (&S3Connector{Endpoint: ts.URL, Region: "us-east-1", PathStyle: true}).
		Fetch(context.Background(), "b/docs/notes"); err == nil {
		t.Error("expected no-objects error")
	}
}

func TestS3Connector_ListError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(403)
	}))
	defer ts.Close()
	if _, err := (&S3Connector{Endpoint: ts.URL, Region: "us-east-1", PathStyle: true}).
		Fetch(context.Background(), "b"); err == nil {
		t.Error("expected list error")
	}
}
