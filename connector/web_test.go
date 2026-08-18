package connector

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebConnector_HTML(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(`<html><head><title>Page T</title></head><body><script>x</script><p>Hello web</p></body></html>`))
	}))
	defer ts.Close()

	docs, err := (&WebConnector{}).Fetch(context.Background(), ts.URL)
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	d := docs[0]
	if d.Title != "Page T" {
		t.Errorf("title: got %q", d.Title)
	}
	if !strings.Contains(d.Content, "Hello web") || strings.Contains(d.Content, "script") {
		t.Errorf("content: %q", d.Content)
	}
}

func TestWebConnector_PlainText(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("raw body"))
	}))
	defer ts.Close()

	docs, err := (&WebConnector{}).Fetch(context.Background(), ts.URL)
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if docs[0].Content != "raw body" {
		t.Errorf("content: %q", docs[0].Content)
	}
}

func TestWebConnector_Errors(t *testing.T) {
	w := &WebConnector{}
	if _, err := w.Fetch(context.Background(), "ftp://x"); err == nil {
		t.Error("expected scheme error")
	}
	if _, err := w.Fetch(context.Background(), "not a url://"); err == nil {
		t.Error("expected parse error")
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/missing" {
			w.WriteHeader(404)
			return
		}
		if r.URL.Path == "/binary" {
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Write([]byte{1, 2, 3})
			return
		}
		if r.URL.Path == "/big" {
			w.Header().Set("Content-Type", "text/plain")
			w.Write(make([]byte, 200))
			return
		}
	}))
	defer ts.Close()
	if _, err := w.Fetch(context.Background(), ts.URL+"/missing"); err == nil {
		t.Error("expected 404 error")
	}
	if _, err := w.Fetch(context.Background(), ts.URL+"/binary"); err == nil {
		t.Error("expected content-type error")
	}
	small := &WebConnector{MaxBytes: 100}
	if _, err := small.Fetch(context.Background(), ts.URL+"/big"); err == nil {
		t.Error("expected max-bytes error")
	}
}

func TestWebConnector_RateLimit(t *testing.T) {
	// An absurdly high rate must not block; a zero rate uses the default.
	if err := (&WebConnector{RateLimit: 1e6}).wait(context.Background()); err != nil {
		t.Fatalf("wait: %v", err)
	}
}
