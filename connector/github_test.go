package connector

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGitHubConnector_ReadmeAndIssues(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/repos/acme/widget/readme", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "application/vnd.github.raw+json" {
			t.Errorf("missing raw accept header")
		}
		w.Write([]byte("# Widget\nGreat widget."))
	})
	mux.HandleFunc("/repos/acme/widget/issues", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[
			{"number": 1, "title": "Bug A", "body": "Broken", "state": "open", "labels": [{"name": "bug"}], "url": "https://x/1"},
			{"number": 2, "title": "PR X", "body": "", "state": "closed", "url": "https://x/2", "pull_request": {"url": "p"}}
		]`))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	g := &GitHubConnector{BaseURL: ts.URL, IncludeIssues: true, MaxIssues: 10}
	docs, err := g.Fetch(context.Background(), "acme/widget")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(docs) != 2 { // readme + 1 issue (PR filtered out)
		t.Fatalf("expected 2 docs, got %d", len(docs))
	}
	if !strings.Contains(docs[0].Content, "Widget") {
		t.Errorf("readme content: %q", docs[0].Content)
	}
	if docs[1].Title != "Bug A" || !strings.Contains(docs[1].Content, "Broken") {
		t.Errorf("issue doc: %+v", docs[1])
	}
}

func TestGitHubConnector_NoReadme(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/repos/o/r/readme", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	})
	mux.HandleFunc("/repos/o/r/issues", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[{"number": 7, "title": "T", "body": "B", "state": "open"}]`))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	docs, err := (&GitHubConnector{BaseURL: ts.URL, IncludeIssues: true}).Fetch(context.Background(), "o/r")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(docs) != 1 || docs[0].Title != "T" {
		t.Fatalf("got %d docs", len(docs))
	}
}

func TestGitHubConnector_Errors(t *testing.T) {
	g := &GitHubConnector{BaseURL: "http://127.0.0.1:1"}
	if _, err := g.Fetch(context.Background(), "no-slash"); err == nil {
		t.Error("expected ref format error")
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	}))
	defer ts.Close()
	if _, err := (&GitHubConnector{BaseURL: ts.URL}).Fetch(context.Background(), "o/r"); err == nil {
		t.Error("expected error when nothing is available")
	}
}
