package ingest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/deagy/recall/connector"
	"github.com/deagy/recall/index"
)

// TestEndToEnd_WebToStore validates the full ingestion path:
// WebConnector (HTTP fetch + HTML extraction) -> pipeline -> store
// (chunk + embed + index) -> search.
func TestEndToEnd_WebToStore(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/physics.html", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<html><head><title>Physics</title></head><body>
			<p>Quantum entanglement lets distant particles share correlated states.</p>
			<p>Measurements on one particle instantaneously constrain the other.</p>
		</body></html>`))
	})
	mux.HandleFunc("/baking.html", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<html><head><title>Baking</title></head><body>
			<p>Sourdough bread needs a stiff dough and long cold fermentation.</p>
		</body></html>`))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	s := makeStore(t)
	wc := &connector.WebConnector{Client: ts.Client()}

	for _, page := range []string{"/physics.html", "/baking.html"} {
		p, err := NewPipeline(Options{Store: s, Connector: wc, Source: ts.URL + page})
		if err != nil {
			t.Fatalf("pipeline %s: %v", page, err)
		}
		res, err := p.Run(context.Background())
		if err != nil {
			t.Fatalf("run %s: %v", page, err)
		}
		if res.Loaded != 1 || res.Uploaded != 1 {
			t.Fatalf("run %s: %+v", page, res)
		}
	}

	// Search must retrieve the baking chunk for a bread query.
	results, err := s.Search(context.Background(), "sourdough bread", index.SearchOptions{TopK: 1})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no search results")
	}
	if !strings.Contains(strings.ToLower(results[0].Chunk.Content), "sourdough") {
		t.Errorf("top result not about sourdough: %q", results[0].Chunk.Content)
	}
}

// TestEndToEnd_IncrementalWithConnector verifies incremental state works
// with connector-sourced documents (URL-stable IDs).
func TestEndToEnd_IncrementalWithConnector(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("stable connector body text for incremental testing."))
	}))
	defer ts.Close()

	s := makeStore(t)
	inc, err := NewIncremental("")
	if err != nil {
		t.Fatal(err)
	}
	opts := Options{Store: s, Connector: &connector.WebConnector{Client: ts.Client()}, Source: ts.URL, Incremental: inc}
	p, _ := NewPipeline(opts)

	if res, err := p.Run(context.Background()); err != nil || res.Uploaded != 1 {
		t.Fatalf("run1: %v %+v", err, res)
	}
	if res, err := p.Run(context.Background()); err != nil || res.Uploaded != 0 || res.Skipped != 1 {
		t.Fatalf("run2: %v %+v", err, res)
	}
}
