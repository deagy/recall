package client

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/deagy/recall/api"
	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/distributed"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

// seedGraph adds a small transitive graph: Alice works_at Acme, Acme
// located_in Berlin.
func seedGraph(gs *store.MemoryGraphStore) {
	g := gs.Graph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("acme", "Acme", graph.EntityOrganizer))
	g.AddEntity(graph.NewEntity("berlin", "Berlin", graph.EntityLocation))
	g.AddRelation(graph.NewRelation("alice", "acme", "works_at", 1))
	g.AddRelation(graph.NewRelation("acme", "berlin", "located_in", 1))
}

// apiTestServer starts a real recall-server (in-memory store, mock
// embedder, pipeline, seeded graph, reasoner) and returns a client pointed
// at it.
func apiTestServer(t *testing.T, opts struct{ auth, graph bool }) *Client {
	t.Helper()
	cfg := &config.Config{}
	cfg.WithDefaults()
	st, err := store.NewMemoryStore(store.Config{
		Namespace:      cfg.Store.Namespace,
		Embedder:       embedder.NewMockEmbedder(cfg.Store.Embedder.Dimension),
		ChunkerFactory: chunker.NewFixed,
	})
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	apiCfg := api.Config{
		Store:    st,
		Pipeline: pipeline.NewRAGPipeline(st, nil).WithCitations(),
		Host:     "127.0.0.1",
		Port:     8080,
	}
	if opts.graph {
		gs := store.NewMemoryGraphStore()
		seedGraph(gs)
		apiCfg.Graph = gs
		apiCfg.Reasoner = reasoning.NewEngine(gs.Graph(), reasoning.DefaultConfig())
	}
	if opts.auth {
		apiCfg.Authenticator = api.NewScopedAPIKeyAuth(api.KeySpec{Key: "test-key"})
	}
	srv, err := api.NewServer(apiCfg)
	if err != nil {
		t.Fatalf("api.NewServer: %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	c, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestNew_Validation(t *testing.T) {
	if _, err := New(Config{BaseURL: ""}); err == nil {
		t.Error("empty base URL: expected error")
	}
	if _, err := New(Config{BaseURL: "notaurl"}); err == nil {
		t.Error("hostless URL: expected error")
	}
	c, err := New(Config{BaseURL: "http://localhost:8080/"})
	if err != nil {
		t.Fatalf("valid URL: %v", err)
	}
	if c.BaseURL() != "http://localhost:8080" {
		t.Errorf("base = %q, want trailing slash trimmed", c.BaseURL())
	}
}

func TestUpload_Search_HybridSearch(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{})

	up, err := c.Upload(ctx, UploadRequest{
		ID: "doc-1", Title: "Go Notes", Content: "Go is a compiled language with garbage collection.",
	})
	if err != nil {
		t.Fatalf("upload: %v", err)
	}
	if up.ID != "doc-1" || up.Chunks < 1 {
		t.Errorf("upload result = %+v", up)
	}
	if _, err := c.Upload(ctx, UploadRequest{
		ID: "doc-2", Title: "Rust Notes", Content: "Rust is a systems language with ownership and no garbage collector at runtime.",
	}); err != nil {
		t.Fatalf("upload 2: %v", err)
	}

	res, err := c.Search(ctx, "compiled language", SearchOptions{TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if res.Query != "compiled language" || res.Count != len(res.Results) {
		t.Errorf("search meta = %+v", res)
	}
	if len(res.Results) < 1 || res.Results[0].Document != "doc-1" {
		t.Errorf("search results = %+v", res.Results)
	}

	res, err = c.HybridSearch(ctx, "compiled language", SearchOptions{TopK: 5, BM25Weight: 0.7})
	if err != nil {
		t.Fatalf("hybrid: %v", err)
	}
	if len(res.Results) < 1 {
		t.Error("hybrid returned no results")
	}
}

func TestUpload_Errors(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{})

	// Missing content is a 400 from the server.
	if _, err := c.Upload(ctx, UploadRequest{ID: "x"}); err == nil {
		t.Fatal("expected error for empty content")
	} else {
		var ce *Error
		if !errors.As(err, &ce) || ce.StatusCode != http.StatusBadRequest {
			t.Errorf("unexpected error: %v", err)
		}
	}

	// An unreachable server is a transport error, not an API error.
	dead, _ := New(Config{BaseURL: "http://127.0.0.1:1", Timeout: time.Second})
	if _, err := dead.Upload(ctx, UploadRequest{ID: "x", Content: "y"}); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestRAG(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{})
	if _, err := c.Upload(ctx, UploadRequest{
		ID: "doc-1", Title: "Go Notes",
		Content: "Go is a compiled language with garbage collection and goroutines.",
	}); err != nil {
		t.Fatalf("upload: %v", err)
	}

	resp, err := c.RAG(ctx, "What is Go?", false)
	if err != nil {
		t.Fatalf("rag: %v", err)
	}
	if resp.Answer == "" || resp.Context == "" {
		t.Errorf("incomplete rag response: answer=%q context=%q", resp.Answer, resp.Context)
	}
	if resp.Tokens <= 0 || len(resp.Sources) < 1 || len(resp.Citations) < 1 {
		t.Errorf("rag metadata = tokens=%d sources=%d citations=%d", resp.Tokens, len(resp.Sources), len(resp.Citations))
	}

	// Hybrid mode should also work.
	if _, err := c.RAG(ctx, "What is Go?", true); err != nil {
		t.Errorf("rag hybrid: %v", err)
	}

	// Empty query is a 400.
	if _, err := c.RAG(ctx, "   ", false); err == nil {
		t.Error("expected error for blank query")
	}
}

func TestGraphEntity(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{graph: true})

	detail, err := c.GraphEntity(ctx, "alice")
	if err != nil {
		t.Fatalf("graph entity: %v", err)
	}
	if detail.Entity.ID != "alice" || detail.Entity.Label != "Alice" {
		t.Errorf("entity = %+v", detail.Entity)
	}
	if len(detail.Neighbors) != 1 || detail.Neighbors[0].ID != "acme" {
		t.Errorf("neighbors = %+v", detail.Neighbors)
	}
	if len(detail.Relations) != 1 || detail.Relations[0].Type != "works_at" {
		t.Errorf("relations = %+v", detail.Relations)
	}

	// Label lookup also works.
	detail, err = c.GraphEntity(ctx, "Acme")
	if err != nil {
		t.Fatalf("label lookup: %v", err)
	}
	if detail.Entity.ID != "acme" {
		t.Errorf("label lookup resolved to %q", detail.Entity.ID)
	}

	// Unknown entity is a 404 with the API error code.
	_, err = c.GraphEntity(ctx, "ghost")
	var ce *Error
	if !errors.As(err, &ce) || ce.StatusCode != http.StatusNotFound || ce.Code != "not_found" {
		t.Errorf("expected 404 not_found, got %v", err)
	}
}

func TestGraph_NoGraphConfigured(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{})
	if _, err := c.GraphEntity(ctx, "alice"); err == nil {
		t.Error("expected 400 when graph is not configured")
	}
	if _, err := c.Reason(ctx, ReasonRequest{Query: "anything"}); err == nil {
		t.Error("expected 400 when reasoner is not configured")
	}
}

func TestReason(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{graph: true})

	// Path exploration: alice -> berlin should find the 2-hop path through
	// acme (located_in is transitive in the default ruleset? works_at is
	// not; at minimum both entities resolve and paths may be empty).
	out, err := c.Reason(ctx, ReasonRequest{From: "alice", To: "berlin", MaxHops: 3})
	if err != nil {
		t.Fatalf("reason paths: %v", err)
	}
	_ = out

	// Natural-language reasoning over capitalized entity names.
	out, err = c.Reason(ctx, ReasonRequest{Query: "Alice works at Acme and Acme is located in Berlin", MaxHops: 3})
	if err != nil {
		t.Fatalf("reason query: %v", err)
	}

	// Neither query nor from/to is a 400.
	if _, err := c.Reason(ctx, ReasonRequest{MaxHops: 2}); err == nil {
		t.Error("expected error when query and from/to are both empty")
	}
}

func TestHealth_Diagnostics(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{})
	if _, err := c.Upload(ctx, UploadRequest{ID: "d", Content: "Hello world, this is a longer document used for the health check test."}); err != nil {
		t.Fatalf("upload: %v", err)
	}

	h, err := c.Health(ctx)
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if !h.OK || h.Status != "healthy" || h.Backend != "memory" || h.Count < 1 {
		t.Errorf("health = %+v", h)
	}

	d, err := c.Diagnostics(ctx)
	if err != nil {
		t.Fatalf("diagnostics: %v", err)
	}
	if !d.Health.OK || d.GeneratedAt.IsZero() {
		t.Errorf("diagnostics = %+v", d)
	}
}

func TestAuth(t *testing.T) {
	ctx := context.Background()
	c := apiTestServer(t, struct{ auth, graph bool }{auth: true})

	// Without a key, data endpoints are rejected.
	_, err := c.Upload(ctx, UploadRequest{ID: "d", Content: "x"})
	var ce *Error
	if !errors.As(err, &ce) || ce.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 without key, got %v", err)
	}

	// Health remains exempt.
	if _, err := c.Health(ctx); err != nil {
		t.Errorf("health should be exempt from auth: %v", err)
	}

	// With the key, the request succeeds.
	authorized, err := New(Config{BaseURL: c.BaseURL(), APIKey: "test-key"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := authorized.Upload(ctx, UploadRequest{ID: "d", Content: "A document long enough to be chunked by the default fixed chunker settings."}); err != nil {
		t.Errorf("upload with key: %v", err)
	}
}

func TestProbeClusterNode(t *testing.T) {
	ctx := context.Background()

	// A real distributed node endpoint.
	cluster := distributed.NewCluster(distributed.DefaultClusterConfig())
	if err := cluster.AddNode(&distributed.Node{ID: "n1", Address: "http://n1", Status: "online"}); err != nil {
		t.Fatalf("AddNode: %v", err)
	}
	ts := httptest.NewServer(distributed.HealthHandler(cluster, nil))
	defer ts.Close()

	d, err := ProbeClusterNode(ctx, ts.URL, time.Second)
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	if d.Health.Total != 1 || d.Health.Online != 1 || d.Health.Overall == "" {
		t.Errorf("cluster health = %+v", d.Health)
	}

	// A down cluster is reachable but reports Overall "down" in the body
	// (/diagnostics always answers 200; the state is in the payload).
	down := distributed.NewCluster(distributed.DefaultClusterConfig())
	if err := down.AddNode(&distributed.Node{ID: "n1", Address: "http://n1"}); err != nil {
		t.Fatalf("AddNode: %v", err)
	}
	if err := down.SetNodeStatus("n1", "offline"); err != nil {
		t.Fatalf("SetNodeStatus: %v", err)
	}
	downTS := httptest.NewServer(distributed.HealthHandler(down, nil))
	defer downTS.Close()
	downD, err := ProbeClusterNode(ctx, downTS.URL, time.Second)
	if err != nil {
		t.Fatalf("probe down cluster: %v", err)
	}
	if downD.Health.Overall != "down" || downD.Health.Offline != 1 {
		t.Errorf("expected down cluster, got %+v", downD.Health)
	}

	// A node that answers 500 is reported as a typed error.
	errTS := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer errTS.Close()
	_, err = ProbeClusterNode(ctx, errTS.URL, time.Second)
	var ce *Error
	if !errors.As(err, &ce) || ce.StatusCode != http.StatusInternalServerError {
		t.Errorf("expected 500 error, got %v", err)
	}

	// Invalid node URLs are rejected before any request.
	if _, err := ProbeClusterNode(ctx, "not-a-url", time.Second); err == nil {
		t.Error("expected error for invalid node URL")
	}
}
