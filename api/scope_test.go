package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
	"github.com/deagy/recall/testutil"
)

// scopedFixture wires a server over a store holding documents in "ns-a" and
// "ns-b" (plus a "default" document), protected by ScopedAPIKeyAuth:
// "team-a" is scoped to ns-a, "admin" is unrestricted. The graph holds
// alice (sourced from ns-a), bob (sourced from ns-b), and ghost (unsourced).
type scopedFixture struct {
	srv   *Server
	store *store.MemoryStore
}

func newScopedFixture(t *testing.T) *scopedFixture {
	t.Helper()
	emb := testutil.NewMockEmbedder(16)
	st, err := store.NewMemoryStore(store.Config{Namespace: "default", Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	ctx := context.Background()
	docs := []struct {
		id      string
		ns      string
		content string
	}{
		{"doc-ns-a", "ns-a", "Apples are crisp and sweet, picked from the orchard in autumn."},
		{"doc-ns-b", "ns-b", "Submarines glide beneath the ocean surface carrying cargo."},
		{"doc-default", "", "Weather forecasts rely on satellites and barometric pressure."},
	}
	for _, d := range docs {
		doc := core.NewDocument(d.id, d.id, d.id+".txt")
		doc.Namespace = d.ns
		if err := st.Upload(ctx, doc, d.content); err != nil {
			t.Fatalf("Upload %s: %v", d.id, err)
		}
	}

	gs := store.NewMemoryGraphStore()
	g := gs.Graph()
	entities := []*graph.Entity{
		graph.NewEntity("alice", "Alice", graph.EntityPerson),
		graph.NewEntity("bob", "Bob", graph.EntityPerson),
		graph.NewEntity("ghost", "Ghost", graph.EntityPerson),
	}
	entities[0].SourceChunks = []string{"doc-ns-a::chunk-0"}
	entities[1].SourceChunks = []string{"doc-ns-b::chunk-0"}
	for _, e := range entities {
		if !g.AddEntity(e) {
			t.Fatalf("AddEntity %s failed", e.ID)
		}
	}
	if !g.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.7)) {
		t.Fatal("AddRelation alice->bob failed")
	}

	// The chunk IDs above assume the default fixed chunker's
	// "docID::chunk-0" scheme for single-chunk documents.
	for _, id := range []string{"doc-ns-a::chunk-0", "doc-ns-b::chunk-0"} {
		if _, ok := st.GetChunk(id); !ok {
			t.Fatalf("expected chunk %q to exist (chunk id scheme changed?)", id)
		}
	}

	auth := NewScopedAPIKeyAuth(
		KeySpec{Key: "team-a", Namespaces: []string{"ns-a"}},
		KeySpec{Key: "admin"},
	)
	srv, err := NewServer(Config{
		Store:         st,
		Pipeline:      pipeline.NewRAGPipeline(st, nil).WithTopK(5).WithCitations(),
		Graph:         gs,
		Reasoner:      reasoning.NewEngine(g, reasoning.DefaultConfig()),
		Authenticator: auth,
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return &scopedFixture{srv: srv, store: st}
}

// do performs an authenticated request against the fixture server.
func (f *scopedFixture) do(t *testing.T, key, method, target string, body any, out any) *httptest.ResponseRecorder {
	t.Helper()
	var reader *strings.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshaling body: %v", err)
		}
		reader = strings.NewReader(string(b))
	} else {
		reader = strings.NewReader("")
	}
	req := httptest.NewRequest(method, target, reader)
	if key != "" {
		req.Header.Set("X-API-Key", key)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	f.srv.Handler().ServeHTTP(rec, req)
	if out != nil && rec.Code < 300 {
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			t.Fatalf("decoding %d response: %v (body=%s)", rec.Code, err, rec.Body.String())
		}
	}
	return rec
}

func TestScopedServer_Upload(t *testing.T) {
	f := newScopedFixture(t)

	// Unauthenticated upload is rejected.
	rec := f.do(t, "", "POST", "/upload", map[string]any{"namespace": "ns-a", "content": "hi"}, nil)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated upload = %d, want 401", rec.Code)
	}

	// Scoped key may upload into its namespace.
	rec = f.do(t, "team-a", "POST", "/upload", map[string]any{"id": "u1", "namespace": "ns-a", "content": "A new apple tree was planted by the orchard gate last spring morning."}, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("scoped upload to allowed ns = %d, want 200 (body=%s)", rec.Code, rec.Body.String())
	}

	// Scoped key may not upload into another namespace.
	rec = f.do(t, "team-a", "POST", "/upload", map[string]any{"id": "u2", "namespace": "ns-b", "content": "A new submarine docked at the harbor carrying cargo for the fleet."}, nil)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("scoped upload to disallowed ns = %d, want 403 (body=%s)", rec.Code, rec.Body.String())
	}

	// Scoped key without an explicit namespace targets the store default,
	// which is not in its scope.
	rec = f.do(t, "team-a", "POST", "/upload", map[string]any{"id": "u3", "content": "No namespace was given for this document so the store default applies."}, nil)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("scoped upload to default ns = %d, want 403 (body=%s)", rec.Code, rec.Body.String())
	}

	// Unrestricted key uploads anywhere.
	rec = f.do(t, "admin", "POST", "/upload", map[string]any{"id": "u4", "namespace": "ns-b", "content": "Another submarine was commissioned after a long and careful refit."}, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("unrestricted upload = %d, want 200 (body=%s)", rec.Code, rec.Body.String())
	}
}

func TestScopedServer_Search(t *testing.T) {
	f := newScopedFixture(t)

	// The unrestricted key sees documents from every namespace.
	var all searchResponse
	rec := f.do(t, "admin", "GET", "/search?q=the&k=10", nil, &all)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin search = %d, want 200", rec.Code)
	}
	if all.Count < 2 {
		t.Fatalf("admin search count = %d, want >= 2 (all namespaces)", all.Count)
	}

	// The scoped key only sees ns-a chunks.
	var scoped searchResponse
	rec = f.do(t, "team-a", "GET", "/search?q=the&k=10", nil, &scoped)
	if rec.Code != http.StatusOK {
		t.Fatalf("scoped search = %d, want 200", rec.Code)
	}
	if scoped.Count == 0 {
		t.Fatal("scoped search returned no results, want ns-a chunks")
	}
	for _, r := range scoped.Results {
		if ns := r.Metadata["namespace"]; ns != "ns-a" {
			t.Errorf("scoped search leaked chunk %q from namespace %v", r.ID, ns)
		}
	}
}

func TestScopedServer_HybridSearch(t *testing.T) {
	f := newScopedFixture(t)
	var scoped searchResponse
	rec := f.do(t, "team-a", "POST", "/hybrid-search", map[string]any{"query": "the surface", "k": 10}, &scoped)
	if rec.Code != http.StatusOK {
		t.Fatalf("scoped hybrid search = %d, want 200", rec.Code)
	}
	if scoped.Count == 0 {
		t.Fatal("scoped hybrid search returned no results, want ns-a chunks")
	}
	for _, r := range scoped.Results {
		if ns := r.Metadata["namespace"]; ns != "ns-a" {
			t.Errorf("scoped hybrid search leaked chunk %q from namespace %v", r.ID, ns)
		}
	}
}

func TestScopedServer_RAG(t *testing.T) {
	f := newScopedFixture(t)
	var scoped ragResponse
	rec := f.do(t, "team-a", "POST", "/rag", map[string]any{"query": "what is under the ocean"}, &scoped)
	if rec.Code != http.StatusOK {
		t.Fatalf("scoped rag = %d, want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	if len(scoped.Sources) == 0 {
		t.Fatal("scoped rag returned no sources, want ns-a chunks")
	}
	for _, s := range scoped.Sources {
		if ns := s.Metadata["namespace"]; ns != "ns-a" {
			t.Errorf("scoped rag leaked source %q from namespace %v", s.ID, ns)
		}
	}
	for _, c := range scoped.Citations {
		chunk, ok := f.store.GetChunk(c.ChunkID)
		if !ok {
			continue
		}
		if ns := chunk.GetMetadataString(core.MetadataKeyNamespace); ns != "ns-a" {
			t.Errorf("scoped rag cited chunk %q from namespace %q", c.ChunkID, ns)
		}
	}
}

func TestScopedServer_GraphEntity(t *testing.T) {
	f := newScopedFixture(t)

	// Scoped key sees its own namespace's entity, but the neighbor and the
	// relation to the out-of-scope entity are hidden.
	var er entityResponse
	rec := f.do(t, "team-a", "GET", "/graph/alice", nil, &er)
	if rec.Code != http.StatusOK {
		t.Fatalf("scoped graph entity = %d, want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	if er.Entity.ID != "alice" {
		t.Fatalf("entity = %q, want alice", er.Entity.ID)
	}
	for _, n := range er.Neighbors {
		if n.ID == "bob" {
			t.Error("scoped graph leaked out-of-scope neighbor bob")
		}
	}
	for _, rel := range er.Relations {
		if rel.To == "bob" || rel.From == "bob" {
			t.Error("scoped graph leaked relation touching out-of-scope entity bob")
		}
	}

	// Out-of-scope entities are reported as not found (no existence leak).
	rec = f.do(t, "team-a", "GET", "/graph/bob", nil, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("scoped graph out-of-scope entity = %d, want 404", rec.Code)
	}

	// Entities with no verifiable source chunk are hidden (fail closed).
	rec = f.do(t, "team-a", "GET", "/graph/ghost", nil, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("scoped graph unsourced entity = %d, want 404", rec.Code)
	}

	// The unrestricted key sees the full neighborhood.
	var full entityResponse
	rec = f.do(t, "admin", "GET", "/graph/alice", nil, &full)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin graph entity = %d, want 200", rec.Code)
	}
	found := false
	for _, n := range full.Neighbors {
		if n.ID == "bob" {
			found = true
		}
	}
	if !found {
		t.Error("admin graph entity should include neighbor bob")
	}
}

func TestScopedServer_GraphReason(t *testing.T) {
	f := newScopedFixture(t)

	// Path exploration: the alice->bob path touches an out-of-scope entity,
	// so the scoped key sees no paths.
	var scoped reasonResponse
	rec := f.do(t, "team-a", "POST", "/graph/reason", map[string]any{"from": "alice", "to": "bob"}, &scoped)
	if rec.Code != http.StatusOK {
		t.Fatalf("scoped reason = %d, want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	if len(scoped.Paths) != 0 {
		t.Errorf("scoped reason returned %d paths, want 0", len(scoped.Paths))
	}

	// The unrestricted key sees the path.
	var full reasonResponse
	rec = f.do(t, "admin", "POST", "/graph/reason", map[string]any{"from": "alice", "to": "bob"}, &full)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin reason = %d, want 200", rec.Code)
	}
	if len(full.Paths) == 0 {
		t.Error("admin reason should find the alice->bob path")
	}
}
