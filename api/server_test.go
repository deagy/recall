package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
	"github.com/deagy/recall/testutil"
)

// apiFixture wires an in-memory server with documents, a RAG pipeline, a
// knowledge graph (alice -works_at-> acme -located_in-> paris), and a
// reasoning engine.
type apiFixture struct {
	srv   *Server
	store *store.MemoryStore
	graph *store.MemoryGraphStore
}

func newAPIFixture(t *testing.T) *apiFixture {
	t.Helper()
	emb := testutil.NewMockEmbedder(16)
	st, err := store.NewMemoryStore(store.Config{Namespace: "default", Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	ctx := context.Background()
	for _, doc := range []struct{ id, title, content string }{
		{"doc-1", "Alpha", "The quick brown fox jumps over the lazy dog near the river bank."},
		{"doc-2", "Beta", "Gorillas swing through the dense jungle canopy at first light."},
	} {
		if err := st.Upload(ctx, core.NewDocument(doc.id, doc.title, doc.id+".txt"), doc.content); err != nil {
			t.Fatalf("Upload %s: %v", doc.id, err)
		}
	}

	gs := store.NewMemoryGraphStore()
	g := gs.Graph()
	for _, e := range []*graph.Entity{
		graph.NewEntity("alice", "Alice", graph.EntityPerson),
		graph.NewEntity("acme", "Acme Corp", graph.EntityOrganizer),
		graph.NewEntity("paris", "Paris", graph.EntityLocation),
	} {
		if !g.AddEntity(e) {
			t.Fatalf("AddEntity %s failed", e.ID)
		}
	}
	for _, r := range []*graph.Relation{
		graph.NewRelation("alice", "acme", "works_at", 0.9),
		graph.NewRelation("acme", "paris", "located_in", 0.8),
	} {
		if !g.AddRelation(r) {
			t.Fatalf("AddRelation %s->%s failed", r.From, r.To)
		}
	}

	pipe := pipeline.NewRAGPipeline(st, nil).WithTopK(3).WithCitations()
	eng := reasoning.NewEngine(g, reasoning.DefaultConfig())

	srv, err := NewServer(Config{
		Store:    st,
		Pipeline: pipe,
		Graph:    gs,
		Reasoner: eng,
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return &apiFixture{srv: srv, store: st, graph: gs}
}

// do performs a request against the fixture server and decodes the JSON
// response into out (when non-nil).
func (f *apiFixture) do(t *testing.T, method, target string, body any, out any) *httptest.ResponseRecorder {
	t.Helper()
	var reader *bytes.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshaling body: %v", err)
		}
		reader = bytes.NewReader(b)
	} else {
		reader = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, target, reader)
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

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshaling: %v", err)
	}
	return b
}

func TestNewServer_RequiresStore(t *testing.T) {
	if _, err := NewServer(Config{}); err == nil {
		t.Fatal("expected error when Store is nil")
	}
}

func TestNewAPIKey(t *testing.T) {
	k1, err := NewAPIKey()
	if err != nil {
		t.Fatalf("NewAPIKey: %v", err)
	}
	if len(k1) != 32 {
		t.Fatalf("key length = %d, want 32", len(k1))
	}
	k2, err := NewAPIKey()
	if err != nil {
		t.Fatalf("NewAPIKey: %v", err)
	}
	if k1 == k2 {
		t.Fatal("two generated keys should differ")
	}
}

func TestServer_Upload(t *testing.T) {
	f := newAPIFixture(t)
	req := uploadRequest{
		ID: "doc-3", Title: "Gamma", Source: "gamma.txt",
		Tags:     []string{"zoo"},
		Metadata: map[string]any{"author": "tester", "rank": 2, "live": true},
		Content:  "Owls hunt small mammals during the night across the meadow.",
	}
	var resp uploadResponse
	rec := f.do(t, "POST", "/upload", req, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if resp.ID != "doc-3" || resp.Chunks < 1 {
		t.Errorf("response = %+v, want id doc-3 with chunks", resp)
	}
	if resp.CreatedAt == "" || resp.UpdatedAt == "" {
		t.Errorf("expected timestamps in response: %+v", resp)
	}

	rec = f.do(t, "GET", "/search?q=owls+mammals&k=5", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d", rec.Code)
	}
	var sr searchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &sr); err != nil {
		t.Fatalf("decoding search: %v", err)
	}
	if sr.Count == 0 {
		t.Fatal("expected search results after upload")
	}
	foundDoc3 := false
	for _, r := range sr.Results {
		if r.Document == "doc-3" && r.Metadata["author"] == "tester" {
			foundDoc3 = true
		}
	}
	if !foundDoc3 {
		t.Errorf("expected doc-3 with metadata in results, got %+v", sr.Results)
	}
}

func TestServer_Upload_Validation(t *testing.T) {
	f := newAPIFixture(t)

	rec := f.do(t, "POST", "/upload", uploadRequest{}, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("empty content status = %d, want 400", rec.Code)
	}

	req := httptest.NewRequest("POST", "/upload", bytes.NewReader([]byte("{nope")))
	rec = httptest.NewRecorder()
	f.srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("malformed json status = %d, want 400", rec.Code)
	}
	var env Error
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil || env.Code != ErrCodeBadRequest {
		t.Errorf("malformed json body = %q (err=%v)", rec.Body.String(), err)
	}
}

func TestServer_Upload_SizeLimit(t *testing.T) {
	emb := testutil.NewMockEmbedder(8)
	st, err := store.NewMemoryStore(store.Config{Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	srv, err := NewServer(Config{Store: st, MaxUploadBytes: 64})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	big := uploadRequest{ID: "big", Content: fmt.Sprintf("%0256d", 1)}
	req := httptest.NewRequest("POST", "/upload", bytes.NewReader(mustJSON(t, big)))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized upload status = %d, want 413", rec.Code)
	}
}

func TestServer_Search(t *testing.T) {
	f := newAPIFixture(t)

	rec := f.do(t, "GET", "/search", nil, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing q status = %d, want 400", rec.Code)
	}

	rec = f.do(t, "GET", "/search?q=fox&k=3&min_score=0", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d", rec.Code)
	}
	var sr searchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &sr); err != nil {
		t.Fatalf("decoding search: %v", err)
	}
	if sr.Query != "fox" || sr.Count != len(sr.Results) || len(sr.Results) == 0 || len(sr.Results) > 3 {
		t.Fatalf("unexpected search response: %+v", sr)
	}
	for _, r := range sr.Results {
		if r.ID == "" || r.Content == "" {
			t.Errorf("result missing fields: %+v", r)
		}
	}
}

func TestServer_HybridSearch(t *testing.T) {
	f := newAPIFixture(t)

	rec := f.do(t, "POST", "/hybrid-search", searchRequest{}, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("empty query status = %d, want 400", rec.Code)
	}

	w := 0.7
	rec = f.do(t, "POST", "/hybrid-search", searchRequest{Query: "jungle gorillas", TopK: 4, BM25Weight: &w}, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("hybrid status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var sr searchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &sr); err != nil {
		t.Fatalf("decoding hybrid: %v", err)
	}
	if sr.Count == 0 || len(sr.Results) > 4 {
		t.Fatalf("unexpected hybrid response: %+v", sr)
	}
}
func TestServer_RAG(t *testing.T) {
	f := newAPIFixture(t)

	rec := f.do(t, "POST", "/rag", ragRequest{}, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("empty query status = %d, want 400", rec.Code)
	}

	var resp ragResponse
	rec = f.do(t, "POST", "/rag", ragRequest{Query: "what animal jumps?"}, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("rag status = %d, body=%s", rec.Code, rec.Body.String())
	}
	if resp.Answer == "" || resp.Context == "" {
		t.Errorf("expected answer and context, got %+v", resp)
	}
	if len(resp.Sources) == 0 || len(resp.Sources) > 3 {
		t.Errorf("expected up to 3 sources, got %d", len(resp.Sources))
	}
	if len(resp.Citations) == 0 {
		t.Errorf("expected citations (pipeline configured with WithCitations), got none")
	}
}

func TestServer_RAG_Hybrid(t *testing.T) {
	f := newAPIFixture(t)
	var resp ragResponse
	rec := f.do(t, "POST", "/rag", ragRequest{Query: "jungle animals", Hybrid: true}, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("rag hybrid status = %d, body=%s", rec.Code, rec.Body.String())
	}
	if resp.Query != "jungle animals" {
		t.Errorf("query echoed = %q", resp.Query)
	}
}

func TestServer_RAG_NotConfigured(t *testing.T) {
	emb := testutil.NewMockEmbedder(8)
	st, err := store.NewMemoryStore(store.Config{Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	srv, err := NewServer(Config{Store: st})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	req := httptest.NewRequest("POST", "/rag", bytes.NewReader(mustJSON(t, ragRequest{Query: "x"})))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("rag without pipeline status = %d, want 400", rec.Code)
	}
}

func TestServer_GraphEntity(t *testing.T) {
	f := newAPIFixture(t)

	var resp entityResponse
	rec := f.do(t, "GET", "/graph/alice", nil, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("entity status = %d, body=%s", rec.Code, rec.Body.String())
	}
	if resp.Entity.ID != "alice" || resp.Entity.Label != "Alice" {
		t.Errorf("entity = %+v", resp.Entity)
	}
	if len(resp.Relations) != 1 || resp.Relations[0].Type != "works_at" {
		t.Errorf("relations = %+v, want single works_at", resp.Relations)
	}
	if len(resp.Neighbors) != 1 || resp.Neighbors[0].ID != "acme" {
		t.Errorf("neighbors = %+v, want [acme]", resp.Neighbors)
	}

	// Label lookup fallback.
	rec = f.do(t, "GET", "/graph/Alice", nil, &resp)
	if rec.Code != http.StatusOK || resp.Entity.ID != "alice" {
		t.Errorf("label lookup = (%d, %+v), want alice", rec.Code, resp.Entity)
	}

	// Not found.
	rec = f.do(t, "GET", "/graph/does-not-exist", nil, nil)
	if rec.Code != http.StatusNotFound {
		t.Errorf("missing entity status = %d, want 404", rec.Code)
	}
}

func TestServer_GraphReason(t *testing.T) {
	f := newAPIFixture(t)

	// Natural-language reasoning between two known entities.
	var resp reasonResponse
	rec := f.do(t, "POST", "/graph/reason", reasonRequest{Query: "Alice Paris"}, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("reason status = %d, body=%s", rec.Code, rec.Body.String())
	}
	if len(resp.Inferences) == 0 {
		t.Fatalf("expected inferences, got none")
	}
	inf := resp.Inferences[0]
	if inf.From != "alice" || inf.To != "paris" {
		t.Errorf("inference endpoints = %q -> %q, want alice -> paris", inf.From, inf.To)
	}

	// Path exploration.
	resp = reasonResponse{}
	rec = f.do(t, "POST", "/graph/reason", reasonRequest{From: "alice", To: "paris"}, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("reason paths status = %d, body=%s", rec.Code, rec.Body.String())
	}
	if len(resp.Paths) != 1 {
		t.Fatalf("expected 1 path, got %d", len(resp.Paths))
	}
	p := resp.Paths[0]
	if len(p.Entities) != 3 || p.Entities[0] != "alice" || p.Entities[2] != "paris" {
		t.Errorf("path entities = %+v, want [alice acme paris]", p.Entities)
	}
	if len(p.Relations) != 2 {
		t.Errorf("path relations = %+v, want 2", p.Relations)
	}

	// No arguments.
	rec = f.do(t, "POST", "/graph/reason", reasonRequest{}, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("reason without args status = %d, want 400", rec.Code)
	}
}

func TestServer_OpsEndpoints(t *testing.T) {
	f := newAPIFixture(t)

	rec := f.do(t, "GET", "/healthz", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("healthz status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var health struct {
		OK      bool   `json:"ok"`
		Backend string `json:"backend"`
		Status  string `json:"status"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &health); err != nil {
		t.Fatalf("decoding healthz: %v", err)
	}
	if !health.OK || health.Backend != "memory" {
		t.Errorf("health = %+v, want ok memory backend", health)
	}

	rec = f.do(t, "GET", "/readyz", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("readyz status = %d", rec.Code)
	}

	rec = f.do(t, "GET", "/diagnostics", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("diagnostics status = %d", rec.Code)
	}

	rec = f.do(t, "GET", "/openapi.json", nil, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("openapi status = %d", rec.Code)
	}
	var spec map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &spec); err != nil {
		t.Fatalf("openapi is not valid JSON: %v", err)
	}
	if spec["openapi"] == "" || len(spec["paths"].(map[string]any)) == 0 {
		t.Errorf("openapi spec missing fields: %v", spec["openapi"])
	}
}

func TestServer_AuthEnforced(t *testing.T) {
	emb := testutil.NewMockEmbedder(8)
	st, err := store.NewMemoryStore(store.Config{Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	if err := st.Upload(context.Background(), core.NewDocument("d", "D", "d.txt"), "Some content for auth testing that is long enough to survive the fixed chunker's minimum chunk size."); err != nil {
		t.Fatalf("upload: %v", err)
	}
	srv, err := NewServer(Config{Store: st, Authenticator: NewAPIKeyAuth("sekret")})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	h := srv.Handler()

	// Data endpoint without credentials -> 401.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/search?q=x", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated search status = %d, want 401", rec.Code)
	}

	// Health endpoints remain open.
	for _, path := range []string{"/healthz", "/readyz", "/openapi.json"} {
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		if rec.Code != http.StatusOK {
			t.Errorf("%s should be open, got %d", path, rec.Code)
		}
	}

	// Data endpoint with valid key -> 200.
	req := httptest.NewRequest("GET", "/search?q=x", nil)
	req.Header.Set("X-API-Key", "sekret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("authenticated search status = %d, want 200", rec.Code)
	}
}

func TestServer_AuthJWT(t *testing.T) {
	emb := testutil.NewMockEmbedder(8)
	st, err := store.NewMemoryStore(store.Config{Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	jwt, err := NewJWTAuth(JWTConfig{Secret: "topsecret"})
	if err != nil {
		t.Fatalf("NewJWTAuth: %v", err)
	}
	srv, err := NewServer(Config{Store: st, Authenticator: jwt})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	h := srv.Handler()

	now := time.Now().Unix()
	token := signJWT(t, "topsecret", map[string]any{"sub": "dave", "exp": now + 300})
	req := httptest.NewRequest("GET", "/search?q=x", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("jwt search status = %d, want 200", rec.Code)
	}
}

func TestServer_CORS(t *testing.T) {
	emb := testutil.NewMockEmbedder(8)
	st, err := store.NewMemoryStore(store.Config{Embedder: emb})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	srv, err := NewServer(Config{Store: st, AllowCORS: true})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	h := srv.Handler()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/healthz", nil))
	if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("CORS allow-origin = %q, want *", rec.Header().Get("Access-Control-Allow-Origin"))
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("OPTIONS", "/search", nil))
	if rec.Code != http.StatusNoContent {
		t.Errorf("OPTIONS status = %d, want 204", rec.Code)
	}
}

func TestServer_MethodNotAllowed(t *testing.T) {
	f := newAPIFixture(t)
	rec := f.do(t, "GET", "/upload", nil, nil)
	if rec.Code != http.StatusMethodNotAllowed && rec.Code != http.StatusNotFound {
		t.Errorf("GET /upload status = %d, want 405 or 404", rec.Code)
	}
	rec = f.do(t, "POST", "/search", nil, nil)
	if rec.Code != http.StatusMethodNotAllowed && rec.Code != http.StatusNotFound {
		t.Errorf("POST /search status = %d, want 405 or 404", rec.Code)
	}
}

func TestServer_LiveSocket(t *testing.T) {
	f := newAPIFixture(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	ts := httptest.NewUnstartedServer(f.srv.Handler())
	ts.Listener.Close()
	ts.Listener = ln
	ts.Start()
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("live healthz status = %d, want 200", resp.StatusCode)
	}
}
