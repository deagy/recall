package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/deagy/recall/api"
	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

// runCLI executes the recall root command with args and returns the captured
// stdout plus the execution error.
func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	root := newRootCmd("test")
	var stdout, stderr bytes.Buffer
	root.SetOut(&stdout)
	root.SetErr(&stderr)
	root.SetArgs(args)
	err := root.Execute()
	return stdout.String(), err
}

// mustRunCLI runs the CLI and fails the test on error.
func mustRunCLI(t *testing.T, args ...string) string {
	t.Helper()
	out, err := runCLI(t, args...)
	if err != nil {
		t.Fatalf("recall %v: %v\noutput: %s", args, err, out)
	}
	return out
}

// writeSQLiteConfig writes a config file for a SQLite-backed local store in
// a temp directory and returns (config path, database path).
func writeSQLiteConfig(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "recall.db")
	cfg := &config.Config{}
	cfg.Store.Backend = config.BackendSQLite
	cfg.Store.Path = dbPath
	cfg.Store.Embedder.Dimension = 8
	cfg.Store.Chunking.MaxTokens = 40
	cfg.Store.Chunking.Overlap = 5
	cfgPath := filepath.Join(dir, "recall.json")
	if err := config.Save(cfgPath, cfg); err != nil {
		t.Fatalf("saving config: %v", err)
	}
	return cfgPath, dbPath
}

// writeNotesFile writes a small ingestible text file and returns its path.
func writeNotesFile(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "notes.txt")
	content := "Go is a compiled language with garbage collection and goroutines. " +
		"It was designed at Google. Rust is a systems language with ownership."
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing notes file: %v", err)
	}
	return path
}

// decodeJSON unmarshals CLI JSON output into out.
func decodeJSON(t *testing.T, data string, out any) {
	t.Helper()
	if err := json.Unmarshal([]byte(data), out); err != nil {
		t.Fatalf("decoding JSON output: %v\noutput: %s", err, data)
	}
}

// seedSQLiteGraph seeds the SQLite graph store at dbPath with a small
// transitive chain: Alice works_at Acme, Acme located_in Berlin.
func seedSQLiteGraph(t *testing.T, dbPath string) {
	t.Helper()
	gs, err := store.NewSQLiteGraphStore(dbPath)
	if err != nil {
		t.Fatalf("graph store: %v", err)
	}
	defer gs.Close()
	if err := gs.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson)); err != nil {
		t.Fatalf("AddEntity alice: %v", err)
	}
	if err := gs.AddEntity(graph.NewEntity("acme", "Acme", graph.EntityOrganizer)); err != nil {
		t.Fatalf("AddEntity acme: %v", err)
	}
	if err := gs.AddEntity(graph.NewEntity("berlin", "Berlin", graph.EntityLocation)); err != nil {
		t.Fatalf("AddEntity berlin: %v", err)
	}
	if err := gs.AddRelation(graph.NewRelation("alice", "acme", "works_at", 1)); err != nil {
		t.Fatalf("AddRelation works_at: %v", err)
	}
	if err := gs.AddRelation(graph.NewRelation("acme", "berlin", "located_in", 1)); err != nil {
		t.Fatalf("AddRelation located_in: %v", err)
	}
}

// newTestAPIServer starts a recall-server API backed by an in-memory store
// with a mock embedder. With opts.graph the graph is seeded with the
// Alice/Acme/Berlin chain; with opts.auth every data endpoint requires the
// "test-key" API key.
func newTestAPIServer(t *testing.T, opts struct{ auth, graph bool }) *httptest.Server {
	t.Helper()
	st, err := store.NewMemoryStore(store.Config{
		Namespace:      "default",
		Embedder:       embedder.NewMockEmbedder(8),
		ChunkerFactory: chunker.NewFixed,
	})
	if err != nil {
		t.Fatalf("memory store: %v", err)
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
		g := gs.Graph()
		g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
		g.AddEntity(graph.NewEntity("acme", "Acme", graph.EntityOrganizer))
		g.AddEntity(graph.NewEntity("berlin", "Berlin", graph.EntityLocation))
		g.AddRelation(graph.NewRelation("alice", "acme", "works_at", 1))
		g.AddRelation(graph.NewRelation("acme", "berlin", "located_in", 1))
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
	return ts
}
