package app

import (
	"path/filepath"
	"testing"

	"github.com/deagy/recall/api"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/store"
)

func TestBuildAuthenticator(t *testing.T) {
	// Plain API keys are served by the scoped authenticator with no scope.
	auth, err := BuildAuthenticator(config.AuthConfig{APIKeys: []string{"k1"}})
	if err != nil {
		t.Fatalf("plain keys: %v", err)
	}
	sc, ok := auth.(*api.ScopedAPIKeyAuth)
	if !ok {
		t.Fatalf("expected ScopedAPIKeyAuth, got %T", auth)
	}
	if ns := sc.Namespaces("k1"); ns != nil {
		t.Errorf("plain key should be unrestricted, got %v", ns)
	}

	// Scoped keys keep their scope end to end.
	scoped := config.ScopedKeyConfig{Key: "team", Namespaces: []string{"ns-a"}}
	auth, err = BuildAuthenticator(config.AuthConfig{ScopedKeys: []config.ScopedKeyConfig{scoped}})
	if err != nil {
		t.Fatalf("scoped keys: %v", err)
	}
	sc, ok = auth.(*api.ScopedAPIKeyAuth)
	if !ok {
		t.Fatalf("expected ScopedAPIKeyAuth, got %T", auth)
	}
	if ns := sc.Namespaces("team"); len(ns) != 1 || ns[0] != "ns-a" {
		t.Errorf("scoped key lost its scope: %v", ns)
	}

	// Scoped keys + JWT compose, and the composite still exposes the scope.
	auth, err = BuildAuthenticator(config.AuthConfig{
		ScopedKeys: []config.ScopedKeyConfig{scoped},
		JWTSecret:  "s3cret",
	})
	if err != nil {
		t.Fatalf("scoped + jwt: %v", err)
	}
	comp, ok := auth.(*api.Composite)
	if !ok {
		t.Fatalf("expected Composite, got %T", auth)
	}
	if ns := comp.Namespaces("team"); len(ns) != 1 || ns[0] != "ns-a" {
		t.Errorf("composite lost the key scope: %v", ns)
	}

	// JWT alone stays a plain JWTAuth.
	auth, err = BuildAuthenticator(config.AuthConfig{JWTSecret: "s3cret"})
	if err != nil {
		t.Fatalf("jwt only: %v", err)
	}
	if _, ok := auth.(*api.JWTAuth); !ok {
		t.Errorf("expected JWTAuth, got %T", auth)
	}

	// No credentials is an error.
	if _, err := BuildAuthenticator(config.AuthConfig{}); err == nil {
		t.Error("expected error when no credentials are configured")
	}
}

func TestBuildEmbedder(t *testing.T) {
	t.Setenv("TEST_RECALL_KEY", "sk-test")

	emb, err := BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderMock, Dimension: 8})
	if err != nil || emb == nil {
		t.Fatalf("mock: %v", err)
	}

	emb, err = BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderOpenAI, Model: "m", Dimension: 8, APIKeyEnv: "TEST_RECALL_KEY"})
	if err != nil || emb == nil {
		t.Fatalf("openai: %v", err)
	}
	if _, err := BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderOpenAI, Model: "m", Dimension: 8, APIKeyEnv: "TEST_RECALL_UNSET_VAR"}); err == nil {
		t.Error("openai without key env: expected error")
	}
	if _, err := BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderOpenAI, Model: "m", Dimension: 8}); err == nil {
		t.Error("openai without api_key_env: expected error")
	}

	emb, err = BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderCohere, Model: "m", Dimension: 8, APIKeyEnv: "TEST_RECALL_KEY"})
	if err != nil || emb == nil {
		t.Fatalf("cohere: %v", err)
	}
	if _, err := BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderCohere, Model: "m", Dimension: 8, APIKeyEnv: "TEST_RECALL_UNSET_VAR"}); err == nil {
		t.Error("cohere without key env: expected error")
	}

	emb, err = BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderOllama, Model: "nomic-embed-text"})
	if err != nil || emb == nil {
		t.Fatalf("ollama: %v", err)
	}

	if _, err := BuildEmbedder(config.EmbedderConfig{Type: config.EmbedderONNX, Path: "/tmp/model.onnx"}); err == nil {
		t.Error("onnx: expected error (tokenizer cannot be expressed in config)")
	}
	if _, err := BuildEmbedder(config.EmbedderConfig{Type: "nope"}); err == nil {
		t.Error("unknown type: expected error")
	}
}

func TestBuildStore(t *testing.T) {
	base := config.StoreConfig{
		Backend:   config.BackendMemory,
		Namespace: "ns",
		Embedder:  config.EmbedderConfig{Type: config.EmbedderMock, Dimension: 8},
		Chunking:  config.ChunkingConfig{Strategy: config.ChunkingRecursive, MaxTokens: 64, Overlap: 8},
	}

	st, err := BuildStore(base)
	if err != nil {
		t.Fatalf("memory: %v", err)
	}
	defer st.Close()
	mem, ok := st.(*store.MemoryStore)
	if !ok {
		t.Fatalf("expected MemoryStore, got %T", st)
	}
	if mem.Namespace() != "ns" {
		t.Errorf("namespace = %q, want ns", mem.Namespace())
	}

	sql := base
	sql.Backend = config.BackendSQLite
	sql.Path = filepath.Join(t.TempDir(), "test.db")
	st, err = BuildStore(sql)
	if err != nil {
		t.Fatalf("sqlite: %v", err)
	}
	defer st.Close()
	if _, ok := st.(*store.SQLiteStore); !ok {
		t.Errorf("expected SQLiteStore, got %T", st)
	}

	noPath := sql
	noPath.Path = ""
	if _, err := BuildStore(noPath); err == nil {
		t.Error("sqlite without path: expected error")
	}

	bad := base
	bad.Backend = "bogus"
	if _, err := BuildStore(bad); err == nil {
		t.Error("unknown backend: expected error")
	}

	noKey := base
	noKey.Embedder = config.EmbedderConfig{Type: config.EmbedderOpenAI, Model: "m", Dimension: 8, APIKeyEnv: "TEST_RECALL_UNSET_VAR"}
	if _, err := BuildStore(noKey); err == nil {
		t.Error("unresolvable embedder: expected error")
	}
}

func TestBuildService(t *testing.T) {
	cfg := &config.Config{}
	cfg.WithDefaults() // memory backend, mock embedder

	svc, cleanup, err := BuildService(cfg)
	if err != nil {
		t.Fatalf("BuildService: %v", err)
	}
	defer cleanup()
	if svc.Store == nil || svc.Pipeline == nil || svc.Graph == nil || svc.Reasoner == nil {
		t.Fatalf("incomplete service: %+v", svc)
	}
}

func TestBuildAPIServer(t *testing.T) {
	cfg := &config.Config{}
	cfg.WithDefaults()

	srv, cleanup, err := BuildAPIServer(cfg)
	if err != nil {
		t.Fatalf("BuildAPIServer: %v", err)
	}
	defer cleanup()
	if srv.Handler() == nil {
		t.Fatal("expected non-nil handler")
	}
	if srv.Addr() != "127.0.0.1:8080" {
		t.Errorf("addr = %q, want 127.0.0.1:8080", srv.Addr())
	}
}

func TestBuildAPIServer_Auth(t *testing.T) {
	cfg := &config.Config{}
	cfg.WithDefaults()
	cfg.Auth.Enabled = true
	cfg.Auth.APIKeys = []string{"k1"}
	cfg.Auth.ScopedKeys = []config.ScopedKeyConfig{{Key: "team", Namespaces: []string{"ns-a"}}}
	cfg.Auth.JWTSecret = "s3cret"

	srv, cleanup, err := BuildAPIServer(cfg)
	if err != nil {
		t.Fatalf("BuildAPIServer with auth: %v", err)
	}
	defer cleanup()
	if srv.Handler() == nil {
		t.Fatal("expected non-nil handler")
	}
}

func TestBuildAPIServer_Errors(t *testing.T) {
	cfg := &config.Config{}
	cfg.WithDefaults()
	cfg.Store.Backend = "bogus"
	if _, _, err := BuildAPIServer(cfg); err == nil {
		t.Error("invalid backend: expected error")
	}

	cfg = &config.Config{}
	cfg.WithDefaults()
	cfg.Auth.Enabled = true // no credentials
	if _, _, err := BuildAPIServer(cfg); err == nil {
		t.Error("auth without credentials: expected error")
	}
}
