// Package app assembles Recall service components (embedder, store, RAG
// pipeline, knowledge graph, reasoning engine, and the REST API server) from
// a config.Config. Both the recall-server and recall CLI entry points build
// their stacks through this package, so the two always agree on how a
// configuration maps to components.
package app

import (
	"errors"
	"fmt"
	"os"

	"github.com/deagy/recall/api"
	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

// Service is the assembled set of core components a Recall deployment runs:
// the store plus the RAG pipeline, knowledge graph, and reasoning engine
// built on top of it.
type Service struct {
	// Store is the knowledge store (memory or SQLite).
	Store store.Store

	// Pipeline is the RAG pipeline over Store (citations enabled).
	Pipeline *pipeline.RAGPipeline

	// Graph is the knowledge graph store.
	Graph store.GraphStore

	// Reasoner is the multi-hop reasoning engine over Graph.
	Reasoner *reasoning.Engine
}

// BuildEmbedder constructs the configured embedding provider. API keys are
// never read from the config file: APIKeyEnv names the environment variable
// that holds the key.
func BuildEmbedder(e config.EmbedderConfig) (embedder.Embedder, error) {
	switch e.Type {
	case config.EmbedderMock:
		return embedder.NewMockEmbedder(e.Dimension), nil
	case config.EmbedderOpenAI:
		key, err := envKey(e.APIKeyEnv)
		if err != nil {
			return nil, err
		}
		return embedder.NewOpenAIEmbedder(embedder.OpenAIConfig{
			APIKey:    key,
			Model:     e.Model,
			BaseURL:   e.BaseURL,
			Dimension: e.Dimension,
		})
	case config.EmbedderCohere:
		key, err := envKey(e.APIKeyEnv)
		if err != nil {
			return nil, err
		}
		return embedder.NewCohereEmbedder(embedder.CohereConfig{
			APIKey:    key,
			Model:     e.Model,
			BaseURL:   e.BaseURL,
			Dimension: e.Dimension,
		})
	case config.EmbedderOllama:
		return embedder.NewOllamaEmbedder(embedder.OllamaConfig{
			Model:     e.Model,
			BaseURL:   e.BaseURL,
			Dimension: e.Dimension,
		})
	case config.EmbedderONNX:
		// The ONNX embedder requires a tokenizer that cannot be
		// expressed in the flat config schema; use the library API
		// (embedder.NewOnnxEmbedder) for that path.
		return nil, fmt.Errorf("onnx embedder is not supported by configuration (a tokenizer is required); use the library API")
	default:
		return nil, fmt.Errorf("unknown embedder type %q", e.Type)
	}
}

// ChunkerFactory maps the configured strategy to a chunker factory.
//
// The returned factory ignores the chunker.Config it is handed and builds from
// the store's own ChunkingConfig. The stores invoke their factory with
// chunker.DefaultConfig(), so anything read from the caller's argument would
// discard the configured max_tokens, overlap and min_chunk_size.
func ChunkerFactory(k config.ChunkingConfig) chunker.Factory {
	cc := chunker.Config{
		MaxTokens:     k.MaxTokens,
		OverlapTokens: k.Overlap,
		MinChunkSize:  k.MinChunkSize,
	}
	return func(chunker.Config) chunker.Chunker {
		switch k.Strategy {
		case config.ChunkingRecursive:
			return chunker.NewRecursive(cc)
		case config.ChunkingDocumentAware:
			// The boundary is the intended chunk edge, so the inner chunker
			// only subdivides sections that exceed MaxTokens.
			da := chunker.NewDocumentAware(chunker.NewFixed(cc))
			da.Boundary = k.Boundary
			return da
		default:
			return chunker.NewFixed(cc)
		}
	}
}

// BuildStore opens the store described by sc ("memory" or "sqlite"). The
// sqlite backend requires sc.Path to be set.
func BuildStore(sc config.StoreConfig) (store.Store, error) {
	emb, err := BuildEmbedder(sc.Embedder)
	if err != nil {
		return nil, fmt.Errorf("embedder: %w", err)
	}
	cfg := store.Config{
		Namespace:      sc.Namespace,
		Embedder:       emb,
		ChunkerFactory: ChunkerFactory(sc.Chunking),
	}
	switch sc.Backend {
	case config.BackendSQLite:
		if sc.Path == "" {
			return nil, errors.New("store: sqlite backend requires store.path")
		}
		return store.NewSQLiteStore(cfg, sc.Path)
	case config.BackendMemory, "":
		return store.NewMemoryStore(cfg)
	default:
		return nil, fmt.Errorf("store: unknown backend %q (want %q or %q)", sc.Backend, config.BackendMemory, config.BackendSQLite)
	}
}

// BuildAuthenticator assembles the API authenticator from config. API keys
// (plain and scoped) are always served by a single api.ScopedAPIKeyAuth so
// per-key namespace scopes are enforced; JWT auth is composed alongside it
// when configured.
func BuildAuthenticator(a config.AuthConfig) (api.Authenticator, error) {
	var auths []api.Authenticator
	if len(a.APIKeys) > 0 || len(a.ScopedKeys) > 0 {
		specs := make([]api.KeySpec, 0, len(a.APIKeys)+len(a.ScopedKeys))
		for _, k := range a.APIKeys {
			specs = append(specs, api.KeySpec{Key: k})
		}
		for _, sk := range a.ScopedKeys {
			specs = append(specs, api.KeySpec{Key: sk.Key, Namespaces: sk.Namespaces})
		}
		auths = append(auths, api.NewScopedAPIKeyAuth(specs...))
	}
	if a.JWTSecret != "" {
		jwt, err := api.NewJWTAuth(api.JWTConfig{
			Secret:   a.JWTSecret,
			Issuer:   a.JWTIssuer,
			Audience: a.JWTAudience,
		})
		if err != nil {
			return nil, err
		}
		auths = append(auths, jwt)
	}
	if len(auths) == 0 {
		return nil, errors.New("auth enabled but no credentials configured")
	}
	if len(auths) == 1 {
		return auths[0], nil
	}
	return api.NewComposite(auths...), nil
}

// BuildService assembles the core service stack (store, RAG pipeline,
// knowledge graph, reasoning engine) from cfg. The returned cleanup
// function releases backend resources.
func BuildService(cfg *config.Config) (*Service, func(), error) {
	st, err := BuildStore(cfg.Store)
	if err != nil {
		return nil, nil, fmt.Errorf("store: %w", err)
	}
	gs := store.NewMemoryGraphStore()
	pipe := pipeline.NewRAGPipeline(st, nil).WithCitations()
	engine := reasoning.NewEngine(gs.Graph(), reasoning.DefaultConfig())
	return &Service{
		Store:    st,
		Pipeline: pipe,
		Graph:    gs,
		Reasoner: engine,
	}, func() { _ = st.Close() }, nil
}

// BuildAPIServer assembles the full REST API service from cfg (store,
// pipeline, graph, reasoner, and HTTP server configuration, plus the
// authenticator when auth is enabled). The returned cleanup function
// releases backend resources.
func BuildAPIServer(cfg *config.Config) (*api.Server, func(), error) {
	svc, cleanup, err := BuildService(cfg)
	if err != nil {
		return nil, nil, err
	}

	apiCfg := api.Config{
		Store:          svc.Store,
		Pipeline:       svc.Pipeline,
		Graph:          svc.Graph,
		Reasoner:       svc.Reasoner,
		Host:           cfg.Server.Host,
		Port:           cfg.Server.Port,
		MaxUploadBytes: cfg.Server.MaxUploadBytes,
		ReadTimeout:    cfg.Server.ReadTimeout.AsDuration(),
		WriteTimeout:   cfg.Server.WriteTimeout.AsDuration(),
		IdleTimeout:    cfg.Server.IdleTimeout.AsDuration(),
		AllowCORS:      cfg.Server.AllowCORS,
	}
	if cfg.Auth.Enabled {
		auth, err := BuildAuthenticator(cfg.Auth)
		if err != nil {
			cleanup()
			return nil, nil, fmt.Errorf("auth: %w", err)
		}
		apiCfg.Authenticator = auth
	}

	srv, err := api.NewServer(apiCfg)
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	return srv, cleanup, nil
}

// envKey reads a secret from the named environment variable.
func envKey(name string) (string, error) {
	if name == "" {
		return "", errors.New("embedder: api_key_env is not set")
	}
	v := os.Getenv(name)
	if v == "" {
		return "", fmt.Errorf("environment variable %s is not set", name)
	}
	return v, nil
}
