// Command recall-server runs the Recall REST API as a standalone service.
//
// Usage:
//
//	recall-server [-config /etc/recall/recall.yaml]
//
// Without -config it starts with defaults: an in-memory store with a
// deterministic mock embedder, listening on 127.0.0.1:8080 — handy for a
// local smoke test. In production, pass a config file (JSON or YAML);
// environment variables (RECALL__SECTION__KEY) override file values. The
// server supports graceful shutdown on SIGINT/SIGTERM.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/deagy/recall/api"
	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

func main() {
	configPath := flag.String("config", "", "path to a JSON or YAML config file (default: built-in dev defaults)")
	probeURL := flag.String("probe-url", "http://127.0.0.1:8080/healthz", "URL probed in health-probe mode")
	healthProbe := flag.Bool("health-probe", false, "probe the local health endpoint and exit (0 = healthy, 1 = unhealthy); for container HEALTHCHECK on images without curl")
	flag.Parse()

	if *healthProbe {
		os.Exit(runHealthProbe(*probeURL))
	}

	var cfg *config.Config
	if *configPath != "" {
		var err error
		cfg, err = config.Load(*configPath)
		if err != nil {
			log.Fatalf("loading config: %v", err)
		}
		cfg.ApplyEnv("")
		if err := cfg.Validate(); err != nil {
			log.Fatalf("config: %v", err)
		}
		log.Printf("loaded config from %s", *configPath)
	} else {
		cfg = &config.Config{}
		cfg.WithDefaults()
		log.Printf("no -config flag; using dev defaults (%s, in-memory store, mock embedder)", cfg.Server.Host)
	}

	srv, cleanup, err := buildServer(cfg)
	if err != nil {
		log.Fatalf("building server: %v", err)
	}
	defer cleanup()

	// Graceful shutdown on SIGINT/SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()

	select {
	case err := <-errCh:
		if err != nil {
			log.Fatalf("server: %v", err)
		}
	case <-ctx.Done():
		log.Printf("shutting down...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Printf("graceful shutdown: %v", err)
		}
	}
	log.Printf("stopped")
}

// runHealthProbe performs a GET against url and returns 0 for a 2xx
// response, 1 otherwise. It is used by the container HEALTHCHECK on
// distroless images that ship no curl or wget.
func runHealthProbe(url string) int {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 1
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return 0
	}
	return 1
}

// buildServer assembles the full service (store, embedder, pipeline,
// graph, HTTP server) from the configuration. The returned cleanup
// function releases backend resources.
func buildServer(cfg *config.Config) (*api.Server, func(), error) {
	emb, err := buildEmbedder(cfg.Store.Embedder)
	if err != nil {
		return nil, nil, fmt.Errorf("embedder: %w", err)
	}

	var st store.Store
	switch cfg.Store.Backend {
	case config.BackendSQLite:
		st, err = store.NewSQLiteStore(store.Config{
			Namespace:      cfg.Store.Namespace,
			Embedder:       emb,
			ChunkerFactory: chunkerFactory(cfg.Store.Chunking),
		}, cfg.Store.Path)
	default:
		st, err = store.NewMemoryStore(store.Config{
			Namespace:      cfg.Store.Namespace,
			Embedder:       emb,
			ChunkerFactory: chunkerFactory(cfg.Store.Chunking),
		})
	}
	if err != nil {
		return nil, nil, fmt.Errorf("store: %w", err)
	}

	gs := store.NewMemoryGraphStore()
	pipe := pipeline.NewRAGPipeline(st, nil).WithCitations()
	engine := reasoning.NewEngine(gs.Graph(), reasoning.DefaultConfig())

	apiCfg := api.Config{
		Store:          st,
		Pipeline:       pipe,
		Graph:          gs,
		Reasoner:       engine,
		Host:           cfg.Server.Host,
		Port:           cfg.Server.Port,
		MaxUploadBytes: cfg.Server.MaxUploadBytes,
		ReadTimeout:    cfg.Server.ReadTimeout.AsDuration(),
		WriteTimeout:   cfg.Server.WriteTimeout.AsDuration(),
		IdleTimeout:    cfg.Server.IdleTimeout.AsDuration(),
		AllowCORS:      cfg.Server.AllowCORS,
	}
	if cfg.Auth.Enabled {
		apiCfg.Authenticator, err = buildAuthenticator(cfg.Auth)
		if err != nil {
			_ = st.Close()
			return nil, nil, fmt.Errorf("auth: %w", err)
		}
	}

	srv, err := api.NewServer(apiCfg)
	if err != nil {
		_ = st.Close()
		return nil, nil, err
	}
	return srv, func() { _ = st.Close() }, nil
}

// buildEmbedder constructs the configured embedding provider.
func buildEmbedder(e config.EmbedderConfig) (embedder.Embedder, error) {
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
		return nil, fmt.Errorf("onnx embedder is not supported by recall-server (a tokenizer is required); use the library API")
	default:
		return nil, fmt.Errorf("unknown embedder type %q", e.Type)
	}
}

// envKey reads a secret from the named environment variable.
func envKey(name string) (string, error) {
	v := os.Getenv(name)
	if v == "" {
		return "", fmt.Errorf("environment variable %s is not set", name)
	}
	return v, nil
}

// buildAuthenticator assembles the API authenticator from config.
func buildAuthenticator(a config.AuthConfig) (api.Authenticator, error) {
	var auths []api.Authenticator
	if len(a.APIKeys) > 0 {
		auths = append(auths, api.NewAPIKeyAuth(a.APIKeys...))
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
		return nil, fmt.Errorf("auth enabled but no credentials configured")
	}
	if len(auths) == 1 {
		return auths[0], nil
	}
	return api.NewComposite(auths...), nil
}

// chunkerFactory maps the configured strategy to a chunker factory.
func chunkerFactory(k config.ChunkingConfig) chunker.Factory {
	switch k.Strategy {
	case config.ChunkingRecursive:
		return chunker.NewRecursive
	default:
		return chunker.NewFixed
	}
}
