package api

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

// Config configures a Server.
type Config struct {
	// Store is the knowledge store backing the API (required).
	Store store.Store

	// Pipeline is the RAG pipeline for POST /rag. Optional: when nil the
	// /rag endpoint returns 400.
	Pipeline *pipeline.RAGPipeline

	// Graph is the knowledge graph store for /graph endpoints. Optional.
	Graph store.GraphStore

	// Reasoner is the multi-hop reasoning engine for POST /graph/reason.
	// Optional: when nil the endpoint returns 400.
	Reasoner *reasoning.Engine

	// Authenticator, when non-nil, protects all endpoints except the
	// health/readiness/openapi paths (see ExemptPaths).
	Authenticator Authenticator

	// ExemptPaths are path prefixes served without authentication, in
	// addition to the defaults (/healthz, /readyz, /diagnostics,
	// /openapi.json).
	ExemptPaths []string

	// MaxUploadBytes caps the request body size for POST /upload.
	// Defaults to 10 MiB.
	MaxUploadBytes int64

	// Host is the listen address host (used by ListenAndServe). Defaults
	// to "127.0.0.1".
	Host string

	// Port is the listen port (used by ListenAndServe). Defaults to 8080.
	Port int

	// ReadTimeout bounds reading the entire request. Defaults to 30s.
	ReadTimeout time.Duration

	// WriteTimeout bounds writing the response. Defaults to 60s.
	WriteTimeout time.Duration

	// IdleTimeout bounds keep-alive connections. Defaults to 120s.
	IdleTimeout time.Duration

	// AllowCORS enables permissive CORS headers on all responses.
	AllowCORS bool
}

// defaultExempt are the unauthenticated path prefixes by default.
var defaultExempt = []string{"/healthz", "/readyz", "/diagnostics", "/openapi.json"}

// Server is a Recall HTTP API server.
type Server struct {
	cfg     Config
	handler http.Handler
	httpSrv *http.Server
}

// NewServer creates a new Server, validating the configuration and wiring
// the HTTP handler (see Handler).
func NewServer(cfg Config) (*Server, error) {
	if cfg.Store == nil {
		return nil, errors.New("api: Store is required")
	}
	if cfg.MaxUploadBytes <= 0 {
		cfg.MaxUploadBytes = 10 << 20 // 10 MiB
	}
	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.Port <= 0 || cfg.Port > 65535 {
		cfg.Port = 8080
	}
	if cfg.ReadTimeout <= 0 {
		cfg.ReadTimeout = 30 * time.Second
	}
	if cfg.WriteTimeout <= 0 {
		cfg.WriteTimeout = 60 * time.Second
	}
	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = 120 * time.Second
	}
	cfg.ExemptPaths = append(append([]string{}, defaultExempt...), cfg.ExemptPaths...)

	s := &Server{cfg: cfg}
	s.handler = s.buildHandler()
	return s, nil
}

// Handler returns the http.Handler implementing the API. It can be mounted
// in any http.ServeMux or served directly.
func (s *Server) Handler() http.Handler { return s.handler }

// Addr returns the listen address ("host:port") used by ListenAndServe.
func (s *Server) Addr() string {
	return net.JoinHostPort(s.cfg.Host, strconv.Itoa(s.cfg.Port))
}

// ListenAndServe starts the HTTP server on the configured address and
// blocks until the server is stopped via Shutdown or a fatal network error
// occurs. It returns http.ErrServerClosed after a clean Shutdown.
func (s *Server) ListenAndServe() error {
	s.httpSrv = &http.Server{
		Addr:         s.Addr(),
		Handler:      s.handler,
		ReadTimeout:  s.cfg.ReadTimeout,
		WriteTimeout: s.cfg.WriteTimeout,
		IdleTimeout:  s.cfg.IdleTimeout,
	}
	if err := s.httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("api server: %w", err)
	}
	return nil
}

// ListenAndServeTLS starts the TLS HTTP server with the given key and
// certificate files (PEM).
func (s *Server) ListenAndServeTLS(certFile, keyFile string) error {
	s.httpSrv = &http.Server{
		Addr:         s.Addr(),
		Handler:      s.handler,
		ReadTimeout:  s.cfg.ReadTimeout,
		WriteTimeout: s.cfg.WriteTimeout,
		IdleTimeout:  s.cfg.IdleTimeout,
	}
	if err := s.httpSrv.ListenAndServeTLS(certFile, keyFile); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("api server: %w", err)
	}
	return nil
}

// Shutdown gracefully stops the server, waiting for in-flight requests up
// to the given context deadline. It is a no-op when the server was never
// started.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.httpSrv == nil {
		return nil
	}
	return s.httpSrv.Shutdown(ctx)
}

// NewAPIKey generates a cryptographically random API key (48 base62-ish
// URL-safe characters) suitable for seeding Config.Authenticator.
func NewAPIKey() (string, error) {
	const n = 32
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generating api key: %w", err)
	}
	return encodeKey(b), nil
}

func encodeKey(b []byte) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	out := make([]byte, len(b))
	for i, v := range b {
		out[i] = alphabet[v%byte(len(alphabet))]
	}
	return string(out)
}
