// Package config provides declarative configuration for running Recall as a
// service or through the command-line client. Configurations load from JSON
// or YAML files, can be overridden by environment variables
// (RECALL__SECTION__KEY), are validated before use, and can be watched for
// changes (hot reload).
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Backend identifiers for StoreConfig.Backend.
const (
	BackendMemory = "memory"
	BackendSQLite = "sqlite"
)

// Embedder type identifiers for EmbedderConfig.Type.
const (
	EmbedderMock   = "mock"
	EmbedderOpenAI = "openai"
	EmbedderCohere = "cohere"
	EmbedderOllama = "ollama"
	EmbedderONNX   = "onnx"
)

// Chunking strategy identifiers for ChunkingConfig.Strategy.
const (
	ChunkingFixed     = "fixed"
	ChunkingRecursive = "recursive"
	// ChunkingDocumentAware splits on an explicit boundary marker first and
	// chunks each section independently, so no chunk spans two sections. Use
	// it for content whose sections are the intended retrieval unit -- chat
	// exports where each turn is a section, for example.
	ChunkingDocumentAware = "document_aware"
)

// Duration is a time.Duration that marshals as a string ("30s") in both
// JSON and YAML.
type Duration time.Duration

// AsDuration converts to a time.Duration.
func (d Duration) AsDuration() time.Duration { return time.Duration(d) }

// MarshalText renders the duration as a Go duration string.
func (d Duration) MarshalText() ([]byte, error) {
	return []byte(time.Duration(d).String()), nil
}

// MarshalJSON renders the duration as a quoted Go duration string.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

// UnmarshalText parses a Go duration string ("30s", "5m").
func (d *Duration) UnmarshalText(b []byte) error {
	v, err := time.ParseDuration(string(b))
	if err != nil {
		return fmt.Errorf("parsing duration %q: %w", string(b), err)
	}
	*d = Duration(v)
	return nil
}

// UnmarshalJSON parses a quoted Go duration string.
func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return d.UnmarshalText(b)
	}
	return d.UnmarshalText([]byte(s))
}

// Config is the top-level service configuration.
type Config struct {
	// Server configures the HTTP API server.
	Server ServerConfig `json:"server" yaml:"server"`

	// Store configures the knowledge store backend.
	Store StoreConfig `json:"store" yaml:"store"`

	// Auth configures API authentication.
	Auth AuthConfig `json:"auth" yaml:"auth"`

	// CLI configures the recall command-line client. The server ignores
	// this section.
	CLI CLIConfig `json:"cli" yaml:"cli"`
}

// ServerConfig configures the HTTP API server.
type ServerConfig struct {
	// Host is the listen address host. Defaults to "127.0.0.1".
	Host string `json:"host" yaml:"host"`

	// Port is the listen port. Defaults to 8080.
	Port int `json:"port" yaml:"port"`

	// MaxUploadBytes caps upload request bodies. Defaults to 10 MiB.
	MaxUploadBytes int64 `json:"max_upload_bytes" yaml:"max_upload_bytes"`

	// ReadTimeout bounds reading an entire request. Defaults to 30s.
	ReadTimeout Duration `json:"read_timeout" yaml:"read_timeout"`

	// WriteTimeout bounds writing a response. Defaults to 60s.
	WriteTimeout Duration `json:"write_timeout" yaml:"write_timeout"`

	// IdleTimeout bounds keep-alive connections. Defaults to 120s.
	IdleTimeout Duration `json:"idle_timeout" yaml:"idle_timeout"`

	// AllowCORS enables permissive CORS headers.
	AllowCORS bool `json:"allow_cors" yaml:"allow_cors"`
}

// StoreConfig configures the knowledge store.
type StoreConfig struct {
	// Backend is "memory" or "sqlite". Defaults to "memory".
	Backend string `json:"backend" yaml:"backend"`

	// Path is the SQLite database file (required for backend "sqlite").
	Path string `json:"path" yaml:"path"`

	// Namespace is the default store namespace. Defaults to "default".
	Namespace string `json:"namespace" yaml:"namespace"`

	// Embedder configures the embedding provider.
	Embedder EmbedderConfig `json:"embedder" yaml:"embedder"`

	// Chunking configures document chunking.
	Chunking ChunkingConfig `json:"chunking" yaml:"chunking"`
}

// EmbedderConfig configures the embedding provider.
type EmbedderConfig struct {
	// Type is "mock", "openai", "cohere", "ollama", or "onnx".
	// Defaults to "mock".
	Type string `json:"type" yaml:"type"`

	// Model is the embedding model name (provider-specific).
	Model string `json:"model" yaml:"model"`

	// Dimension is the embedding dimension. Defaults to 384 for mock;
	// providers validate against their known model dimensions.
	Dimension int `json:"dimension" yaml:"dimension"`

	// APIKeyEnv is the name of the environment variable holding the API
	// key for openai/cohere providers. Keys are never stored in config
	// files.
	APIKeyEnv string `json:"api_key_env" yaml:"api_key_env"`

	// BaseURL overrides the provider endpoint (openai-compatible and
	// ollama providers).
	BaseURL string `json:"base_url" yaml:"base_url"`

	// Path is the local model file for the onnx provider.
	Path string `json:"path" yaml:"path"`
}

// ChunkingConfig configures document chunking.
type ChunkingConfig struct {
	// Strategy is "fixed" or "recursive". Defaults to "fixed".
	Strategy string `json:"strategy" yaml:"strategy"`

	// MaxTokens is the maximum tokens per chunk. Defaults to 512.
	MaxTokens int `json:"max_tokens" yaml:"max_tokens"`

	// Overlap is the overlap in tokens between adjacent chunks.
	// Defaults to 50.
	Overlap int `json:"overlap" yaml:"overlap"`

	// MinChunkSize is the minimum characters a chunk may have. A chunk below
	// it that cannot be combined with neighbouring text is discarded, so on a
	// boundary-split strategy this silently drops whole short sections.
	// Defaults to 50 for "fixed" and "recursive", and to 0 for
	// "document_aware", where every section is meant to survive.
	// Set explicitly to override either default.
	MinChunkSize int `json:"min_chunk_size" yaml:"min_chunk_size"`

	// Boundary is the section separator for "document_aware". Empty means the
	// chunker default ("\n---\n"). Choose a marker the content cannot contain:
	// prose and chat transcripts routinely include "---".
	Boundary string `json:"boundary" yaml:"boundary"`
}

// AuthConfig configures API authentication.
type AuthConfig struct {
	// Enabled turns on authentication for data endpoints.
	Enabled bool `json:"enabled" yaml:"enabled"`

	// APIKeys are static API keys accepted via X-API-Key or Bearer.
	APIKeys []string `json:"api_keys" yaml:"api_keys"`

	// ScopedKeys are API keys with per-key namespace restrictions (see
	// api.ScopedAPIKeyAuth): a key with non-empty Namespaces may only
	// access those namespaces. Keys listed in APIKeys remain unrestricted.
	ScopedKeys []ScopedKeyConfig `json:"scoped_keys,omitempty" yaml:"scoped_keys,omitempty"`

	// JWTSecret is the shared HS256 secret for JWT bearer tokens.
	JWTSecret string `json:"jwt_secret" yaml:"jwt_secret"`

	// JWTIssuer, when set, is required in the "iss" claim.
	JWTIssuer string `json:"jwt_issuer" yaml:"jwt_issuer"`

	// JWTAudience, when set, must appear in the "aud" claim.
	JWTAudience string `json:"jwt_audience" yaml:"jwt_audience"`
}

// ScopedKeyConfig pairs an API key with the namespaces it may access.
type ScopedKeyConfig struct {
	// Key is the API key value (required).
	Key string `json:"key" yaml:"key"`

	// Namespaces restricts the key to these namespaces. Empty means all.
	Namespaces []string `json:"namespaces,omitempty" yaml:"namespaces,omitempty"`
}

// Output format identifiers for CLIConfig.Output.
const (
	OutputTable = "table"
	OutputJSON  = "json"
	OutputYAML  = "yaml"
)

// CLIConfig configures the recall command-line client. Only the `recall`
// CLI consults this section; the API server ignores it.
type CLIConfig struct {
	// URL is the base URL of a running recall-server (e.g.
	// "http://localhost:8080"). When empty, the CLI operates in-process on
	// the local store defined in the Store section.
	URL string `json:"url" yaml:"url"`

	// APIKey authenticates requests to the server (sent as both Bearer and
	// X-API-Key). Prefer the RECALL__CLI__API_KEY environment variable over
	// storing keys in config files.
	APIKey string `json:"api_key" yaml:"api_key"`

	// Timeout bounds HTTP requests to the server. Defaults to 30s.
	Timeout Duration `json:"timeout" yaml:"timeout"`

	// Output is the default result format: "table", "json", or "yaml".
	// Defaults to "table".
	Output string `json:"output" yaml:"output"`

	// ClusterNodes are the base URLs probed by `recall cluster status`;
	// each should serve the distributed /healthz and /diagnostics
	// endpoints.
	ClusterNodes []string `json:"cluster_nodes" yaml:"cluster_nodes"`
}

// WithDefaults fills zero-valued fields with sensible defaults. It is
// applied automatically by Load and is safe to call on a partial config.
func (c *Config) WithDefaults() {
	s := &c.Server
	if s.Host == "" {
		s.Host = "127.0.0.1"
	}
	if s.Port == 0 {
		s.Port = 8080
	}
	if s.MaxUploadBytes == 0 {
		s.MaxUploadBytes = 10 << 20 // 10 MiB
	}
	if s.ReadTimeout == 0 {
		s.ReadTimeout = Duration(30 * time.Second)
	}
	if s.WriteTimeout == 0 {
		s.WriteTimeout = Duration(60 * time.Second)
	}
	if s.IdleTimeout == 0 {
		s.IdleTimeout = Duration(120 * time.Second)
	}

	st := &c.Store
	if st.Backend == "" {
		st.Backend = BackendMemory
	}
	if st.Namespace == "" {
		st.Namespace = "default"
	}
	if st.Embedder.Type == "" {
		st.Embedder.Type = EmbedderMock
	}
	if st.Embedder.Dimension == 0 {
		st.Embedder.Dimension = 384
	}
	if st.Chunking.Strategy == "" {
		st.Chunking.Strategy = ChunkingFixed
	}
	if st.Chunking.MaxTokens == 0 {
		st.Chunking.MaxTokens = 512
	}
	if st.Chunking.Overlap == 0 {
		st.Chunking.Overlap = 50
	}
	// MinChunkSize has no zero-value default: 0 is a meaningful setting (keep
	// every chunk). Only fill it in when the field was never mentioned, which
	// a negative sentinel cannot express -- so document_aware, whose sections
	// are the retrieval unit, keeps 0 and the size-based strategies get 50.
	if st.Chunking.MinChunkSize == 0 && st.Chunking.Strategy != ChunkingDocumentAware {
		st.Chunking.MinChunkSize = 50
	}

	cl := &c.CLI
	if cl.Timeout == 0 {
		cl.Timeout = Duration(30 * time.Second)
	}
	if cl.Output == "" {
		cl.Output = OutputTable
	}
}

// Load reads a configuration file (JSON or YAML, chosen by extension),
// applies defaults, and validates the result.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	cfg := &Config{}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parsing YAML config %s: %w", path, err)
		}
	case ".json", "":
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parsing JSON config %s: %w", path, err)
		}
	default:
		return nil, fmt.Errorf("unsupported config extension %q (want .json, .yaml, or .yml)", filepath.Ext(path))
	}
	cfg.normalizeSlices()
	cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validating config %s: %w", path, err)
	}
	return cfg, nil
}

// normalizeSlices canonicalizes empty slices to nil so that Save/Load
// round-trips are stable across the JSON and YAML encodings (YAML decodes
// "[]" and "null" into different slice states).
func (c *Config) normalizeSlices() {
	if len(c.Auth.APIKeys) == 0 {
		c.Auth.APIKeys = nil
	}
	if len(c.Auth.ScopedKeys) == 0 {
		c.Auth.ScopedKeys = nil
	}
	if len(c.CLI.ClusterNodes) == 0 {
		c.CLI.ClusterNodes = nil
	}
}

// Save writes the configuration to path as JSON or YAML based on the
// file extension.
func Save(path string, c *Config) error {
	var data []byte
	var err error
	switch strings.ToLower(filepath.Ext(path)) {
	case ".yaml", ".yml":
		data, err = yaml.Marshal(c)
	case ".json", "":
		data, err = json.MarshalIndent(c, "", "  ")
	default:
		return fmt.Errorf("unsupported config extension %q (want .json, .yaml, or .yml)", filepath.Ext(path))
	}
	if err != nil {
		return fmt.Errorf("encoding config: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("writing config %s: %w", path, err)
	}
	return nil
}

// Validate checks the configuration for correctness, returning a combined
// error describing every problem found.
func (c *Config) Validate() error {
	var problems []string

	s := c.Server
	if s.Host == "" {
		problems = append(problems, "server.host is required")
	}
	if s.Port < 0 || s.Port > 65535 {
		problems = append(problems, fmt.Sprintf("server.port %d out of range 1-65535", s.Port))
	}
	if s.MaxUploadBytes < 0 {
		problems = append(problems, "server.max_upload_bytes must be >= 0")
	}
	if s.ReadTimeout < 0 || s.WriteTimeout < 0 || s.IdleTimeout < 0 {
		problems = append(problems, "server timeouts must be >= 0")
	}

	st := c.Store
	switch st.Backend {
	case BackendMemory, BackendSQLite:
	case "":
		problems = append(problems, "store.backend is required")
	default:
		problems = append(problems, fmt.Sprintf("store.backend %q unknown (want %q or %q)", st.Backend, BackendMemory, BackendSQLite))
	}
	if st.Backend == BackendSQLite && st.Path == "" {
		problems = append(problems, "store.path is required for sqlite backend")
	}

	e := st.Embedder
	switch e.Type {
	case EmbedderMock, EmbedderOpenAI, EmbedderCohere, EmbedderOllama, EmbedderONNX:
	case "":
		problems = append(problems, "store.embedder.type is required")
	default:
		problems = append(problems, fmt.Sprintf("store.embedder.type %q unknown", e.Type))
	}
	if e.Type == EmbedderOpenAI || e.Type == EmbedderCohere {
		if e.APIKeyEnv == "" {
			problems = append(problems, fmt.Sprintf("store.embedder.api_key_env is required for %s embedder", e.Type))
		}
	}
	if e.Type == EmbedderOllama && e.Model == "" {
		problems = append(problems, "store.embedder.model is required for ollama embedder")
	}
	if e.Type == EmbedderONNX && e.Path == "" {
		problems = append(problems, "store.embedder.path is required for onnx embedder")
	}
	if e.Dimension < 0 {
		problems = append(problems, "store.embedder.dimension must be >= 0")
	}

	k := st.Chunking
	switch k.Strategy {
	case ChunkingFixed, ChunkingRecursive, ChunkingDocumentAware:
	case "":
		problems = append(problems, "store.chunking.strategy is required")
	default:
		problems = append(problems, fmt.Sprintf("store.chunking.strategy %q unknown (want %q, %q or %q)", k.Strategy, ChunkingFixed, ChunkingRecursive, ChunkingDocumentAware))
	}
	if k.MinChunkSize < 0 {
		problems = append(problems, "store.chunking.min_chunk_size must be >= 0")
	}
	if k.MaxTokens <= 0 {
		problems = append(problems, "store.chunking.max_tokens must be > 0")
	}
	if k.Overlap < 0 || k.Overlap >= k.MaxTokens {
		problems = append(problems, "store.chunking.overlap must be in [0, max_tokens)")
	}

	a := c.Auth
	if a.Enabled {
		if len(a.APIKeys) == 0 && len(a.ScopedKeys) == 0 && a.JWTSecret == "" {
			problems = append(problems, "auth: enable requires at least one api_keys or scoped_keys entry, or jwt_secret")
		}
	}
	plain := make(map[string]struct{}, len(a.APIKeys))
	for _, k := range a.APIKeys {
		plain[k] = struct{}{}
	}
	for i, sk := range a.ScopedKeys {
		if strings.TrimSpace(sk.Key) == "" {
			problems = append(problems, fmt.Sprintf("auth.scoped_keys[%d].key is required", i))
			continue
		}
		if _, dup := plain[sk.Key]; dup {
			problems = append(problems, fmt.Sprintf("auth: key is defined in both api_keys and scoped_keys: %q", sk.Key))
		}
	}

	cl := c.CLI
	if cl.URL != "" {
		if _, err := url.ParseRequestURI(cl.URL); err != nil {
			problems = append(problems, fmt.Sprintf("cli.url %q is not a valid URL: %v", cl.URL, err))
		}
	}
	switch cl.Output {
	case "", OutputTable, OutputJSON, OutputYAML:
	default:
		problems = append(problems, fmt.Sprintf("cli.output %q unknown (want %q, %q, or %q)", cl.Output, OutputTable, OutputJSON, OutputYAML))
	}
	if cl.Timeout < 0 {
		problems = append(problems, "cli.timeout must be >= 0")
	}

	if len(problems) > 0 {
		return errors.New("invalid config: " + strings.Join(problems, "; "))
	}
	return nil
}

// EnvPrefix is the default environment variable prefix. Setters look for
// variables named <prefix>__<SECTION>__<KEY> (double underscore separates
// nesting levels), e.g. RECALL__SERVER__PORT or RECALL__AUTH__ENABLED.
const EnvPrefix = "RECALL"

// ApplyEnv overrides configuration fields from environment variables using
// the prefix (default "RECALL"). Nested fields use double underscores,
// e.g. RECALL__SERVER__PORT=9090. List fields (auth.api_keys) are
// comma-separated. Unknown or malformed values are ignored so that a
// mis-typed variable never blocks startup; Validate catches the result.
func (c *Config) ApplyEnv(prefix string) {
	if prefix == "" {
		prefix = EnvPrefix
	}
	applyString(prefix, &c.Server.Host, "SERVER__HOST")
	applyInt(prefix, &c.Server.Port, "SERVER__PORT")
	applyInt64(prefix, &c.Server.MaxUploadBytes, "SERVER__MAX_UPLOAD_BYTES")
	applyDuration(prefix, &c.Server.ReadTimeout, "SERVER__READ_TIMEOUT")
	applyDuration(prefix, &c.Server.WriteTimeout, "SERVER__WRITE_TIMEOUT")
	applyDuration(prefix, &c.Server.IdleTimeout, "SERVER__IDLE_TIMEOUT")
	applyBool(prefix, &c.Server.AllowCORS, "SERVER__ALLOW_CORS")

	applyString(prefix, &c.Store.Backend, "STORE__BACKEND")
	applyString(prefix, &c.Store.Path, "STORE__PATH")
	applyString(prefix, &c.Store.Namespace, "STORE__NAMESPACE")
	applyString(prefix, &c.Store.Embedder.Type, "STORE__EMBEDDER__TYPE")
	applyString(prefix, &c.Store.Embedder.Model, "STORE__EMBEDDER__MODEL")
	applyInt(prefix, &c.Store.Embedder.Dimension, "STORE__EMBEDDER__DIMENSION")
	applyString(prefix, &c.Store.Embedder.APIKeyEnv, "STORE__EMBEDDER__API_KEY_ENV")
	applyString(prefix, &c.Store.Embedder.BaseURL, "STORE__EMBEDDER__BASE_URL")
	applyString(prefix, &c.Store.Embedder.Path, "STORE__EMBEDDER__PATH")
	applyString(prefix, &c.Store.Chunking.Strategy, "STORE__CHUNKING__STRATEGY")
	applyInt(prefix, &c.Store.Chunking.MaxTokens, "STORE__CHUNKING__MAX_TOKENS")
	applyInt(prefix, &c.Store.Chunking.Overlap, "STORE__CHUNKING__OVERLAP")

	applyBool(prefix, &c.Auth.Enabled, "AUTH__ENABLED")
	applyStringSlice(prefix, &c.Auth.APIKeys, "AUTH__API_KEYS")
	applyString(prefix, &c.Auth.JWTSecret, "AUTH__JWT_SECRET")
	applyString(prefix, &c.Auth.JWTIssuer, "AUTH__JWT_ISSUER")
	applyString(prefix, &c.Auth.JWTAudience, "AUTH__JWT_AUDIENCE")

	applyString(prefix, &c.CLI.URL, "CLI__URL")
	applyString(prefix, &c.CLI.APIKey, "CLI__API_KEY")
	applyDuration(prefix, &c.CLI.Timeout, "CLI__TIMEOUT")
	applyString(prefix, &c.CLI.Output, "CLI__OUTPUT")
	applyStringSlice(prefix, &c.CLI.ClusterNodes, "CLI__CLUSTER_NODES")
}

func parseInt(v string) (int, error)     { return strconv.Atoi(v) }
func parseInt64(v string) (int64, error) { return strconv.ParseInt(v, 10, 64) }
func parseBool(v string) (bool, error)   { return strconv.ParseBool(v) }

func envKey(prefix, suffix string) string { return prefix + "__" + suffix }

func applyString(prefix string, dst *string, suffix string) {
	if v, ok := os.LookupEnv(envKey(prefix, suffix)); ok && v != "" {
		*dst = v
	}
}

func applyInt(prefix string, dst *int, suffix string) {
	if v, ok := os.LookupEnv(envKey(prefix, suffix)); ok && v != "" {
		if n, err := parseInt(v); err == nil {
			*dst = n
		}
	}
}

func applyInt64(prefix string, dst *int64, suffix string) {
	if v, ok := os.LookupEnv(envKey(prefix, suffix)); ok && v != "" {
		if n, err := parseInt64(v); err == nil {
			*dst = n
		}
	}
}

func applyBool(prefix string, dst *bool, suffix string) {
	if v, ok := os.LookupEnv(envKey(prefix, suffix)); ok && v != "" {
		if b, err := parseBool(v); err == nil {
			*dst = b
		}
	}
}

func applyDuration(prefix string, dst *Duration, suffix string) {
	if v, ok := os.LookupEnv(envKey(prefix, suffix)); ok && v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			*dst = Duration(d)
		}
	}
}

func applyStringSlice(prefix string, dst *[]string, suffix string) {
	if v, ok := os.LookupEnv(envKey(prefix, suffix)); ok && v != "" {
		parts := strings.Split(v, ",")
		out := make([]string, 0, len(parts))
		for _, p := range parts {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
		if len(out) > 0 {
			*dst = out
		}
	}
}
