package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func writeTemp(t *testing.T, name string, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("writing temp config: %v", err)
	}
	return path
}

const jsonConfig = `{
  "server": {"host": "0.0.0.0", "port": 9090, "read_timeout": "15s", "allow_cors": true},
  "store": {
    "backend": "sqlite",
    "path": "/tmp/recall.db",
    "namespace": "kb",
    "embedder": {"type": "openai", "model": "text-embedding-3-small", "api_key_env": "OPENAI_API_KEY", "dimension": 1536},
    "chunking": {"strategy": "recursive", "max_tokens": 256, "overlap": 32}
  },
  "auth": {"enabled": true, "api_keys": ["k1", "k2"], "jwt_secret": "s3c", "jwt_issuer": "recall"}
}`

func TestLoad_JSON(t *testing.T) {
	path := writeTemp(t, "recall.json", jsonConfig)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Server.Host != "0.0.0.0" || cfg.Server.Port != 9090 {
		t.Errorf("server = %+v", cfg.Server)
	}
	if cfg.Server.ReadTimeout != Duration(15*time.Second) {
		t.Errorf("read timeout = %v, want 15s", cfg.Server.ReadTimeout)
	}
	if cfg.Server.WriteTimeout != Duration(60*time.Second) {
		t.Errorf("default write timeout = %v, want 60s", cfg.Server.WriteTimeout)
	}
	if !cfg.Server.AllowCORS {
		t.Error("allow_cors not applied")
	}
	if cfg.Store.Backend != BackendSQLite || cfg.Store.Path != "/tmp/recall.db" {
		t.Errorf("store = %+v", cfg.Store)
	}
	if cfg.Store.Embedder.Type != EmbedderOpenAI || cfg.Store.Embedder.APIKeyEnv != "OPENAI_API_KEY" {
		t.Errorf("embedder = %+v", cfg.Store.Embedder)
	}
	if cfg.Store.Chunking.Strategy != ChunkingRecursive || cfg.Store.Chunking.MaxTokens != 256 {
		t.Errorf("chunking = %+v", cfg.Store.Chunking)
	}
	if !cfg.Auth.Enabled || len(cfg.Auth.APIKeys) != 2 || cfg.Auth.APIKeys[0] != "k1" {
		t.Errorf("auth = %+v", cfg.Auth)
	}
	if cfg.Server.MaxUploadBytes != int64(10<<20) {
		t.Errorf("default max upload = %d, want 10MiB", cfg.Server.MaxUploadBytes)
	}
}

func TestLoad_YAML(t *testing.T) {
	yaml := `
server:
  host: 127.0.0.1
  port: 7070
  write_timeout: 45s
store:
  backend: memory
  embedder:
    type: mock
    dimension: 64
auth:
  enabled: false
`
	path := writeTemp(t, "recall.yaml", yaml)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load YAML: %v", err)
	}
	if cfg.Server.Port != 7070 {
		t.Errorf("port = %d, want 7070", cfg.Server.Port)
	}
	if cfg.Server.WriteTimeout != Duration(45*time.Second) {
		t.Errorf("write timeout = %v, want 45s", cfg.Server.WriteTimeout)
	}
	if cfg.Store.Backend != BackendMemory || cfg.Store.Embedder.Dimension != 64 {
		t.Errorf("store = %+v", cfg.Store)
	}
	if cfg.Auth.Enabled {
		t.Error("auth should be disabled")
	}
}

func TestLoad_Errors(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("expected error for missing file")
	}
	if _, err := Load(writeTemp(t, "bad.json", "{oops")); err == nil {
		t.Error("expected error for invalid JSON")
	}
	if _, err := Load(writeTemp(t, "bad.toml", "a = 1")); err == nil {
		t.Error("expected error for unsupported extension")
	}
}

func TestLoad_InvalidValues(t *testing.T) {
	cases := map[string]string{
		"port":        `{"server": {"port": 70000}}`,
		"backend":     `{"store": {"backend": "cassandra"}}`,
		"sqlite path": `{"store": {"backend": "sqlite"}}`,
		"embedder":    `{"store": {"embedder": {"type": "openai"}}}`,
		"onnx path":   `{"store": {"embedder": {"type": "onnx"}}}`,
		"auth creds":  `{"auth": {"enabled": true}}`,
		"overlap":     `{"store": {"chunking": {"max_tokens": 100, "overlap": 100}}}`,
		"bad timeout": `{"server": {"read_timeout": "nope"}}`,
	}
	for name, content := range cases {
		path := writeTemp(t, "cfg.json", content)
		if _, err := Load(path); err == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
}
func TestValidate_ValidConfig(t *testing.T) {
	c := &Config{}
	c.WithDefaults()
	if err := c.Validate(); err != nil {
		t.Fatalf("default config should validate, got: %v", err)
	}
}

func TestValidate_MultipleProblems(t *testing.T) {
	c := &Config{}
	c.Server.Port = -1
	c.Store.Backend = "bogus"
	c.Auth.Enabled = true
	err := c.Validate()
	if err == nil {
		t.Fatal("expected validation error")
	}
	msg := err.Error()
	for _, want := range []string{"port", "backend", "auth"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error %q should mention %q", msg, want)
		}
	}
}

func TestValidate_ScopedKeys(t *testing.T) {
	// Scoped keys alone satisfy the auth-enabled requirement.
	c := &Config{}
	c.WithDefaults()
	c.Auth.Enabled = true
	c.Auth.ScopedKeys = []ScopedKeyConfig{{Key: "team-a", Namespaces: []string{"ns-a"}}}
	if err := c.Validate(); err != nil {
		t.Fatalf("scoped-key-only auth should validate, got: %v", err)
	}

	// A blank scoped key is rejected.
	bad := &Config{}
	bad.WithDefaults()
	bad.Auth.ScopedKeys = []ScopedKeyConfig{{Key: "  "}}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "scoped_keys[0].key") {
		t.Errorf("expected scoped_keys[0].key error, got: %v", err)
	}

	// A key in both api_keys and scoped_keys is ambiguous and rejected.
	dup := &Config{}
	dup.WithDefaults()
	dup.Auth.Enabled = true
	dup.Auth.APIKeys = []string{"shared"}
	dup.Auth.ScopedKeys = []ScopedKeyConfig{{Key: "shared"}}
	if err := dup.Validate(); err == nil || !strings.Contains(err.Error(), "both api_keys and scoped_keys") {
		t.Errorf("expected duplicate-key error, got: %v", err)
	}
}

func TestLoad_ScopedKeys(t *testing.T) {
	content := `{"auth": {"enabled": true, "api_keys": ["admin"], "scoped_keys": [{"key": "team-a", "namespaces": ["ns-a", "ns-b"]}]}}`
	path := writeTemp(t, "cfg.json", content)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Auth.ScopedKeys) != 1 || cfg.Auth.ScopedKeys[0].Key != "team-a" ||
		len(cfg.Auth.ScopedKeys[0].Namespaces) != 2 || cfg.Auth.ScopedKeys[0].Namespaces[0] != "ns-a" {
		t.Errorf("scoped keys not loaded: %+v", cfg.Auth.ScopedKeys)
	}
}

func TestApplyEnv(t *testing.T) {
	c := &Config{}
	c.WithDefaults()

	t.Setenv("RECALL__SERVER__PORT", "1234")
	t.Setenv("RECALL__SERVER__HOST", "10.0.0.5")
	t.Setenv("RECALL__SERVER__MAX_UPLOAD_BYTES", "2048")
	t.Setenv("RECALL__SERVER__READ_TIMEOUT", "90s")
	t.Setenv("RECALL__SERVER__ALLOW_CORS", "true")
	t.Setenv("RECALL__STORE__BACKEND", "sqlite")
	t.Setenv("RECALL__STORE__PATH", "/var/lib/recall.db")
	t.Setenv("RECALL__STORE__EMBEDDER__TYPE", "ollama")
	t.Setenv("RECALL__STORE__EMBEDDER__MODEL", "nomic-embed-text")
	t.Setenv("RECALL__STORE__CHUNKING__MAX_TOKENS", "128")
	t.Setenv("RECALL__AUTH__ENABLED", "true")
	t.Setenv("RECALL__AUTH__API_KEYS", "a, b ,c")
	t.Setenv("RECALL__AUTH__JWT_SECRET", "env-secret")
	// Malformed values must be ignored without panicking.
	t.Setenv("RECALL__STORE__CHUNKING__OVERLAP", "not-a-number")

	c.ApplyEnv("")

	if c.Server.Port != 1234 || c.Server.Host != "10.0.0.5" {
		t.Errorf("server env overrides not applied: %+v", c.Server)
	}
	if c.Server.MaxUploadBytes != 2048 {
		t.Errorf("max upload = %d, want 2048", c.Server.MaxUploadBytes)
	}
	if c.Server.ReadTimeout != Duration(90*time.Second) {
		t.Errorf("read timeout = %v, want 90s", c.Server.ReadTimeout)
	}
	if !c.Server.AllowCORS {
		t.Error("allow_cors env override not applied")
	}
	if c.Store.Backend != BackendSQLite || c.Store.Path != "/var/lib/recall.db" {
		t.Errorf("store env overrides not applied: %+v", c.Store)
	}
	if c.Store.Embedder.Type != EmbedderOllama || c.Store.Embedder.Model != "nomic-embed-text" {
		t.Errorf("embedder env overrides not applied: %+v", c.Store.Embedder)
	}
	if c.Store.Chunking.MaxTokens != 128 {
		t.Errorf("max tokens = %d, want 128", c.Store.Chunking.MaxTokens)
	}
	if !c.Auth.Enabled || len(c.Auth.APIKeys) != 3 || c.Auth.APIKeys[1] != "b" {
		t.Errorf("auth env overrides not applied: %+v", c.Auth)
	}
	if c.Auth.JWTSecret != "env-secret" {
		t.Errorf("jwt secret = %q, want env-secret", c.Auth.JWTSecret)
	}
	// Default overlap (50) should survive the malformed value.
	if c.Store.Chunking.Overlap != 50 {
		t.Errorf("overlap = %d, want 50 (malformed env ignored)", c.Store.Chunking.Overlap)
	}
}

func TestApplyEnv_CustomPrefix(t *testing.T) {
	c := &Config{}
	c.WithDefaults()
	t.Setenv("MYAPP__SERVER__PORT", "5555")
	t.Setenv("RECALL__SERVER__PORT", "1111")
	c.ApplyEnv("MYAPP")
	if c.Server.Port != 5555 {
		t.Errorf("port = %d, want 5555 (custom prefix)", c.Server.Port)
	}
}

func TestCLI_Defaults(t *testing.T) {
	c := &Config{}
	c.WithDefaults()
	if c.CLI.Timeout != Duration(30*time.Second) {
		t.Errorf("cli timeout default = %v, want 30s", c.CLI.Timeout)
	}
	if c.CLI.Output != OutputTable {
		t.Errorf("cli output default = %q, want %q", c.CLI.Output, OutputTable)
	}
}

func TestLoad_CLISection(t *testing.T) {
	const yaml = `
server:
  host: 127.0.0.1
store:
  backend: memory
cli:
  url: http://recall.example:8080
  api_key: secret
  timeout: 5s
  output: json
  cluster_nodes: ["http://n1:9000", "http://n2:9000"]
`
	path := writeTemp(t, "recall.yaml", yaml)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.CLI.URL != "http://recall.example:8080" {
		t.Errorf("cli.url = %q", cfg.CLI.URL)
	}
	if cfg.CLI.APIKey != "secret" {
		t.Errorf("cli.api_key = %q", cfg.CLI.APIKey)
	}
	if cfg.CLI.Timeout != Duration(5*time.Second) {
		t.Errorf("cli.timeout = %v, want 5s", cfg.CLI.Timeout)
	}
	if cfg.CLI.Output != OutputJSON {
		t.Errorf("cli.output = %q, want json", cfg.CLI.Output)
	}
	if len(cfg.CLI.ClusterNodes) != 2 || cfg.CLI.ClusterNodes[0] != "http://n1:9000" {
		t.Errorf("cli.cluster_nodes = %v", cfg.CLI.ClusterNodes)
	}
}

func TestApplyEnv_CLI(t *testing.T) {
	c := &Config{}
	c.WithDefaults()
	t.Setenv("RECALL__CLI__URL", "http://env:9090")
	t.Setenv("RECALL__CLI__API_KEY", "envkey")
	t.Setenv("RECALL__CLI__TIMEOUT", "7s")
	t.Setenv("RECALL__CLI__OUTPUT", "yaml")
	t.Setenv("RECALL__CLI__CLUSTER_NODES", "http://a:1, http://b:2")
	c.ApplyEnv("")
	if c.CLI.URL != "http://env:9090" {
		t.Errorf("cli.url env override not applied: %q", c.CLI.URL)
	}
	if c.CLI.APIKey != "envkey" {
		t.Errorf("cli.api_key env override not applied: %q", c.CLI.APIKey)
	}
	if c.CLI.Timeout != Duration(7*time.Second) {
		t.Errorf("cli.timeout env override not applied: %v", c.CLI.Timeout)
	}
	if c.CLI.Output != OutputYAML {
		t.Errorf("cli.output env override not applied: %q", c.CLI.Output)
	}
	if len(c.CLI.ClusterNodes) != 2 || c.CLI.ClusterNodes[1] != "http://b:2" {
		t.Errorf("cli.cluster_nodes env override not applied: %v", c.CLI.ClusterNodes)
	}
}

func TestValidate_CLIProblems(t *testing.T) {
	c := &Config{}
	c.WithDefaults()
	c.CLI.URL = "://not-a-url"
	c.CLI.Output = "xml"
	c.CLI.Timeout = Duration(-time.Second)
	err := c.Validate()
	if err == nil {
		t.Fatal("expected validation errors for cli section")
	}
	for _, want := range []string{"cli.url", "cli.output", "cli.timeout"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("validation error missing %q: %v", want, err)
		}
	}
}

func TestSave_Roundtrip(t *testing.T) {
	orig, err := Load(writeTemp(t, "in.json", jsonConfig))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	for _, ext := range []string{".json", ".yaml"} {
		out := filepath.Join(t.TempDir(), "out"+ext)
		if err := Save(out, orig); err != nil {
			t.Fatalf("Save %s: %v", ext, err)
		}
		got, err := Load(out)
		if err != nil {
			t.Fatalf("Load %s: %v", ext, err)
		}
		a, _ := json.Marshal(orig)
		b, _ := json.Marshal(got)
		if string(a) != string(b) {
			t.Errorf("%s roundtrip mismatch:\n%s\nvs\n%s", ext, a, b)
		}
	}

	if err := Save(filepath.Join(t.TempDir(), "out.toml"), orig); err == nil {
		t.Error("expected error for unsupported Save extension")
	}
}

func TestDuration_MarshalJSON(t *testing.T) {
	d := Duration(90 * time.Second)
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	if string(b) != `"1m30s"` {
		t.Errorf("marshaled = %s, want \"1m30s\"", b)
	}
	var back Duration
	if err := json.Unmarshal([]byte(`"2m"`), &back); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if back != Duration(2*time.Minute) {
		t.Errorf("unmarshaled = %v, want 2m", back)
	}
	if err := json.Unmarshal([]byte(`"bogus"`), &back); err == nil {
		t.Error("expected error for bad duration")
	}
}

func TestWatcher_ValidationError(t *testing.T) {
	c := &Config{}
	c.Store.Backend = "bogus"
	if err := c.Validate(); err == nil {
		t.Fatal("expected error")
	}
}

func TestWatcher_DetectsChange(t *testing.T) {
	path := writeTemp(t, "watch.json", `{"server": {"port": 1111}}`)

	type result struct {
		cfg *Config
		err error
	}
	results := make(chan result, 4)
	w, err := NewWatcher(path, 100*time.Millisecond, func(cfg *Config) error {
		results <- result{cfg: cfg}
		return nil
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}
	if err := w.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer w.Stop()

	// Double start must fail.
	if err := w.Start(); err == nil {
		t.Error("expected error starting an already-running watcher")
	}

	// Rewrite with a different port and a different size.
	time.Sleep(5 * time.Millisecond)
	if err := os.WriteFile(path, []byte(`{"server": {"port": 2222, "host": "x.y.z"}}`), 0o600); err != nil {
		t.Fatalf("rewriting: %v", err)
	}

	select {
	case res := <-results:
		if res.err != nil {
			t.Fatalf("callback error: %v", res.err)
		}
		if res.cfg.Server.Port != 2222 {
			t.Errorf("reloaded port = %d, want 2222", res.cfg.Server.Port)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("watcher did not fire within 3s")
	}

	if err := w.LastError(); err != nil {
		t.Errorf("LastError after success = %v, want nil", err)
	}
}

func TestWatcher_SkipsInvalidFile(t *testing.T) {
	path := writeTemp(t, "watch.json", `{"server": {"port": 1111}}`)

	var calls atomic.Int32
	w, err := NewWatcher(path, 100*time.Millisecond, func(*Config) error {
		calls.Add(1)
		return nil
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}
	if err := w.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer w.Stop()

	// Write an invalid file; the watcher must not invoke the callback and
	// must record the error.
	time.Sleep(5 * time.Millisecond)
	if err := os.WriteFile(path, []byte(`{invalid json`), 0o600); err != nil {
		t.Fatalf("rewriting: %v", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for w.LastError() == nil && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if w.LastError() == nil {
		t.Fatal("expected LastError after invalid rewrite")
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("callback invoked %d times for invalid file, want 0", n)
	}
}

func TestWatcher_RejectsOnChangeError(t *testing.T) {
	path := writeTemp(t, "watch.json", `{"server": {"port": 1111}}`)

	w, err := NewWatcher(path, 100*time.Millisecond, func(*Config) error {
		return fmt.Errorf("simulated rejection")
	})
	if err != nil {
		t.Fatalf("NewWatcher: %v", err)
	}
	if err := w.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer w.Stop()

	time.Sleep(5 * time.Millisecond)
	if err := os.WriteFile(path, []byte(`{"server": {"port": 3333}}`), 0o600); err != nil {
		t.Fatalf("rewriting: %v", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for w.LastError() == nil && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if w.LastError() == nil || !strings.Contains(w.LastError().Error(), "simulated rejection") {
		t.Errorf("LastError = %v, want simulated rejection", w.LastError())
	}
}

func TestExampleDeployConfigs(t *testing.T) {
	// The bundled deploy examples must always be loadable.
	for _, path := range []string{
		filepath.Join("..", "deploy", "config", "recall.example.json"),
		filepath.Join("..", "deploy", "config", "recall.example.yaml"),
	} {
		cfg, err := Load(path)
		if err != nil {
			t.Errorf("Load(%s): %v", path, err)
			continue
		}
		if cfg.Server.Port != 8080 || cfg.Store.Backend != BackendSQLite {
			t.Errorf("%s: unexpected values port=%d backend=%s", path, cfg.Server.Port, cfg.Store.Backend)
		}
	}
}

func TestWatcher_RequiresArgs(t *testing.T) {
	if _, err := NewWatcher("", time.Second, func(*Config) error { return nil }); err == nil {
		t.Error("expected error for empty path")
	}
	if _, err := NewWatcher("/tmp/x.json", time.Second, nil); err == nil {
		t.Error("expected error for nil callback")
	}
	if _, err := NewWatcher(filepath.Join(t.TempDir(), "nope.json"), time.Second, func(*Config) error { return nil }); err == nil {
		t.Error("expected error for missing file")
	}
}
