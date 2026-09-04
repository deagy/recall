package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/deagy/recall/config"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

func TestLoaderForPath(t *testing.T) {
	dir := t.TempDir()
	if _, err := loaderForPath(dir, true, config.StoreConfig{}); err != nil {
		t.Errorf("directory loader: %v", err)
	}

	txt := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(txt, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loaderForPath(txt, true, config.StoreConfig{}); err != nil {
		t.Errorf("file loader: %v", err)
	}

	if _, err := loaderForPath(filepath.Join(dir, "missing.txt"), true, config.StoreConfig{}); err == nil {
		t.Error("missing path: expected error")
	}

	xyz := filepath.Join(dir, "a.xyz")
	if err := os.WriteFile(xyz, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loaderForPath(xyz, true, config.StoreConfig{}); err == nil {
		t.Error("unsupported extension: expected error")
	}
}

func TestToCoreDocument(t *testing.T) {
	d := &loader.Document{
		ID:       "doc-1",
		Title:    "Notes",
		Source:   "/tmp/notes.txt",
		Metadata: map[string]core.Value{"k": core.String{Value: "v"}},
	}
	cd := toCoreDocument(d, "ns-a")
	if cd.ID != "doc-1" || cd.Title != "Notes" || cd.Source != "/tmp/notes.txt" {
		t.Errorf("document fields lost: %+v", cd)
	}
	if cd.Namespace != "ns-a" {
		t.Errorf("namespace = %q, want ns-a", cd.Namespace)
	}
	if got, ok := cd.Metadata["k"].(core.String); !ok || got.Value != "v" {
		t.Errorf("metadata not copied: %v", cd.Metadata)
	}

	// No metadata stays empty (core.NewDocument initializes the map).
	if cd := toCoreDocument(&loader.Document{ID: "d"}, "ns"); len(cd.Metadata) != 0 {
		t.Errorf("expected empty metadata, got %v", cd.Metadata)
	}
}

// customValue exercises the default branch of metadataToAny.
type customValue struct{}

func (customValue) Kind() core.ValueKind { return core.ValueKindLiteral }
func (customValue) String() string       { return "custom" }

func TestMetadataToAny(t *testing.T) {
	if got := metadataToAny(nil); got != nil {
		t.Errorf("nil metadata: %v", got)
	}
	meta := map[string]core.Value{
		"s":   core.String{Value: "text"},
		"n":   core.Number{Value: 1.5},
		"b":   core.Boolean{Value: true},
		"u":   core.URI{Value: "https://example.com"},
		"l":   core.Literal{Value: "lit"},
		"nil": nil,
		"c":   customValue{},
	}
	got := metadataToAny(meta)
	if got["s"] != "text" || got["n"] != 1.5 || got["b"] != true ||
		got["u"] != "https://example.com" || got["l"] != "lit" {
		t.Errorf("typed values not unwrapped: %v", got)
	}
	if _, exists := got["nil"]; exists {
		t.Error("nil values should be dropped")
	}
	if got["c"] != "custom" {
		t.Errorf("unknown value kind should use String(), got %v", got["c"])
	}
}

func TestNamespaceFilter(t *testing.T) {
	if fs := namespaceFilter(""); fs != nil {
		t.Errorf("empty namespace should have no filter, got %v", fs)
	}
	if fs := namespaceFilter("ns"); len(fs) != 1 {
		t.Fatalf("expected 1 filter, got %d", len(fs))
	}
}

func TestQuoteLiteral(t *testing.T) {
	if got := quoteLiteral("O'Brien"); got != "'O''Brien'" {
		t.Errorf("quoteLiteral = %s", got)
	}
}

func TestBaseName(t *testing.T) {
	cases := map[string]string{
		"/a/b/c.db": "c.db",
		"c.db":      "c.db",
		`a\b.db`:    "b.db",
	}
	for in, want := range cases {
		if got := baseName(in); got != want {
			t.Errorf("baseName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestJoinStrings(t *testing.T) {
	if got := joinStrings(nil); got != "" {
		t.Errorf("empty = %q", got)
	}
	if got := joinStrings([]string{"a", "b", "c"}); got != "a, b, c" {
		t.Errorf("join = %q", got)
	}
}

func TestJoinNonEmpty(t *testing.T) {
	if got := joinNonEmpty([]string{" a ", "", "b", "  "}); got != "a\nb" {
		t.Errorf("joinNonEmpty = %q", got)
	}
}

func TestExitError(t *testing.T) {
	e := &exitError{Code: 2, Message: "boom"}
	if e.Error() != "boom" {
		t.Errorf("Error() = %q", e.Error())
	}
}

func TestLoadConfig_ExplicitPath(t *testing.T) {
	cfgPath, dbPath := writeSQLiteConfig(t)
	o := &globalOptions{configPath: cfgPath}
	cfg, err := o.loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Store.Backend != config.BackendSQLite || cfg.Store.Path != dbPath {
		t.Errorf("config = %+v", cfg.Store)
	}

	o = &globalOptions{configPath: filepath.Join(t.TempDir(), "missing.json")}
	if _, err := o.loadConfig(); err == nil {
		t.Error("missing explicit config: expected error")
	}
}

func TestLoadConfig_HomeDir(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	yaml := "store:\n  backend: sqlite\n  path: /tmp/home.db\n"
	if err := os.WriteFile(filepath.Join(home, ".recall.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := (&globalOptions{}).loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Store.Backend != config.BackendSQLite || cfg.Store.Path != "/tmp/home.db" {
		t.Errorf("home config not loaded: %+v", cfg.Store)
	}
}

func TestLoadConfig_Defaults(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cfg, err := (&globalOptions{}).loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Store.Backend != config.BackendMemory {
		t.Errorf("default backend = %q, want memory", cfg.Store.Backend)
	}
}

func TestEffectiveNamespace(t *testing.T) {
	cfg := &config.Config{}
	cfg.Store.Namespace = "from-config"

	o := &globalOptions{namespace: "from-flag", cfg: cfg}
	if got := o.effectiveNamespace(); got != "from-flag" {
		t.Errorf("flag should win, got %q", got)
	}
	o = &globalOptions{cfg: cfg}
	if got := o.effectiveNamespace(); got != "from-config" {
		t.Errorf("config should apply, got %q", got)
	}
	o = &globalOptions{cfg: &config.Config{}}
	if got := o.effectiveNamespace(); got != "default" {
		t.Errorf("fallback = %q, want default", got)
	}
}

func TestParseMigrationHeader(t *testing.T) {
	mig, err := parseMigrationHeader("version=2 name=add_col")
	if err != nil {
		t.Fatalf("valid header: %v", err)
	}
	if mig.Version != 2 || mig.Name != "add_col" {
		t.Errorf("migration = %+v", mig)
	}

	// Missing name gets a default.
	mig, err = parseMigrationHeader("version=3")
	if err != nil {
		t.Fatalf("version only: %v", err)
	}
	if mig.Name != "migration-3" {
		t.Errorf("default name = %q", mig.Name)
	}

	for _, bad := range []string{"name=only", "version=0", "version=x", "noversion"} {
		if _, err := parseMigrationHeader(bad); err == nil {
			t.Errorf("header %q: expected error", bad)
		}
	}
}

func TestParseMigrationFile(t *testing.T) {
	content := `-- file-level comment, ignored
-- recall-migration: version=1 name=first
CREATE TABLE a (id INTEGER);

-- recall-migration: version=2 name=second
ALTER TABLE a ADD COLUMN name TEXT;
`
	path := filepath.Join(t.TempDir(), "m.sql")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	migs, err := parseMigrationFile(path)
	if err != nil {
		t.Fatalf("parseMigrationFile: %v", err)
	}
	if len(migs) != 2 {
		t.Fatalf("expected 2 migrations, got %d", len(migs))
	}
	if migs[0].Version != 1 || migs[0].Name != "first" || migs[0].SQL != "CREATE TABLE a (id INTEGER);" {
		t.Errorf("migration 1 = %+v", migs[0])
	}
	if migs[1].Version != 2 || migs[1].SQL != "ALTER TABLE a ADD COLUMN name TEXT;" {
		t.Errorf("migration 2 = %+v", migs[1])
	}

	// No headers is an error.
	noHeader := filepath.Join(t.TempDir(), "empty.sql")
	if err := os.WriteFile(noHeader, []byte("-- just a comment\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := parseMigrationFile(noHeader); err == nil {
		t.Error("file without headers: expected error")
	}

	// Malformed header is an error.
	badHeader := filepath.Join(t.TempDir(), "bad.sql")
	if err := os.WriteFile(badHeader, []byte("-- recall-migration: name=x\nSELECT 1;\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := parseMigrationFile(badHeader); err == nil {
		t.Error("malformed header: expected error")
	}

	if _, err := parseMigrationFile(filepath.Join(t.TempDir(), "missing.sql")); err == nil {
		t.Error("missing file: expected error")
	}
}
