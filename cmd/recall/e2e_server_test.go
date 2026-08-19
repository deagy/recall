package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/deagy/recall/config"
)

func TestServerUploadSearchHybridRAGInfo(t *testing.T) {
	ts := newTestAPIServer(t, struct{ auth, graph bool }{})
	cfgPath, _ := writeSQLiteConfig(t) // config required for init; store section unused in server mode
	notes := writeNotesFile(t)
	base := []string{"--config", cfgPath, "--server", ts.URL}

	out := mustRunCLI(t, append(base, "upload", notes)...)
	if !strings.Contains(out, "uploaded 1 document(s), 0 failed") || !strings.Contains(out, "mode: server") {
		t.Fatalf("server upload output: %s", out)
	}

	out = mustRunCLI(t, append(base, "search", "garbage collection", "-o", "json")...)
	var so searchOutput
	decodeJSON(t, out, &so)
	if so.Mode != "server" || so.Count == 0 {
		t.Errorf("server search = %+v", so)
	}

	out = mustRunCLI(t, append(base, "hybrid-search", "garbage collection", "-o", "json")...)
	decodeJSON(t, out, &so)
	if so.Mode != "server" || !so.Hybrid || so.Count == 0 {
		t.Errorf("server hybrid search = %+v", so)
	}

	out = mustRunCLI(t, append(base, "rag", "What is Go?", "-o", "json")...)
	var ro ragOutput
	decodeJSON(t, out, &ro)
	if ro.Mode != "server" || ro.Answer == "" || len(ro.Sources) == 0 {
		t.Errorf("server rag = %+v", ro)
	}

	out = mustRunCLI(t, append(base, "store", "info", "-o", "json")...)
	var info storeInfoOutput
	decodeJSON(t, out, &info)
	if info.Mode != "server" || info.Backend != "memory" || !info.OK {
		t.Errorf("server store info = %+v", info)
	}
}

func TestServerViaConfigURL(t *testing.T) {
	ts := newTestAPIServer(t, struct{ auth, graph bool }{})
	dir := t.TempDir()

	cfg := &config.Config{}
	cfg.CLI.URL = ts.URL
	cfg.Store.Backend = config.BackendSQLite
	cfg.Store.Path = filepath.Join(dir, "local.db")
	cfgPath := filepath.Join(dir, "recall.json")
	if err := config.Save(cfgPath, cfg); err != nil {
		t.Fatal(err)
	}

	// cli.url switches data commands to server mode without --server.
	out := mustRunCLI(t, "--config", cfgPath, "store", "info", "-o", "json")
	var info storeInfoOutput
	decodeJSON(t, out, &info)
	if info.Mode != "server" {
		t.Errorf("cli.url should enable server mode, got %+v", info)
	}

	// An explicit empty --server flag overrides cli.url back to local mode.
	out = mustRunCLI(t, "--config", cfgPath, "--server", "", "store", "info", "-o", "json")
	decodeJSON(t, out, &info)
	if info.Mode != "local" {
		t.Errorf("--server \"\" should restore local mode, got %+v", info)
	}
}

func TestServerGraphAndReason(t *testing.T) {
	ts := newTestAPIServer(t, struct{ auth, graph bool }{graph: true})
	cfgPath, _ := writeSQLiteConfig(t)
	base := []string{"--config", cfgPath, "--server", ts.URL}

	out := mustRunCLI(t, append(base, "graph", "alice", "-o", "json")...)
	var ge graphEntityOutput
	decodeJSON(t, out, &ge)
	if ge.Mode != "server" || ge.Entity.ID != "alice" || len(ge.Neighbors) != 1 {
		t.Errorf("server graph = %+v", ge)
	}

	out = mustRunCLI(t, append(base, "reason", "--from", "alice", "--to", "berlin", "-o", "json")...)
	var ro reasonOutput
	decodeJSON(t, out, &ro)
	if ro.Mode != "server" || len(ro.Paths) == 0 {
		t.Errorf("server reason = %+v", ro)
	}

	// Natural-language reasoning against the server.
	mustRunCLI(t, append(base, "reason", "Alice")...)

	// Local-only commands are rejected in server mode.
	if _, err := runCLI(t, append(base, "graph", "list")...); err == nil || !strings.Contains(err.Error(), "local-only") {
		t.Errorf("graph list in server mode: expected local-only error, got %v", err)
	}
	if _, err := runCLI(t, append(base, "eval")...); err == nil || !strings.Contains(err.Error(), "local-only") {
		t.Errorf("eval in server mode: expected local-only error, got %v", err)
	}
	if _, err := runCLI(t, append(base, "store", "backup", "x.db")...); err == nil || !strings.Contains(err.Error(), "local-only") {
		t.Errorf("store backup in server mode: expected local-only error, got %v", err)
	}
	if _, err := runCLI(t, append(base, "store", "restore", "x.db")...); err == nil || !strings.Contains(err.Error(), "local-only") {
		t.Errorf("store restore in server mode: expected local-only error, got %v", err)
	}
}

func TestServerAuth(t *testing.T) {
	ts := newTestAPIServer(t, struct{ auth, graph bool }{auth: true})
	cfgPath, _ := writeSQLiteConfig(t)
	notes := writeNotesFile(t)

	// Without a key the upload is rejected (per-document failure, reported
	// in the output rather than as a command error).
	out, err := runCLI(t, "--config", cfgPath, "--server", ts.URL, "upload", notes)
	if err != nil {
		t.Fatalf("upload command: %v", err)
	}
	if !strings.Contains(out, "1 failed") || !strings.Contains(out, "401") {
		t.Fatalf("expected 401 failure without key, got: %s", out)
	}

	// The --api-key flag authenticates.
	out = mustRunCLI(t, "--config", cfgPath, "--server", ts.URL, "--api-key", "test-key", "upload", notes)
	if !strings.Contains(out, "uploaded 1 document(s)") {
		t.Errorf("upload with key: %s", out)
	}

	// The key can also come from the environment (RECALL__CLI__API_KEY).
	t.Setenv("RECALL__CLI__API_KEY", "test-key")
	out = mustRunCLI(t, "--config", cfgPath, "--server", ts.URL, "upload", notes)
	if !strings.Contains(out, "uploaded 1 document(s)") {
		t.Errorf("upload with env key: %s", out)
	}
}
