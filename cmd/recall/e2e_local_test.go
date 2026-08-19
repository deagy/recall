package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLocalUploadSearchHybridRAG(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)
	notes := writeNotesFile(t)

	out := mustRunCLI(t, "--config", cfgPath, "upload", notes)
	if !strings.Contains(out, "uploaded 1 document(s), 0 failed") {
		t.Fatalf("upload output: %s", out)
	}

	// Vector search with JSON output.
	out = mustRunCLI(t, "--config", cfgPath, "search", "garbage collection", "-o", "json")
	var so searchOutput
	decodeJSON(t, out, &so)
	if so.Mode != "local" || so.Hybrid {
		t.Errorf("search output meta = %+v", so)
	}
	if so.Count == 0 || len(so.Results) == 0 {
		t.Fatalf("expected results, got %+v", so)
	}
	if so.Results[0].Rank != 1 || so.Results[0].ID == "" {
		t.Errorf("first hit malformed: %+v", so.Results[0])
	}

	// Hybrid search.
	out = mustRunCLI(t, "--config", cfgPath, "hybrid-search", "garbage collection", "--bm25-weight", "0.7", "-o", "json")
	decodeJSON(t, out, &so)
	if !so.Hybrid || so.Count == 0 {
		t.Errorf("hybrid output = %+v", so)
	}

	// Table rendering for search.
	out = mustRunCLI(t, "--config", cfgPath, "search", "garbage collection")
	if !strings.Contains(out, "result(s) for") {
		t.Errorf("table output: %s", out)
	}

	// RAG (JSON + table).
	out = mustRunCLI(t, "--config", cfgPath, "rag", "What is Go?", "-o", "json")
	var ro ragOutput
	decodeJSON(t, out, &ro)
	if ro.Mode != "local" || ro.Answer == "" || ro.Context == "" {
		t.Errorf("rag output = %+v", ro)
	}
	if len(ro.Sources) == 0 || len(ro.Citations) == 0 {
		t.Errorf("rag should carry sources and citations: %+v", ro)
	}
	out = mustRunCLI(t, "--config", cfgPath, "rag", "What is Go?", "--smart-context", "--hybrid")
	if !strings.Contains(out, "answer:") || !strings.Contains(out, "context:") {
		t.Errorf("rag table output: %s", out)
	}

	// Store info.
	out = mustRunCLI(t, "--config", cfgPath, "store", "info", "-o", "json")
	var info storeInfoOutput
	decodeJSON(t, out, &info)
	if info.Mode != "local" || info.Backend != "sqlite" || !info.OK || !info.Connected {
		t.Errorf("store info = %+v", info)
	}
	if info.Chunks == 0 {
		t.Errorf("store info missing stats: %+v", info)
	}
}

func TestLocalUploadDirectoryAndFailures(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)

	// Directory upload recurses by default.
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	doc := "Rust ownership makes memory safety possible without a garbage collector."
	if err := os.WriteFile(filepath.Join(dir, "sub", "doc.txt"), []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	out := mustRunCLI(t, "--config", cfgPath, "upload", dir)
	if !strings.Contains(out, "uploaded 1 document(s), 0 failed") {
		t.Fatalf("directory upload output: %s", out)
	}

	// Missing path is reported as a failed document, not a fatal error.
	out = mustRunCLI(t, "--config", cfgPath, "upload", filepath.Join(t.TempDir(), "missing.txt"))
	if !strings.Contains(out, "uploaded 0 document(s), 1 failed") {
		t.Fatalf("missing upload output: %s", out)
	}

	// Unsupported extension is likewise a failed document.
	xyz := filepath.Join(t.TempDir(), "data.xyz")
	if err := os.WriteFile(xyz, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	out = mustRunCLI(t, "--config", cfgPath, "upload", xyz)
	if !strings.Contains(out, "1 failed") {
		t.Fatalf("unsupported upload output: %s", out)
	}

	// Upload requires at least one argument.
	if _, err := runCLI(t, "--config", cfgPath, "upload"); err == nil {
		t.Error("upload without args: expected error")
	}
}

func TestLocalNamespaceFiltering(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)
	notes := writeNotesFile(t)

	mustRunCLI(t, "--config", cfgPath, "--namespace", "notes", "upload", notes)

	out := mustRunCLI(t, "--config", cfgPath, "search", "garbage collection", "--namespace", "notes", "-o", "json")
	var so searchOutput
	decodeJSON(t, out, &so)
	if so.Count == 0 {
		t.Error("search in the upload namespace should find results")
	}

	out = mustRunCLI(t, "--config", cfgPath, "search", "garbage collection", "--namespace", "other", "-o", "json")
	decodeJSON(t, out, &so)
	if so.Count != 0 {
		t.Errorf("search in a foreign namespace should be empty, got %d", so.Count)
	}

	// Store info lists the namespace.
	out = mustRunCLI(t, "--config", cfgPath, "store", "info", "-o", "json")
	var info storeInfoOutput
	decodeJSON(t, out, &info)
	found := false
	for _, ns := range info.Namespaces {
		if ns == "notes" {
			found = true
		}
	}
	if !found {
		t.Errorf("namespaces = %v, want to contain notes", info.Namespaces)
	}
}

func TestInvalidOutputFormat(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)
	_, err := runCLI(t, "--config", cfgPath, "-o", "xml", "store", "info")
	if err == nil || !strings.Contains(err.Error(), "invalid output format") {
		t.Errorf("expected invalid output format error, got %v", err)
	}
}
