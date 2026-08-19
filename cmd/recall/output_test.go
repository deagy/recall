package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// newTestPrinter returns a printer writing to a buffer.
func newTestPrinter(format string) (*printer, *bytes.Buffer) {
	var buf bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&buf)
	return newPrinter(cmd, format), &buf
}

func TestPrinter_JSON(t *testing.T) {
	p, buf := newTestPrinter("json")
	err := p.emit(map[string]any{"name": "a<b", "count": 2}, nil)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, buf.String())
	}
	if decoded["name"] != "a<b" {
		t.Errorf("HTML escaping should be disabled, got %v", decoded["name"])
	}
}

func TestPrinter_YAML(t *testing.T) {
	p, buf := newTestPrinter("yaml")
	err := p.emit(map[string]any{"name": "recall", "count": 2}, nil)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	var decoded map[string]any
	if err := yaml.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("output is not valid YAML: %v\n%s", err, buf.String())
	}
	if decoded["name"] != "recall" {
		t.Errorf("name = %v", decoded["name"])
	}
}

func TestPrinter_TableRender(t *testing.T) {
	p, buf := newTestPrinter("table")
	rendered := false
	err := p.emit(map[string]any{"ignored": true}, func(p *printer) {
		rendered = true
		tw := p.table("rank", "score")
		tw.Flush()
	})
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	if !rendered {
		t.Error("table renderer was not invoked")
	}
	if !strings.Contains(buf.String(), "RANK") || !strings.Contains(buf.String(), "SCORE") {
		t.Errorf("header not upper-cased: %q", buf.String())
	}
}

func TestPrinter_TableFallsBackToJSON(t *testing.T) {
	p, buf := newTestPrinter("table")
	if err := p.emit(map[string]any{"k": "v"}, nil); err != nil {
		t.Fatalf("emit: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("expected JSON fallback, got: %s", buf.String())
	}
}

func TestSnippet(t *testing.T) {
	if got := snippet("short", 10); got != "short" {
		t.Errorf("short text changed: %q", got)
	}
	if got := snippet("line one\nline two", 100); strings.Contains(got, "\n") {
		t.Errorf("newlines not flattened: %q", got)
	}
	got := snippet("abcdefghij", 4)
	if got != "abcd…" {
		t.Errorf("snippet = %q, want abcd…", got)
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("short", 10); got != "short" {
		t.Errorf("short text changed: %q", got)
	}
	got := truncate("abcdefghij", 4)
	if !strings.HasPrefix(got, "abcd") || !strings.Contains(got, "(truncated)") {
		t.Errorf("truncate = %q", got)
	}
}
