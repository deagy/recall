package ingest

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIncremental_SkipAndMark(t *testing.T) {
	inc, _ := NewIncremental("")
	if inc.ShouldSkip("a", "h1") {
		t.Error("unknown doc should not skip")
	}
	inc.Mark("a", "h1")
	if !inc.ShouldSkip("a", "h1") {
		t.Error("unchanged doc should skip")
	}
	if inc.ShouldSkip("a", "h2") {
		t.Error("changed doc should not skip")
	}
	inc.Mark("a", "h2")
	if !inc.ShouldSkip("a", "h2") {
		t.Error("updated doc should skip after mark")
	}
	inc.Forget("a")
	if inc.ShouldSkip("a", "h2") {
		t.Error("forgotten doc should not skip")
	}
}

func TestIncremental_SaveLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inc.json")
	inc, _ := NewIncremental(path)
	inc.Mark("a", "h1")
	inc.Mark("b", "h2")
	if err := inc.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}
	inc2, err := NewIncremental(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if inc2.Len() != 2 || !inc2.ShouldSkip("a", "h1") || !inc2.ShouldSkip("b", "h2") {
		t.Errorf("round-trip mismatch: len=%d", inc2.Len())
	}
}

func TestIncremental_SaveNoop(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inc.json")
	inc, _ := NewIncremental(path)
	inc.Mark("a", "h1")
	if err := inc.Save(); err != nil {
		t.Fatal(err)
	}
	// Second save without changes should be a no-op (file untouched).
	st1, _ := os.Stat(path)
	if err := inc.Save(); err != nil {
		t.Fatal(err)
	}
	st2, _ := os.Stat(path)
	if !st1.ModTime().Equal(st2.ModTime()) {
		t.Error("expected no rewrite when state is clean")
	}
}

func TestIncremental_MissingFile(t *testing.T) {
	inc, err := NewIncremental(filepath.Join(t.TempDir(), "none.json"))
	if err != nil || inc.Len() != 0 {
		t.Errorf("missing file should be empty: %v %d", err, inc.Len())
	}
}

func TestIncremental_BadJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewIncremental(path); err == nil {
		t.Error("expected decode error")
	}
}
