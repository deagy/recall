package ingest

import (
	"path/filepath"
	"testing"
)

func TestProgress_Counters(t *testing.T) {
	var events []string
	p := NewProgress()
	p.OnDocument = func(id, outcome string) { events = append(events, id+":"+outcome) }
	p.SetPhase("load")
	p.Loaded(3)
	p.Upload("a")
	p.Skip("b")
	p.Fail("c")

	loaded, skipped, uploaded, failed := p.Counters()
	if loaded != 3 || skipped != 1 || uploaded != 1 || failed != 1 {
		t.Errorf("counters: %d %d %d %d", loaded, skipped, uploaded, failed)
	}
	if len(events) != 3 || events[0] != "a:uploaded" || events[1] != "b:skipped" || events[2] != "c:failed" {
		t.Errorf("events: %v", events)
	}
	if p.Phase != "load" {
		t.Errorf("phase: %q", p.Phase)
	}
	if got := p.Summary(); got == "" {
		t.Error("empty summary")
	}
}

func TestDeduplicator_SaveLoad(t *testing.T) {
	d := NewDeduplicator()
	d.Mark("hello")
	d.Mark("world")
	path := filepath.Join(t.TempDir(), "dedup.json")
	if err := d.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	d2, err := LoadDeduplicator(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !d2.IsDuplicate("hello") || !d2.IsDuplicate("world") || d2.IsDuplicate("nope") {
		t.Error("round-trip mismatch")
	}
	if d2.Len() != 2 {
		t.Errorf("len: %d", d2.Len())
	}
}

func TestDeduplicator_MissingFile(t *testing.T) {
	d, err := LoadDeduplicator(filepath.Join(t.TempDir(), "none.json"))
	if err != nil {
		t.Fatalf("missing file should be empty, got %v", err)
	}
	if d.Len() != 0 {
		t.Errorf("len: %d", d.Len())
	}
}

func TestContentHash(t *testing.T) {
	if ContentHash("a") == ContentHash("b") {
		t.Error("different content must hash differently")
	}
	if ContentHash("a") != ContentHash("a") {
		t.Error("same content must hash the same")
	}
}
