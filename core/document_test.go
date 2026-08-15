package core

import (
	"testing"
)

func TestNewDocument(t *testing.T) {
	d := NewDocument("doc-1", "Test Doc", "/path/to/file.txt")
	if d.ID != "doc-1" {
		t.Errorf("expected ID 'doc-1', got %q", d.ID)
	}
	if d.Title != "Test Doc" {
		t.Errorf("expected Title 'Test Doc', got %q", d.Title)
	}
	if d.Source != "/path/to/file.txt" {
		t.Errorf("expected Source '/path/to/file.txt', got %q", d.Source)
	}
	if d.Metadata == nil {
		t.Error("expected non-nil metadata map")
	}
	if d.ChunkCount != 0 {
		t.Errorf("expected ChunkCount 0, got %d", d.ChunkCount)
	}
}

func TestDocumentAddTag(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.AddTag("go")
	d.AddTag("rust")
	d.AddTag("go") // duplicate

	if len(d.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d: %v", len(d.Tags), d.Tags)
	}
}
