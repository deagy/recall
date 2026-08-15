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

func TestDocumentAddTag_EmptyTag(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.AddTag("")
	d.AddTag("go")

	// Empty string is still added as a tag (no filtering)
	if len(d.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d: %v", len(d.Tags), d.Tags)
	}
	// Find "go" in tags
	found := false
	for _, tag := range d.Tags {
		if tag == "go" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected 'go' in tags")
	}
}

func TestDocumentAddTag_MultipleDuplicates(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.AddTag("a")
	d.AddTag("b")
	d.AddTag("a")
	d.AddTag("b")
	d.AddTag("c")

	if len(d.Tags) != 3 {
		t.Errorf("expected 3 tags, got %d: %v", len(d.Tags), d.Tags)
	}
}

func TestDocument_EmptyFields(t *testing.T) {
	d := NewDocument("", "", "")

	if d.ID != "" {
		t.Errorf("expected empty ID, got %q", d.ID)
	}
	if d.Title != "" {
		t.Errorf("expected empty title, got %q", d.Title)
	}
	if d.Source != "" {
		t.Errorf("expected empty source, got %q", d.Source)
	}
	if d.Metadata == nil {
		t.Error("expected non-nil metadata map")
	}
	if len(d.Tags) != 0 {
		t.Errorf("expected 0 tags, got %d", len(d.Tags))
	}
}

func TestDocument_Metadata(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.Metadata["key"] = String{Value: "value"}
	d.Metadata["number"] = Number{Value: 42}

	if d.Metadata["key"].String() != "value" {
		t.Errorf("expected 'value', got %q", d.Metadata["key"].String())
	}
	if d.Metadata["number"].String() != "42" {
		t.Errorf("expected '42', got %q", d.Metadata["number"].String())
	}
}

func TestDocument_ChunkCount(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	if d.ChunkCount != 0 {
		t.Errorf("expected 0, got %d", d.ChunkCount)
	}

	d.ChunkCount = 5
	if d.ChunkCount != 5 {
		t.Errorf("expected 5, got %d", d.ChunkCount)
	}
}

