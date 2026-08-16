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

func TestDocument_AddTag_EmptyTag(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.AddTag("")
	// AddTag may not filter empty strings
	_ = d.Tags
}

func TestDocument_AddTag_MultipleEmptyTags(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.AddTag("")
	d.AddTag("")
	d.AddTag("")
	// AddTag may not filter empty strings
	_ = d.Tags
}

func TestDocument_Metadata_Nil(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	if d.Metadata == nil {
		t.Error("expected non-nil metadata")
	}
}

func TestDocument_Metadata_SetAndGet(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.Metadata["key"] = String{Value: "value"}
	if d.Metadata["key"].String() != "value" {
		t.Error("expected 'value'")
	}
}

func TestDocument_Metadata_UpdateExisting(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.Metadata["key"] = String{Value: "original"}
	d.Metadata["key"] = String{Value: "updated"}
	if d.Metadata["key"].String() != "updated" {
		t.Error("expected 'updated'")
	}
}

func TestDocument_Metadata_DeleteKey(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.Metadata["key"] = String{Value: "value"}
	delete(d.Metadata, "key")
	if d.Metadata["key"] != nil {
		t.Error("expected nil after delete")
	}
}

func TestDocument_ChunkCount_Zero(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.ChunkCount = 0
	if d.ChunkCount != 0 {
		t.Error("expected 0")
	}
}

func TestDocument_ChunkCount_Large(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.ChunkCount = 1000
	if d.ChunkCount != 1000 {
		t.Error("expected 1000")
	}
}

func TestDocument_CreatedAt_NotZero(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	if d.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}
}

func TestDocument_UpdatedAt_NotZero(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	if d.UpdatedAt.IsZero() {
		t.Error("expected non-zero UpdatedAt")
	}
}

func TestDocument_UpdatedAt_AfterCreatedAt(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	if d.UpdatedAt.Before(d.CreatedAt) {
		t.Error("UpdatedAt should be >= CreatedAt")
	}
}
