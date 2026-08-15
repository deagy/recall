package chunker

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/deagy/recall/core"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MaxTokens != 512 {
		t.Errorf("expected MaxTokens 512, got %d", cfg.MaxTokens)
	}
	if cfg.MinChunkSize != 50 {
		t.Errorf("expected MinChunkSize 50, got %d", cfg.MinChunkSize)
	}
	if cfg.OverlapTokens != 50 {
		t.Errorf("expected OverlapTokens 50, got %d", cfg.OverlapTokens)
	}
}

func TestFixedChunker_EmptyContent(t *testing.T) {
	c := NewFixed(DefaultConfig())
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if chunks != nil {
		t.Errorf("expected nil chunks, got %d", len(chunks))
	}
}

func TestFixedChunker_SmallContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 512
	cfg.MinChunkSize = 10
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "Hello world, this is a small piece of text."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Content != content {
		t.Errorf("expected content to match, got %q", chunks[0].Content)
	}
	if chunks[0].DocumentRef != "doc-1" {
		t.Errorf("expected DocumentRef 'doc-1', got %q", chunks[0].DocumentRef)
	}
}

func TestFixedChunker_LargeContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 50 // small for testing
	cfg.MinChunkSize = 10
	cfg.OverlapTokens = 10
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	// Create content that will need multiple chunks
	var sb strings.Builder
	for i := 0; i < 20; i++ {
		sb.WriteString(strings.Repeat("word ", 30))
		sb.WriteString("\n\n")
	}
	content := sb.String()

	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	// Verify all chunks reference the same document
	for _, chunk := range chunks {
		if chunk.DocumentRef != "doc-1" {
			t.Errorf("expected DocumentRef 'doc-1', got %q", chunk.DocumentRef)
		}
		if chunk.ChunkIndex < 0 {
			t.Errorf("expected non-negative ChunkIndex, got %d", chunk.ChunkIndex)
		}
	}
}

func TestFixedChunker_MetadataPropagation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	doc.Metadata = map[string]core.Value{
		"author": core.String{Value: "Alice"},
		"date":   core.Number{Value: 2024},
	}
	content := "Short text."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Metadata["author"].String() != "Alice" {
		t.Errorf("expected author 'Alice', got %q", chunks[0].Metadata["author"].String())
	}
}

func TestRecursiveChunker_EmptyContent(t *testing.T) {
	c := NewRecursive(DefaultConfig())
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if chunks != nil {
		t.Errorf("expected nil chunks, got %d", len(chunks))
	}
}

func TestRecursiveChunker_SmallContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "Hello world."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestRecursiveChunker_ParagraphSplitting(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 5 // extremely small to force splitting
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "First paragraph with enough words to exceed the tiny limit here.\n\nSecond paragraph also with enough words to exceed the limit here.\n\nThird paragraph with enough words to exceed the limit here too."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks from paragraph splitting, got %d", len(chunks))
	}
}

func TestRecursiveChunker_VeryLongContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 10 // tiny for testing
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	// Create a very long string that must be split many times
	content := strings.Repeat("a very long sentence that exceeds the limit. ", 200)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for very long content, got %d", len(chunks))
	}

	// Verify all chunks are within size limit
	maxChars := cfg.MaxTokens * 4
	for i, chunk := range chunks {
		runes := utf8.RuneCountInString(chunk.Content)
		if runes > maxChars {
			t.Errorf("chunk %d has %d runes, exceeds max %d", i, runes, maxChars)
		}
	}
}

func TestFixedChunker_ChunkIDs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := strings.Repeat("word ", 100)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify unique chunk IDs
	seen := make(map[string]bool)
	for _, chunk := range chunks {
		if seen[chunk.ID] {
			t.Errorf("duplicate chunk ID: %s", chunk.ID)
		}
		seen[chunk.ID] = true
		if !strings.HasPrefix(chunk.ID, "doc-1::chunk-") {
			t.Errorf("unexpected chunk ID format: %s", chunk.ID)
		}
	}
}
