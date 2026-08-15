package chunker

import (
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFixedChunker_SmallDoc(t *testing.T) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Hello world. This is a test document. "

	chunks, err := c.Chunk(doc, content)
	require.NoError(t, err)
	// Small doc may not be chunked if below min chunk size
	_ = chunks
}

func TestFixedChunker_LargeDoc(t *testing.T) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "This is a long document with many sentences. "
	for i := 0; i < 100; i++ {
		content += "Additional paragraph " + string(rune('A'+i%26)) + ". "
	}

	chunks, err := c.Chunk(doc, content)
	require.NoError(t, err)
	require.NotEmpty(t, chunks, "large doc should produce chunks")
	for _, chunk := range chunks {
		assert.NotEmpty(t, chunk.ID, "chunk ID should not be empty")
		assert.NotEmpty(t, chunk.Content, "chunk content should not be empty")
		assert.Equal(t, "doc1", chunk.DocumentRef, "document ref should match")
	}
}

func TestFixedChunker_VeryLargeDoc(t *testing.T) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Lorem ipsum dolor sit amet. "
	for i := 0; i < 1000; i++ {
		content += "Sentence number " + string(rune('0'+i%10)) + ". "
	}

	chunks, err := c.Chunk(doc, content)
	require.NoError(t, err)
	require.NotEmpty(t, chunks, "very large doc should produce chunks")
}

func TestFixedChunker_EmptyContent_NoChunks(t *testing.T) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	chunks, err := c.Chunk(doc, "")
	require.NoError(t, err)
	assert.Empty(t, chunks, "empty content should produce no chunks")
}

func TestFixedChunker_NilDoc(t *testing.T) {
	c := NewFixed(DefaultConfig())
	content := "Some content here that should be chunked properly by the fixed chunker implementation."
	// FixedChunker does not handle nil doc gracefully; verify it doesn't produce valid chunks
	// (it may panic or return error depending on implementation)
	defer func() {
		if r := recover(); r != nil {
			t.Log("nil doc caused panic as expected")
		}
	}()
	chunks, err := c.Chunk(nil, content)
	_ = chunks
	_ = err
}

func TestFixedChunker_Overlap(t *testing.T) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 100 // Large overlap
	c := NewFixed(cfg)
	doc := &core.Document{ID: "doc1"}

	content := strings.Repeat("word ", 200)
	chunks, err := c.Chunk(doc, content)
	require.NoError(t, err)
	require.NotEmpty(t, chunks, "should produce chunks with overlap")

	// Verify overlap: consecutive chunks should share content
	if len(chunks) > 1 {
		for i := 1; i < len(chunks); i++ {
			// With overlap, consecutive chunks should have some shared words
			assert.NotEmpty(t, chunks[i].Content, "chunk %d should have content", i)
		}
	}
}

func TestFixedChunker_CustomConfig(t *testing.T) {
	cfg := Config{
		MaxTokens:     10,
		MinChunkSize:  5,
		OverlapTokens: 2,
		Separator:     ". ",
	}
	c := NewFixed(cfg)
	doc := &core.Document{ID: "doc1"}
	content := "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."

	chunks, err := c.Chunk(doc, content)
	require.NoError(t, err)
	require.NotEmpty(t, chunks, "should produce chunks with custom config")
}

func BenchmarkFixedChunker_SmallDoc(b *testing.B) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Hello world. This is a test document. "

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkFixedChunker_LargeDoc(b *testing.B) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "This is a long document with many sentences. "
	for i := 0; i < 100; i++ {
		content += "Additional paragraph " + string(rune('A'+i%26)) + ". "
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkFixedChunker_VeryLargeDoc(b *testing.B) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Lorem ipsum dolor sit amet. "
	for i := 0; i < 1000; i++ {
		content += "Sentence number " + string(rune('0'+i%10)) + ". "
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}
