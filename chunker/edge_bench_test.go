package chunker

import (
	"strings"
	"testing"

	"github.com/deagy/recall/core"
)

func BenchmarkFixedChunker_UnicodeContent(b *testing.B) {
	c := NewFixed(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	// Unicode content with various scripts
	content := "Hello world 你好世界 안녕하세요 مرحبا שלום Γειά σου"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkFixedChunker_SingleWord(b *testing.B) {
	c := NewFixed(Config{MaxTokens: 512, MinChunkSize: 50, OverlapTokens: 50})
	doc := &core.Document{ID: "doc1"}
	// Single very long word
	content := strings.Repeat("a", 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkFixedChunker_SmallOverlapping(b *testing.B) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 5 // Very small overlap
	c := NewFixed(cfg)
	doc := &core.Document{ID: "doc1"}
	content := strings.Repeat("word ", 200)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkFixedChunker_LargeOverlap(b *testing.B) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 200 // Very large overlap
	c := NewFixed(cfg)
	doc := &core.Document{ID: "doc1"}
	content := strings.Repeat("word ", 200)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkRecursiveChunker_UnicodeContent(b *testing.B) {
	c := NewRecursive(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Hello world 你好世界 안녕하세요 مرحبا שלום Γειά σου"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkRecursiveChunker_SingleWord(b *testing.B) {
	c := NewRecursive(Config{MaxTokens: 512, MinChunkSize: 50})
	doc := &core.Document{ID: "doc1"}
	content := strings.Repeat("a", 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkRecursiveChunker_MultipleParagraphs(b *testing.B) {
	c := NewRecursive(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	var sb strings.Builder
	for i := 0; i < 50; i++ {
		sb.WriteString("This is paragraph number " + string(rune('0'+i%10)) + ". It contains multiple sentences. ")
		sb.WriteString("Another sentence to make it longer. ")
		sb.WriteString("\n\n")
	}
	content := sb.String()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}
