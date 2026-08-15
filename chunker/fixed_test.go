package chunker

import (
	"testing"

	"github.com/deagy/recall/core"
)

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