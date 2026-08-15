package chunker

import (
	"testing"

	"github.com/deagy/recall/core"
)

func BenchmarkRecursiveChunker_SmallDoc(b *testing.B) {
	c := NewRecursive(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Hello world. This is a test document. "

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkRecursiveChunker_LargeDoc(b *testing.B) {
	c := NewRecursive(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "This is a long document with many sentences. \n\n"
	for i := 0; i < 100; i++ {
		content += "Additional paragraph " + string(rune('A'+i%26)) + ". \n\n"
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}

func BenchmarkRecursiveChunker_VeryLargeDoc(b *testing.B) {
	c := NewRecursive(DefaultConfig())
	doc := &core.Document{ID: "doc1"}
	content := "Lorem ipsum dolor sit amet. \n\n"
	for i := 0; i < 1000; i++ {
		content += "Sentence number " + string(rune('0'+i%10)) + ". \n\n"
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Chunk(doc, content)
	}
}
