package store

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

func BenchmarkSQLiteStore_Upload(b *testing.B) {
	s, err := NewSQLiteStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	}, ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "This is a test document with enough text to be chunked properly and stored in SQLite. "

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Upload(ctx, doc, content)
	}
}

func BenchmarkSQLiteStore_Search(b *testing.B) {
	s, err := NewSQLiteStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	}, ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()
	for i := 0; i < 50; i++ {
		doc := core.NewDocument(string(rune('a'+i%26)), "Test", "source")
		content := "This is test document content number " + string(rune('0'+i%10)) + " with enough text. "
		s.Upload(ctx, doc, content)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(ctx, "test document", index.DefaultSearchOptions(10))
	}
}

func BenchmarkSQLiteStore_Delete(b *testing.B) {
	s, err := NewSQLiteStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	}, ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "This is a test document with enough text to be chunked properly. "
	s.Upload(ctx, doc, content)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.DeleteChunk(ctx, "doc-1::chunk-0")
	}
}

func BenchmarkSQLiteStore_UploadLargeDoc(b *testing.B) {
	s, err := NewSQLiteStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	}, ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "This is a large document with many sentences. "
	for i := 0; i < 100; i++ {
		content += "Additional paragraph " + string(rune('A'+i%26)) + ". "
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Upload(ctx, doc, content)
	}
}

func BenchmarkSQLiteStore_Upload_MultipleDocs(b *testing.B) {
	s, err := NewSQLiteStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	}, ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc := core.NewDocument("doc-"+strings.Repeat("0", 3), "Test", "source")
		s.Upload(ctx, doc, strings.Repeat("word ", 50))
	}
}

func BenchmarkSQLiteStore_Search_WithFilters(b *testing.B) {
	s, err := NewSQLiteStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	}, ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()
	for i := 0; i < 50; i++ {
		doc := core.NewDocument(string(rune('a'+i%26)), "Test", "source")
		content := "This is test document content with enough text to be chunked. "
		s.Upload(ctx, doc, content)
	}

	opts := index.DefaultSearchOptions(10)
	opts.Filters = []index.Filter{
		&index.TermFilter{Key: "source", Value: "source"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(ctx, "test document", opts)
	}
}
