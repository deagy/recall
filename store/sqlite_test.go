package store

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

func newTestSQLiteStore(t *testing.T) *SQLiteStore {
	t.Helper()
	s, err := NewSQLiteStore(Config{
		Namespace: "test",
		Embedder:  embedder.NewMockEmbedder(384),
	}, ":memory:")
	if err != nil {
		t.Fatalf("creating SQLite store: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestSQLiteStore_UploadAndSearch(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test Document", "test.txt")
	content := "Go is a statically typed, compiled programming language designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson. It is used for building scalable network servers and web applications."
	if err := s.Upload(ctx, doc, content); err != nil {
		t.Fatalf("upload: %v", err)
	}

	results, err := s.Search(ctx, "Go programming language", index.SearchOptions{TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected search results, got none")
	}
}

func TestSQLiteStore_GetChunk(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Hello world from the SQLite persistent store implementation in Go."
	if err := s.Upload(ctx, doc, content); err != nil {
		t.Fatalf("upload: %v", err)
	}

	chunk, ok := s.GetChunk("doc1::chunk-0")
	if !ok {
		t.Fatal("expected to find chunk")
	}
	if chunk.Content != content {
		t.Errorf("expected content %q, got %q", content, chunk.Content)
	}
}

func TestSQLiteStore_DeleteChunk(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	if err := s.Upload(ctx, doc, "Delete me please and remove this document from the store completely."); err != nil {
		t.Fatalf("upload: %v", err)
	}

	if err := s.DeleteChunk("doc1::chunk-0"); err != nil {
		t.Fatalf("delete: %v", err)
	}

	_, ok := s.GetChunk("doc1::chunk-0")
	if ok {
		t.Fatal("expected chunk to be deleted")
	}
}

func TestSQLiteStore_DeleteDocument(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	if err := s.Upload(ctx, doc, "Remove this entire document and all its chunks from the database permanently."); err != nil {
		t.Fatalf("upload: %v", err)
	}

	if err := s.DeleteDocument("doc1"); err != nil {
		t.Fatalf("delete doc: %v", err)
	}

	_, ok := s.GetChunk("doc1::chunk-0")
	if ok {
		t.Fatal("expected all chunks to be deleted")
	}
}

func TestSQLiteStore_Count(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc1 := core.NewDocument("doc1", "Doc 1", "test1.txt")
	doc2 := core.NewDocument("doc2", "Doc 2", "test2.txt")

	if err := s.Upload(ctx, doc1, "First document content with enough text to exceed the minimum chunk size threshold."); err != nil {
		t.Fatalf("upload doc1: %v", err)
	}
	if err := s.Upload(ctx, doc2, "Second document content with enough text to exceed the minimum chunk size threshold too."); err != nil {
		t.Fatalf("upload doc2: %v", err)
	}

	count := s.Count()
	if count < 2 {
		t.Errorf("expected at least 2 chunks, got %d", count)
	}
}

func TestSQLiteStore_Namespaces(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	if err := s.Upload(ctx, doc, "Namespace test content with sufficient length to be chunked properly."); err != nil {
		t.Fatalf("upload: %v", err)
	}

	ns := s.Namespaces()
	if len(ns) == 0 {
		t.Fatal("expected at least one namespace")
	}
}

func TestSQLiteStore_HybridSearch(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Machine learning algorithms process large datasets to find patterns and make predictions about future events."
	if err := s.Upload(ctx, doc, content); err != nil {
		t.Fatalf("upload: %v", err)
	}

	results, err := s.SearchHybrid(ctx, "machine learning patterns", index.SearchOptions{
		Hybrid:       true,
		BM25Weight:   0.5,
	})
	if err != nil {
		t.Fatalf("hybrid search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected hybrid search results")
	}
}

func TestSQLiteStore_MetadataRoundTrip(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Document with metadata that will be stored and retrieved from the SQLite database backend."
	if err := s.Upload(ctx, doc, content); err != nil {
		t.Fatalf("upload: %v", err)
	}

	chunk, ok := s.GetChunk("doc1::chunk-0")
	if !ok {
		t.Fatal("expected to find chunk")
	}
	if chunk == nil {
		t.Fatal("chunk is nil")
	}
	if !strings.Contains(chunk.Content, "metadata") {
		t.Errorf("expected content to contain 'metadata', got %q", chunk.Content)
	}
}
