package store

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/fuse"
	"github.com/deagy/recall/index"
)

func newTestStore(t *testing.T) *MemoryStore {
	t.Helper()
	cfg := Config{
		Namespace:      "test",
		Embedder:       embedder.NewMockEmbedder(384),
		ChunkerFactory: chunker.NewFixed,
	}
	s, err := NewMemoryStore(cfg)
	if err != nil {
		t.Fatalf("NewMemoryStore failed: %v", err)
	}
	return s
}

func uploadAndVerify(t *testing.T, s *MemoryStore, docID, content string) {
	t.Helper()
	doc := core.NewDocument(docID, "Test", "source")
	err := s.Upload(context.Background(), doc, content)
	if err != nil {
		t.Fatalf("Upload failed for doc %s: %v", docID, err)
	}
	if s.Count() == 0 {
		t.Fatalf("expected chunks after upload, count is 0")
	}
}

func TestMemoryStore_UploadAndSearch(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a statically typed compiled programming language designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson.")

	results, err := s.Search(context.Background(), "Go programming language", index.DefaultSearchOptions(5))
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) == 0 {
		t.Error("expected search results")
	}
	if results[0].Chunk.DocumentRef != "doc1" {
		t.Errorf("expected doc1, got %s", results[0].Chunk.DocumentRef)
	}
}

func TestMemoryStore_UploadNilDoc(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	err := s.Upload(context.Background(), nil, "content")
	if err != core.ErrInvalidChunk {
		t.Errorf("expected ErrInvalidChunk, got %v", err)
	}
}

func TestMemoryStore_UploadEmptyContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "")
	if err != core.ErrInvalidChunk {
		t.Errorf("expected ErrInvalidChunk, got %v", err)
	}
}

func TestMemoryStore_GetChunk(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Some content here for testing the get chunk functionality of the store.")

	results, _ := s.Search(context.Background(), "content", index.DefaultSearchOptions(1))
	if len(results) == 0 {
		t.Fatal("no results to get chunk from")
	}
	chunk, ok := s.GetChunk(results[0].Chunk.ID)
	if !ok {
		t.Fatal("expected to find chunk")
	}
	if chunk.ID != results[0].Chunk.ID {
		t.Errorf("expected chunk ID %s, got %s", results[0].Chunk.ID, chunk.ID)
	}

	_, ok = s.GetChunk("nonexistent")
	if ok {
		t.Error("expected not to find nonexistent chunk")
	}
}

func TestMemoryStore_DeleteDocument(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Some content for deletion testing that should produce multiple chunks when split properly by the chunker.")

	if s.Count() == 0 {
		t.Fatal("expected chunks after upload")
	}

	err := s.DeleteDocument("doc1")
	if err != nil {
		t.Fatalf("DeleteDocument failed: %v", err)
	}

	if s.Count() != 0 {
		t.Errorf("expected 0 chunks after delete, got %d", s.Count())
	}

	err = s.DeleteDocument("nonexistent")
	if err != core.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestMemoryStore_Namespaces(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Content in the test namespace for verifying namespace tracking works correctly.")

	ns := s.Namespaces()
	if len(ns) != 1 || ns[0] != "test" {
		t.Errorf("expected [test], got %v", ns)
	}
}

func TestMemoryStore_DefaultNamespace(t *testing.T) {
	cfg := Config{
		Embedder: embedder.NewMockEmbedder(384),
	}
	s, err := NewMemoryStore(cfg)
	if err != nil {
		t.Fatalf("NewMemoryStore failed: %v", err)
	}
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Content in the default namespace for testing default namespace behavior.")

	ns := s.Namespaces()
	if len(ns) != 1 || ns[0] != "default" {
		t.Errorf("expected [default], got %v", ns)
	}
}

func TestMemoryStore_MultipleDocuments(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a programming language created at Google for systems programming and building scalable network services.")
	uploadAndVerify(t, s, "doc2", "Python is a high-level general-purpose programming language designed by Guido van Rossum emphasizing code readability.")

	total := s.Count()
	if total < 2 {
		t.Errorf("expected at least 2 chunks, got %d", total)
	}

	results, _ := s.Search(context.Background(), "programming language", index.DefaultSearchOptions(10))
	if len(results) < 2 {
		t.Errorf("expected at least 2 results for 'programming language', got %d", len(results))
	}
}

func TestMemoryStore_HybridSearch(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a statically typed compiled programming language designed at Google by Robert Griesemer Rob Pike and Ken Thompson.")
	uploadAndVerify(t, s, "doc2", "Python is a high-level general-purpose programming language that emphasizes code readability.")
	uploadAndVerify(t, s, "doc3", "The quick brown fox jumps over the lazy dog near the river bank.")

	opts := index.DefaultSearchOptions(5)
	opts.BM25Weight = 0.5 // Equal weighting

	results, err := s.SearchHybrid(context.Background(), "programming language Go", opts)
	if err != nil {
		t.Fatalf("SearchHybrid failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected hybrid search results")
	}

	// At least one result should contain "Go" or "programming"
	found := false
	for _, r := range results {
		content := r.Chunk.Content
		if strings.Contains(strings.ToLower(content), "go") || strings.Contains(strings.ToLower(content), "programming") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected at least one result containing 'Go' or 'programming'")
	}
}

func TestMemoryStore_HybridSearchPureVector(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language systems programming created at Google for building scalable network services.")

	opts := index.DefaultSearchOptions(5)
	opts.BM25Weight = 0 // Pure vector

	vecResults, _ := s.Search(context.Background(), "Go programming", opts)
	hybridResults, _ := s.SearchHybrid(context.Background(), "Go programming", opts)

	if len(vecResults) != len(hybridResults) {
		t.Errorf("expected same number of results (%d), got vec=%d hybrid=%d", len(vecResults), len(vecResults), len(hybridResults))
	}
}

func TestMemoryStore_HybridSearchPureBM25(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language systems programming created at Google for building scalable network services.")

	opts := index.DefaultSearchOptions(5)
	opts.BM25Weight = 1 // Pure BM25

	hybridResults, err := s.SearchHybrid(context.Background(), "Go programming", opts)
	if err != nil {
		t.Fatalf("SearchHybrid failed: %v", err)
	}

	if len(hybridResults) == 0 {
		t.Fatal("expected hybrid search results with pure BM25")
	}
}

func TestMemoryStore_HybridSearchWithRRF(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a programming language for systems programming")
	uploadAndVerify(t, s, "doc2", "Python is a programming language for general purposes")

	opts := index.DefaultSearchOptions(5)
	opts.Fusion = fuse.NewRRFFusion(60)

	results, err := s.SearchHybrid(context.Background(), "programming language", opts)
	if err != nil {
		t.Fatalf("SearchHybrid with RRF failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected hybrid search results with RRF fusion")
	}
}


