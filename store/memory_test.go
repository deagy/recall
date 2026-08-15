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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func newTestStore(t *testing.T) *MemoryStore {
	t.Helper()
	cfg := Config{
		Namespace:      "test",
		Embedder:       embedder.NewMockEmbedder(384),
		ChunkerFactory: chunker.NewFixed,
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err, "NewMemoryStore should not fail")
	return s
}

func newTestStoreWithMocks(t *testing.T) (*MemoryStore, *chunker.MockChunker) {
	t.Helper()
	mockChunker := new(chunker.MockChunker)

	// Default mock behavior: return one chunk
	dim := 384
	mockChunker.On("Chunk", mock.Anything, mock.Anything).Return([]*core.Chunk{
		{ID: "chunk-0", Content: "test content", DocumentRef: "doc1", Embedding: make([]float32, dim)},
	}, nil)

	cfg := Config{
		Namespace:      "test",
		Embedder:       embedder.NewMockEmbedder(384),
		ChunkerFactory: func(cfg chunker.Config) chunker.Chunker { return mockChunker },
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err, "NewMemoryStore should not fail")
	return s, mockChunker
}

func uploadAndVerify(t *testing.T, s *MemoryStore, docID, content string) {
	t.Helper()
	doc := core.NewDocument(docID, "Test", "source")
	err := s.Upload(context.Background(), doc, content)
	require.NoError(t, err, "Upload should not fail for doc %s", docID)
	assert.Greater(t, s.Count(), 0, "expected chunks after upload")
}

func TestMemoryStore_UploadAndSearch(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a statically typed compiled programming language designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson.")

	results, err := s.Search(context.Background(), "Go programming language", index.DefaultSearchOptions(5))
	require.NoError(t, err, "Search should not fail")
	require.NotEmpty(t, results, "expected search results")
	assert.Equal(t, "doc1", results[0].Chunk.DocumentRef, "expected doc1 as top result")
}

func TestMemoryStore_UploadNilDoc(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	err := s.Upload(context.Background(), nil, "content")
	assert.ErrorIs(t, err, core.ErrInvalidChunk, "expected ErrInvalidChunk for nil doc")
}

func TestMemoryStore_UploadEmptyContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "")
	assert.ErrorIs(t, err, core.ErrInvalidChunk, "expected ErrInvalidChunk for empty content")
}

func TestMemoryStore_GetChunk(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Some content here for testing the get chunk functionality of the store.")

	results, err := s.Search(context.Background(), "content", index.DefaultSearchOptions(1))
	require.NoError(t, err)
	require.NotEmpty(t, results, "no results to get chunk from")

	chunk, ok := s.GetChunk(results[0].Chunk.ID)
	require.True(t, ok, "expected to find chunk")
	assert.Equal(t, results[0].Chunk.ID, chunk.ID, "chunk IDs should match")

	_, ok = s.GetChunk("nonexistent")
	assert.False(t, ok, "expected not to find nonexistent chunk")
}

func TestMemoryStore_DeleteDocument(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Some content for deletion testing that should produce multiple chunks when split properly by the chunker.")

	require.Greater(t, s.Count(), 0, "expected chunks after upload")

	err := s.DeleteDocument("doc1")
	require.NoError(t, err, "DeleteDocument should not fail")

	assert.Equal(t, 0, s.Count(), "expected 0 chunks after delete")

	err = s.DeleteDocument("nonexistent")
	assert.ErrorIs(t, err, core.ErrNotFound, "expected ErrNotFound for nonexistent doc")
}

func TestMemoryStore_Namespaces(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "First document with enough text to be chunked properly by the chunker implementation.")
	uploadAndVerify(t, s, "doc2", "Second document with enough text to be chunked properly by the chunker implementation too.")

	ns := s.Namespaces()
	require.NotEmpty(t, ns, "expected at least one namespace")
	assert.Contains(t, ns, "test", "expected default namespace")
}

func TestMemoryStore_MultipleDocuments(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a statically typed compiled programming language designed at Google by Robert Griesemer Rob Pike and Ken Thompson for building scalable network services.")
	uploadAndVerify(t, s, "doc2", "Python is a high-level general-purpose programming language designed by Guido van Rossum emphasizing code readability.")

	total := s.Count()
	require.GreaterOrEqual(t, total, 2, "expected at least 2 chunks")

	results, err := s.Search(context.Background(), "programming language", index.DefaultSearchOptions(10))
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(results), 2, "expected at least 2 results for 'programming language'")
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
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected hybrid search results")

	// At least one result should contain "Go" or "programming"
	found := false
	for _, r := range results {
		content := r.Chunk.Content
		if strings.Contains(strings.ToLower(content), "go") || strings.Contains(strings.ToLower(content), "programming") {
			found = true
			break
		}
	}
	assert.True(t, found, "expected at least one result containing 'Go' or 'programming'")
}

func TestMemoryStore_HybridSearchPureVector(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language systems programming created at Google for building scalable network services.")

	opts := index.DefaultSearchOptions(5)
	opts.BM25Weight = 0 // Pure vector

	vecResults, err := s.Search(context.Background(), "Go programming", opts)
	require.NoError(t, err)
	hybridResults, err := s.SearchHybrid(context.Background(), "Go programming", opts)
	require.NoError(t, err)

	assert.Len(t, vecResults, len(hybridResults), "expected same number of results")
}

func TestMemoryStore_HybridSearchPureBM25(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language systems programming created at Google for building scalable network services.")

	opts := index.DefaultSearchOptions(5)
	opts.BM25Weight = 1 // Pure BM25

	hybridResults, err := s.SearchHybrid(context.Background(), "Go programming", opts)
	require.NoError(t, err)
	require.NotEmpty(t, hybridResults, "expected hybrid search results with pure BM25")
}

func TestMemoryStore_HybridSearchWithRRF(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go is a programming language for systems programming")
	uploadAndVerify(t, s, "doc2", "Python is a programming language for general purposes")

	opts := index.DefaultSearchOptions(5)
	opts.Fusion = fuse.NewRRFFusion(60)

	results, err := s.SearchHybrid(context.Background(), "programming language", opts)
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected hybrid search results with RRF fusion")
}

// --- Mock-based tests using in-package mockery mocks ---

func TestMemoryStore_WithMockChunker(t *testing.T) {
	s, mockChunker := newTestStoreWithMocks(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "test content")
	require.NoError(t, err)

	assert.Equal(t, 1, s.Count(), "expected 1 chunk")

	mockChunker.AssertExpectations(t)
}

func TestMemoryStore_MockChunkerCustomResult(t *testing.T) {
	s, mockChunker := newTestStoreWithMocks(t)
	defer s.Close()

	// Override with custom chunk
	dim := 384
	mockChunker.On("Chunk", mock.Anything, "custom").Return([]*core.Chunk{
		{ID: "custom-0", Content: "custom", DocumentRef: "doc1", Embedding: make([]float32, dim)},
	}, nil)

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "custom")
	require.NoError(t, err)

	assert.Equal(t, 1, s.Count(), "expected 1 chunk")

	mockChunker.AssertExpectations(t)
}
