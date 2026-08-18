package store

import (
	"context"
	"fmt"
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

func newTestStoreWithMocks(t *testing.T) (*MemoryStore, *mockChunker) {
	t.Helper()
	mockChunker := new(mockChunker)

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
	assert.ErrorIs(t, err, core.ErrInvalidDocument, "expected ErrInvalidDocument for nil doc")
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

	err := s.DeleteDocument(context.Background(), "doc1")
	require.NoError(t, err, "DeleteDocument should not fail")

	assert.Equal(t, 0, s.Count(), "expected 0 chunks after delete")

	err = s.DeleteDocument(context.Background(), "nonexistent")
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

func TestMemoryStore_MultipleDocuments_Concurrent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "First document content with enough text to be chunked properly by the chunker.")
	uploadAndVerify(t, s, "doc2", "Second document content with enough text to be chunked properly by the chunker too.")

	count := s.Count()
	assert.GreaterOrEqual(t, count, 2, "expected at least 2 chunks")
}

func TestMemoryStore_DeleteNonExistentChunk(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	err := s.DeleteChunk(context.Background(), "nonexistent")
	assert.ErrorIs(t, err, core.ErrNotFound, "expected ErrNotFound")
}

func TestMemoryStore_DeleteDocumentNonExistent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	err := s.DeleteDocument(context.Background(), "nonexistent")
	assert.ErrorIs(t, err, core.ErrNotFound, "expected ErrNotFound")
}

func TestMemoryStore_Upload_EmptyDocID(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("", "Test", "source")
	err := s.Upload(context.Background(), doc, "Some content here for testing the upload with empty doc ID.")
	// Should succeed (doc is not nil, content is not empty)
	assert.NoError(t, err, "upload with empty doc ID should not fail")
}

func TestMemoryStore_Search_EmptyStore(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	results, err := s.Search(context.Background(), "test query", index.DefaultSearchOptions(10))
	require.NoError(t, err)
	assert.Empty(t, results, "expected empty results for empty store")
}

func TestMemoryStore_SearchHybrid_EmptyStore(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	opts := index.DefaultSearchOptions(10)
	opts.Hybrid = true
	opts.BM25Weight = 0.5

	results, err := s.SearchHybrid(context.Background(), "test query", opts)
	require.NoError(t, err)
	assert.Empty(t, results, "expected empty results for empty store")
}

func TestMemoryStore_Upload_MultipleUploads(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "First upload content with enough text to be chunked properly.")
	uploadAndVerify(t, s, "doc2", "Second upload content with enough text to be chunked properly.")
	uploadAndVerify(t, s, "doc3", "Third upload content with enough text to be chunked properly.")

	count := s.Count()
	assert.GreaterOrEqual(t, count, 3, "expected at least 3 chunks")
}

func TestMemoryStore_Namespaces_Empty(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	ns := s.Namespaces()
	assert.Empty(t, ns, "expected empty namespaces for new store")
}

func TestMemoryStore_Close(t *testing.T) {
	s := newTestStore(t)
	err := s.Close()
	assert.NoError(t, err, "Close should not fail")
}

func TestMemoryStore_Upload_ValidEmbeddingDimension(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "This is test content for embedding validation that should be long enough to be chunked properly by the chunker implementation.")
	// Should succeed with valid embedding dimension
	assert.NoError(t, err, "upload should succeed with matching embedding dimension")
}

func TestMemoryStore_Search_WithMinScore(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language for building scalable systems.")

	opts := index.DefaultSearchOptions(10)
	opts.MinScore = 0.99

	results, err := s.Search(context.Background(), "Go programming", opts)
	require.NoError(t, err)
	for _, r := range results {
		assert.GreaterOrEqual(t, r.Score, 0.99, "expected score >= 0.99")
	}
}

func TestMemoryStore_Search_WithFilters(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language for building scalable systems.")

	filter := &index.TermFilter{Key: "source", Value: "golang.org"}
	opts := index.DefaultSearchOptions(10)
	opts.Filters = []index.Filter{filter}

	results, err := s.Search(context.Background(), "Go", opts)
	require.NoError(t, err)
	for _, r := range results {
		assert.Equal(t, "golang.org", r.Chunk.GetMetadataString("source"), "expected source filter")
	}
}

func TestMemoryStore_GetChunk_NonExistent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	_, ok := s.GetChunk("nonexistent")
	assert.False(t, ok, "expected not to find non-existent chunk")
}

func TestMemoryStore_DeleteChunk_AfterUpload(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Content to be deleted after upload for testing delete functionality.")

	results, err := s.Search(context.Background(), "Content", index.DefaultSearchOptions(1))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results before delete")

	err = s.DeleteChunk(context.Background(), results[0].Chunk.ID)
	assert.NoError(t, err, "delete should not fail")

	_, ok := s.GetChunk(results[0].Chunk.ID)
	assert.False(t, ok, "expected chunk to be deleted")
}

func TestMemoryStore_DeleteDocument_AfterUpload(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Document content that will be deleted along with all its chunks.")

	err := s.DeleteDocument(context.Background(), "doc1")
	assert.NoError(t, err, "delete document should not fail")

	assert.Equal(t, 0, s.Count(), "expected 0 chunks after document deletion")
}

func TestMemoryStore_Upload_DifferentContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Content about Go programming language and its features.")
	uploadAndVerify(t, s, "doc2", "Content about Python programming language and its uses.")

	results, err := s.Search(context.Background(), "Go", index.DefaultSearchOptions(5))
	require.NoError(t, err)
	assert.NotEmpty(t, results, "expected results for Go query")
}

func TestMemoryStore_Upload_VeryLongContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	longContent := strings.Repeat("This is a long sentence that will be chunked into multiple pieces. ", 100)
	uploadAndVerify(t, s, "doc1", longContent)

	assert.Greater(t, s.Count(), 1, "expected multiple chunks for very long content")
}

func TestMemoryStore_Upload_ShortContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	// Short content may not produce chunks if below MinChunkSize
	err := s.Upload(context.Background(), core.NewDocument("doc1", "Test", "source"), "Hi")
	// May return error if no chunks produced
	_ = err
}

func TestMemoryStore_Search_EmptyQuery(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Content to search for empty query testing purposes.")

	results, err := s.Search(context.Background(), "", index.DefaultSearchOptions(10))
	require.NoError(t, err)
	// Empty query should still work (embedding-based search)
	_ = results
}

func TestMemoryStore_SearchHybrid_WithFusion(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language for building scalable systems.")

	opts := index.DefaultSearchOptions(5)
	opts.Hybrid = true
	opts.BM25Weight = 0.5
	opts.Fusion = fuse.NewWeightedFusion(0.5)

	results, err := s.SearchHybrid(context.Background(), "Go programming", opts)
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected hybrid search results with weighted fusion")
}

func TestMemoryStore_HybridSearch_WithRRF(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Go programming language for building scalable systems.")

	opts := index.DefaultSearchOptions(5)
	opts.Hybrid = true
	opts.BM25Weight = 0.5
	opts.Fusion = fuse.NewRRFFusion(60)

	results, err := s.SearchHybrid(context.Background(), "Go programming", opts)
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected hybrid search results with RRF fusion")
}

func TestMemoryStore_Count_AfterDelete(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "Content to be deleted after upload for testing count after delete.")

	initialCount := s.Count()
	assert.Greater(t, initialCount, 0, "expected chunks before delete")

	results, err := s.Search(context.Background(), "Content", index.DefaultSearchOptions(1))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results before delete")

	err = s.DeleteChunk(context.Background(), results[0].Chunk.ID)
	assert.NoError(t, err)

	finalCount := s.Count()
	assert.Less(t, finalCount, initialCount, "expected fewer chunks after delete")
}

func TestMemoryStore_Namespaces_AfterUpload(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	uploadAndVerify(t, s, "doc1", "This is content for namespace testing after upload that should be long enough to be chunked properly by the chunker implementation.")

	ns := s.Namespaces()
	assert.NotEmpty(t, ns, "expected non-empty namespaces after upload")
}

func TestMemoryStore_Upload_WithMetadata(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	doc.Metadata = map[string]core.Value{
		"author": core.String{Value: "TestAuthor"},
		"date":   core.Number{Value: 2024},
	}

	err := s.Upload(context.Background(), doc, "Content with metadata for testing metadata propagation.")
	assert.NoError(t, err, "upload with metadata should succeed")

	results, err := s.Search(context.Background(), "Content", index.DefaultSearchOptions(1))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results")

	chunk := results[0].Chunk
	assert.Equal(t, "TestAuthor", chunk.GetMetadataString("author"), "expected author metadata")
}

func TestMemoryStore_Upload_EmptyNamespace(t *testing.T) {
	cfg := Config{
		Namespace: "", // Empty namespace
		Embedder:  embedder.NewMockEmbedder(384),
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err = s.Upload(context.Background(), doc, "This is content for empty namespace testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with empty namespace should succeed")
}

func TestMemoryStore_Upload_CustomNamespace(t *testing.T) {
	cfg := Config{
		Namespace: "custom-ns",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err = s.Upload(context.Background(), doc, "This is content for custom namespace testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with custom namespace should succeed")

	ns := s.Namespaces()
	assert.Contains(t, ns, "custom-ns", "expected custom namespace")
}

func TestMemoryStore_Upload_MultipleNamespaces(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	// Upload to default namespace
	uploadAndVerify(t, s, "doc1", "This is content for default namespace that should be long enough to be chunked properly by the chunker implementation.")

	// Create a new store with different namespace
	cfg := Config{
		Namespace: "ns2",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	s2, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s2.Close()

	uploadAndVerify(t, s2, "doc2", "This is content for second namespace that should be long enough to be chunked properly by the chunker implementation.")

	ns1 := s.Namespaces()
	ns2 := s2.Namespaces()
	assert.Contains(t, ns1, "test", "expected test namespace")
	assert.Contains(t, ns2, "ns2", "expected ns2 namespace")
}

func TestMemoryStore_Upload_LargeDocument(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	largeContent := strings.Repeat("This is a large document with lots of content for testing. ", 1000)
	uploadAndVerify(t, s, "doc1", largeContent)

	assert.Greater(t, s.Count(), 10, "expected many chunks for large document")
}

func TestMemoryStore_Upload_Concurrent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(idx int) {
			doc := core.NewDocument(fmt.Sprintf("doc-%d", idx), "Test", "source")
			content := fmt.Sprintf("Content for concurrent upload %d with enough text to be chunked properly.", idx)
			err := s.Upload(context.Background(), doc, content)
			if err != nil {
				t.Errorf("concurrent upload %d failed: %v", idx, err)
			}
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	assert.Greater(t, s.Count(), 0, "expected chunks from concurrent uploads")
}

func TestMemoryStore_Upload_EmptyNamespaceConfig(t *testing.T) {
	cfg := Config{
		Namespace: "",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err = s.Upload(context.Background(), doc, "This is content for empty namespace config testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with empty namespace config should succeed")
}

func TestMemoryStore_Upload_DefaultEmbedder(t *testing.T) {
	cfg := Config{
		Namespace: "test",
		// Embedder is nil - should use default
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err = s.Upload(context.Background(), doc, "This is content for default embedder testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with default embedder should succeed")
}

func TestMemoryStore_Upload_DefaultChunker(t *testing.T) {
	cfg := Config{
		Namespace:      "test",
		Embedder:       embedder.NewMockEmbedder(384),
		ChunkerFactory: nil, // Should use default
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err = s.Upload(context.Background(), doc, "This is content for default chunker testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with default chunker should succeed")
}

func TestMemoryStore_Upload_CustomChunker(t *testing.T) {
	mockChunker := new(mockChunker)
	mockChunker.On("Chunk", mock.Anything, mock.Anything).Return([]*core.Chunk{
		{ID: "chunk-0", Content: "test content", DocumentRef: "doc1", Embedding: make([]float32, 384)},
	}, nil)

	cfg := Config{
		Namespace:      "test",
		Embedder:       embedder.NewMockEmbedder(384),
		ChunkerFactory: func(cfg chunker.Config) chunker.Chunker { return mockChunker },
	}
	s, err := NewMemoryStore(cfg)
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err = s.Upload(context.Background(), doc, "Content for custom chunker testing.")
	assert.NoError(t, err, "upload with custom chunker should succeed")
}

func TestMemoryStore_Upload_NilDocument(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	err := s.Upload(context.Background(), nil, "Some content for testing nil document upload.")
	assert.ErrorIs(t, err, core.ErrInvalidDocument, "expected ErrInvalidDocument for nil document")
}

func TestMemoryStore_Upload_EmptyContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "")
	assert.ErrorIs(t, err, core.ErrInvalidChunk, "expected ErrInvalidChunk for empty content")
}

func TestMemoryStore_Upload_WhitespaceContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "   ")
	// Whitespace-only content may produce no chunks
	_ = err
}

func TestMemoryStore_Upload_VeryShortContent(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(context.Background(), doc, "Hi")
	// Very short content may not produce chunks if below MinChunkSize
	_ = err
}

func TestMemoryStore_Upload_ContentWithNewlines(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10"
	err := s.Upload(context.Background(), doc, content)
	assert.NoError(t, err, "upload with newlines should succeed")
}

func TestMemoryStore_Upload_ContentWithSpecialChars(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "source")
	content := "Content with special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?"
	err := s.Upload(context.Background(), doc, content)
	assert.NoError(t, err, "upload with special characters should succeed")
}

func BenchmarkMemoryStore_Upload(b *testing.B) {
	s, err := NewMemoryStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	doc := core.NewDocument("doc-1", "Test", "source")
	content := "This is a test document with enough text to be chunked properly. "

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Upload(context.Background(), doc, content)
	}
}

func BenchmarkMemoryStore_Search(b *testing.B) {
	s, err := NewMemoryStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	ctx := context.Background()
	for i := 0; i < 100; i++ {
		doc := core.NewDocument(string(rune('a'+i%26)), "Test", "source")
		content := "This is a test document with enough text to be chunked properly. "
		s.Upload(ctx, doc, content)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(ctx, "test document", index.DefaultSearchOptions(10))
	}
}

func BenchmarkMemoryStore_Delete(b *testing.B) {
	s, err := NewMemoryStore(Config{
		Namespace:      "bench",
		Embedder:       embedder.NewMockEmbedder(32),
		ChunkerFactory: chunker.NewFixed,
	})
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
		s.DeleteChunk(context.Background(), "doc-1::chunk-0")
	}
}

// mockChunker is a test-only Chunker double (moved here from the
// previously exported chunker.MockChunker).
type mockChunker struct {
	mock.Mock
}

func (_m *mockChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	ret := _m.Called(doc, content)

	if len(ret) == 0 {
		panic("no return value specified for Chunk")
	}

	var r0 []*core.Chunk
	var r1 error
	if rf, ok := ret.Get(0).(func(*core.Document, string) ([]*core.Chunk, error)); ok {
		return rf(doc, content)
	}
	if rf, ok := ret.Get(0).(func(*core.Document, string) []*core.Chunk); ok {
		r0 = rf(doc, content)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*core.Chunk)
		}
	}

	if rf, ok := ret.Get(1).(func(*core.Document, string) error); ok {
		r1 = rf(doc, content)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}
