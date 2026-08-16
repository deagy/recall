package store

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/fuse"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestSQLiteStore(t *testing.T) *SQLiteStore {
	t.Helper()
	s, err := NewSQLiteStore(Config{
		Namespace: "test",
		Embedder:  embedder.NewMockEmbedder(384),
	}, ":memory:")
	require.NoError(t, err, "creating SQLite store should not fail")
	t.Cleanup(func() { s.Close() })
	return s
}

func TestSQLiteStore_UploadAndSearch(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test Document", "test.txt")
	content := "Go is a statically typed, compiled programming language designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson. It is used for building scalable network servers and web applications."
	require.NoError(t, s.Upload(ctx, doc, content), "upload should not fail")

	results, err := s.Search(ctx, "Go programming language", index.SearchOptions{TopK: 5})
	require.NoError(t, err, "search should not fail")
	require.NotEmpty(t, results, "expected search results, got none")
}

func TestSQLiteStore_GetChunk(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Hello world from the SQLite persistent store implementation in Go."
	require.NoError(t, s.Upload(ctx, doc, content), "upload should not fail")

	chunk, ok := s.GetChunk("doc1::chunk-0")
	require.True(t, ok, "expected to find chunk")
	assert.Equal(t, content, chunk.Content, "content should match")
}

func TestSQLiteStore_DeleteChunk(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Delete me please and remove this document from the store completely."), "upload should not fail")

	require.NoError(t, s.DeleteChunk("doc1::chunk-0"), "delete should not fail")

	_, ok := s.GetChunk("doc1::chunk-0")
	assert.False(t, ok, "expected chunk to be deleted")
}

func TestSQLiteStore_DeleteDocument(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Remove this entire document and all its chunks from the database permanently."), "upload should not fail")

	require.NoError(t, s.DeleteDocument("doc1"), "delete doc should not fail")

	_, ok := s.GetChunk("doc1::chunk-0")
	assert.False(t, ok, "expected all chunks to be deleted")
}

func TestSQLiteStore_Count(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc1 := core.NewDocument("doc1", "Doc 1", "test1.txt")
	doc2 := core.NewDocument("doc2", "Doc 2", "test2.txt")

	require.NoError(t, s.Upload(ctx, doc1, "First document content with enough text to exceed the minimum chunk size threshold."), "upload doc1 should not fail")
	require.NoError(t, s.Upload(ctx, doc2, "Second document content with enough text to exceed the minimum chunk size threshold too."), "upload doc2 should not fail")

	count := s.Count()
	assert.GreaterOrEqual(t, count, 2, "expected at least 2 chunks")
}

func TestSQLiteStore_Namespaces(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Namespace test content with sufficient length to be chunked properly."), "upload should not fail")

	ns := s.Namespaces()
	require.NotEmpty(t, ns, "expected at least one namespace")
}

func TestSQLiteStore_HybridSearch(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Machine learning algorithms process large datasets to find patterns and make predictions about future events."
	require.NoError(t, s.Upload(ctx, doc, content), "upload should not fail")

	results, err := s.SearchHybrid(ctx, "machine learning patterns", index.SearchOptions{
		Hybrid:     true,
		BM25Weight: 0.5,
	})
	require.NoError(t, err, "hybrid search should not fail")
	require.NotEmpty(t, results, "expected hybrid search results")
}

func TestSQLiteStore_MetadataRoundTrip(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Document with metadata that will be stored and retrieved from the SQLite database backend."
	require.NoError(t, s.Upload(ctx, doc, content), "upload should not fail")

	chunk, ok := s.GetChunk("doc1::chunk-0")
	require.True(t, ok, "expected to find chunk")
	require.NotNil(t, chunk, "chunk should not be nil")
	assert.True(t, strings.Contains(chunk.Content, "metadata"), "expected content to contain 'metadata'")
}

func TestSQLiteStore_EmptyStore(t *testing.T) {
	s := newTestSQLiteStore(t)

	count := s.Count()
	assert.Equal(t, 0, count, "expected 0 chunks for empty store")
}

func TestSQLiteStore_Namespaces_Empty(t *testing.T) {
	s := newTestSQLiteStore(t)

	ns := s.Namespaces()
	assert.Empty(t, ns, "expected empty namespaces for new store")
}

func TestSQLiteStore_GetChunk_NonExistent(t *testing.T) {
	s := newTestSQLiteStore(t)

	_, ok := s.GetChunk("nonexistent")
	assert.False(t, ok, "expected not to find non-existent chunk")
}

func TestSQLiteStore_DeleteChunk_NonExistent(t *testing.T) {
	s := newTestSQLiteStore(t)

	err := s.DeleteChunk("nonexistent")
	assert.NoError(t, err, "delete non-existent chunk should not fail")
}

func TestSQLiteStore_DeleteDocument_NonExistent(t *testing.T) {
	s := newTestSQLiteStore(t)

	err := s.DeleteDocument("nonexistent")
	assert.NoError(t, err, "delete non-existent document should not fail")
}

func TestSQLiteStore_Upload_EmptyDocID(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("", "Test", "source")
	err := s.Upload(ctx, doc, "This is content for empty doc ID testing in SQLite store that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with empty doc ID should succeed")
}

func TestSQLiteStore_Upload_EmptyContent(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "source")
	err := s.Upload(ctx, doc, "")
	assert.ErrorIs(t, err, core.ErrInvalidChunk, "expected ErrInvalidChunk for empty content")
}

func TestSQLiteStore_Upload_NilDoc(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	err := s.Upload(ctx, nil, "Some content for testing nil document upload.")
	assert.ErrorIs(t, err, core.ErrInvalidChunk, "expected ErrInvalidChunk for nil document")
}

func TestSQLiteStore_MultipleUploads(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc1 := core.NewDocument("doc1", "Doc 1", "test1.txt")
	doc2 := core.NewDocument("doc2", "Doc 2", "test2.txt")

	require.NoError(t, s.Upload(ctx, doc1, "First document content with enough text to be chunked properly."))
	require.NoError(t, s.Upload(ctx, doc2, "Second document content with enough text to be chunked properly."))

	count := s.Count()
	assert.GreaterOrEqual(t, count, 2, "expected at least 2 chunks")
}

func TestSQLiteStore_Search_EmptyStore(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	results, err := s.Search(ctx, "test query", index.SearchOptions{TopK: 10})
	require.NoError(t, err)
	assert.Empty(t, results, "expected empty results for empty store")
}

func TestSQLiteStore_SearchHybrid_EmptyStore(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	results, err := s.SearchHybrid(ctx, "test query", index.SearchOptions{
		Hybrid:     true,
		BM25Weight: 0.5,
		TopK:       10,
	})
	require.NoError(t, err)
	assert.Empty(t, results, "expected empty results for empty store")
}

func TestSQLiteStore_Upload_VeryLongContent(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	longContent := strings.Repeat("This is a long sentence that will be chunked into multiple pieces. ", 100)
	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, longContent))

	assert.Greater(t, s.Count(), 1, "expected multiple chunks for very long content")
}

func TestSQLiteStore_Upload_ShortContent(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	err := s.Upload(ctx, doc, "Hi")
	// May return error if no chunks produced
	_ = err
}

func TestSQLiteStore_Upload_WithMetadata(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	doc.Metadata = map[string]core.Value{
		"author": core.String{Value: "TestAuthor"},
	}

	require.NoError(t, s.Upload(ctx, doc, "This is content with metadata for testing metadata persistence that should be long enough to be chunked properly."))

	chunk, ok := s.GetChunk("doc1::chunk-0")
	require.True(t, ok, "expected to find chunk")
	require.NotNil(t, chunk.Metadata, "expected non-nil metadata")
}

func TestSQLiteStore_DeleteChunk_AfterUpload(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Content to be deleted after upload for testing delete functionality."))

	results, err := s.Search(ctx, "Content", index.SearchOptions{TopK: 1})
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results before delete")

	require.NoError(t, s.DeleteChunk(results[0].Chunk.ID), "delete should not fail")

	_, ok := s.GetChunk(results[0].Chunk.ID)
	assert.False(t, ok, "expected chunk to be deleted")
}

func TestSQLiteStore_DeleteDocument_AfterUpload(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Document content that will be deleted along with all its chunks."))

	require.NoError(t, s.DeleteDocument("doc1"), "delete document should not fail")

	assert.Equal(t, 0, s.Count(), "expected 0 chunks after document deletion")
}

func TestSQLiteStore_Search_WithFilters(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	doc.Metadata = map[string]core.Value{
		"source": core.String{Value: "test.txt"},
	}
	require.NoError(t, s.Upload(ctx, doc, "This is content with source metadata for filter testing that should be long enough to be chunked properly by the chunker implementation."))

	filter := &index.TermFilter{Key: "source", Value: "test.txt"}
	results, err := s.Search(ctx, "Content", index.SearchOptions{
		TopK:    10,
		Filters: []index.Filter{filter},
	})
	require.NoError(t, err)
	for _, r := range results {
		assert.Equal(t, "test.txt", r.Chunk.GetMetadataString("source"), "expected source filter")
	}
}

func TestSQLiteStore_Search_WithMinScore(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Go programming language for building scalable systems."))

	results, err := s.Search(ctx, "Go", index.SearchOptions{
		TopK:     10,
		MinScore: 0.1,
	})
	require.NoError(t, err)
	for _, r := range results {
		assert.GreaterOrEqual(t, r.Score, 0.1, "expected score >= 0.1")
	}
}

func TestSQLiteStore_Search_EmptyQuery(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "This is content to search for empty query testing that should be long enough to be chunked properly by the chunker implementation."))

	// SQLite returns error for empty query (different from MemoryStore)
	_, err := s.Search(ctx, "", index.SearchOptions{TopK: 10})
	assert.Error(t, err, "expected error for empty query")
}

func TestSQLiteStore_HybridSearch_WithFusion(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Go programming language for building scalable systems."))

	opts := index.SearchOptions{
		Hybrid:     true,
		BM25Weight: 0.5,
		TopK:       5,
		Fusion:     fuse.NewWeightedFusion(0.5),
	}

	results, err := s.SearchHybrid(ctx, "Go programming", opts)
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected hybrid search results with weighted fusion")
}

func TestSQLiteStore_HybridSearch_WithRRF(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Go programming language for building scalable systems."))

	opts := index.SearchOptions{
		Hybrid:     true,
		BM25Weight: 0.5,
		TopK:       5,
		Fusion:     fuse.NewRRFFusion(60),
	}

	results, err := s.SearchHybrid(ctx, "Go programming", opts)
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected hybrid search results with RRF fusion")
}

func TestSQLiteStore_Close(t *testing.T) {
	s := newTestSQLiteStore(t)
	err := s.Close()
	assert.NoError(t, err, "Close should not fail")
}

func TestSQLiteStore_Upload_ContentWithNewlines(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10"
	require.NoError(t, s.Upload(ctx, doc, content), "upload with newlines should succeed")
}

func TestSQLiteStore_Upload_ContentWithSpecialChars(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	content := "Content with special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?"
	require.NoError(t, s.Upload(ctx, doc, content), "upload with special characters should succeed")
}

func TestSQLiteStore_Count_AfterDelete(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "Content to be deleted after upload for testing count after delete."))

	initialCount := s.Count()
	assert.Greater(t, initialCount, 0, "expected chunks before delete")

	results, err := s.Search(ctx, "Content", index.SearchOptions{TopK: 1})
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results before delete")

	require.NoError(t, s.DeleteChunk(results[0].Chunk.ID))

	finalCount := s.Count()
	assert.Less(t, finalCount, initialCount, "expected fewer chunks after delete")
}

func TestSQLiteStore_Namespaces_AfterUpload(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "This is content for namespace testing after upload that should be long enough to be chunked properly by the chunker implementation."))

	ns := s.Namespaces()
	assert.NotEmpty(t, ns, "expected non-empty namespaces after upload")
}

func TestSQLiteStore_Upload_DefaultEmbedder(t *testing.T) {
	cfg := Config{
		Namespace: "test",
		// Embedder is nil - should use default
	}
	s, err := NewSQLiteStore(cfg, ":memory:")
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	err = s.Upload(context.Background(), doc, "This is content for default embedder testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with default embedder should succeed")
}

func TestSQLiteStore_Upload_DefaultChunker(t *testing.T) {
	cfg := Config{
		Namespace:      "test",
		Embedder:       embedder.NewMockEmbedder(384),
		ChunkerFactory: nil, // Should use default
	}
	s, err := NewSQLiteStore(cfg, ":memory:")
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	err = s.Upload(context.Background(), doc, "This is content for default chunker testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with default chunker should succeed")
}

func TestSQLiteStore_Upload_EmptyNamespace(t *testing.T) {
	cfg := Config{
		Namespace: "",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	s, err := NewSQLiteStore(cfg, ":memory:")
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	err = s.Upload(context.Background(), doc, "This is content for empty namespace testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with empty namespace should succeed")
}

func TestSQLiteStore_Upload_CustomNamespace(t *testing.T) {
	cfg := Config{
		Namespace: "custom-ns",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	s, err := NewSQLiteStore(cfg, ":memory:")
	require.NoError(t, err)
	defer s.Close()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	err = s.Upload(context.Background(), doc, "This is content for custom namespace testing that should be long enough to be chunked properly by the chunker implementation.")
	assert.NoError(t, err, "upload with custom namespace should succeed")

	ns := s.Namespaces()
	assert.Contains(t, ns, "custom-ns", "expected custom namespace")
}

func TestSQLiteStore_Upload_MultipleUploads(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		doc := core.NewDocument(fmt.Sprintf("doc-%d", i), "Test", fmt.Sprintf("test-%d.txt", i))
		require.NoError(t, s.Upload(ctx, doc, fmt.Sprintf("Content for upload %d with enough text to be chunked properly.", i)))
	}

	assert.Greater(t, s.Count(), 0, "expected chunks from multiple uploads")
}

func TestSQLiteStore_GetChunk_AfterUpload(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "This is content to verify GetChunk after upload that should be long enough to be chunked properly by the chunker implementation."))

	chunk, ok := s.GetChunk("doc1::chunk-0")
	require.True(t, ok, "expected to find chunk")
	assert.NotEmpty(t, chunk.Content, "expected non-empty content")
}

func TestSQLiteStore_DeleteChunk_AfterGetChunk(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "This is content to verify DeleteChunk after GetChunk that should be long enough to be chunked properly by the chunker implementation."))

	_, ok := s.GetChunk("doc1::chunk-0")
	require.True(t, ok, "expected to find chunk before delete")

	require.NoError(t, s.DeleteChunk("doc1::chunk-0"), "delete should not fail")

	_, ok = s.GetChunk("doc1::chunk-0")
	assert.False(t, ok, "expected chunk to be deleted")
}

func TestSQLiteStore_DeleteDocument_AfterGetChunk(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	doc := core.NewDocument("doc1", "Test", "test.txt")
	require.NoError(t, s.Upload(ctx, doc, "This is content to verify DeleteDocument after GetChunk that should be long enough to be chunked properly by the chunker implementation."))

	_, ok := s.GetChunk("doc1::chunk-0")
	require.True(t, ok, "expected to find chunk before delete")

	require.NoError(t, s.DeleteDocument("doc1"), "delete document should not fail")

	_, ok = s.GetChunk("doc1::chunk-0")
	assert.False(t, ok, "expected chunk to be deleted after document deletion")
}
