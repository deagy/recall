package store

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
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
