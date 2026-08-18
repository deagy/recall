package store

import (
	"context"
	"fmt"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// hnswDocContent returns the content used for corpus documents so tests can
// query with the exact same text that was uploaded.
func hnswDocContent(i int) string {
	return fmt.Sprintf("Incremental mirror document number %d with some text.", i)
}

// uploadHNSWCorpus uploads enough documents to build the in-memory HNSW mirror.
func uploadHNSWCorpus(t *testing.T, s *SQLiteStore) {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < index.HNSWThreshold+1; i++ {
		doc := core.NewDocument(fmt.Sprintf("doc-%d", i), fmt.Sprintf("Doc %d", i), "t.txt")
		require.NoError(t, s.Upload(ctx, doc, hnswDocContent(i)), "upload %d should not fail", i)
	}
	require.NotNil(t, s.hnsw, "HNSW mirror should be built after crossing the threshold")
}

func TestSQLiteStore_Upload_HNSWIncremental(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	uploadHNSWCorpus(t, s)

	// An upload after mirror activation must be findable through the HNSW path.
	content := "Brand new document that arrives after the HNSW mirror was built."
	doc := core.NewDocument("doc-new", "New Doc", "t.txt")
	require.NoError(t, s.Upload(ctx, doc, content))
	assert.Equal(t, index.HNSWThreshold+2, s.Count(), "new chunk should be counted")

	results, err := s.Search(ctx, content, index.SearchOptions{TopK: 5})
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results from HNSW search")
	assert.Equal(t, "doc-new::chunk-0", results[0].Chunk.ID,
		"chunk uploaded after activation should be the top match for its own content")
}

func TestSQLiteStore_DeleteChunk_HNSWMirrorPruned(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	// Two extra documents keep the mirror above the threshold after a delete.
	uploadHNSWCorpus(t, s)
	for i := index.HNSWThreshold + 1; i <= index.HNSWThreshold+2; i++ {
		doc := core.NewDocument(fmt.Sprintf("doc-%d", i), fmt.Sprintf("Doc %d", i), "t.txt")
		require.NoError(t, s.Upload(ctx, doc, hnswDocContent(i)))
	}

	require.NoError(t, s.DeleteChunk(ctx, "doc-1::chunk-0"))
	assert.Equal(t, index.HNSWThreshold+2, s.Count(), "deleted chunk should not be counted")
	_, ok := s.GetChunk("doc-1::chunk-0")
	assert.False(t, ok, "deleted chunk should not be retrievable")

	// Search must not surface the deleted chunk even though its node may
	// still exist in the HNSW mirror.
	results, err := s.Search(ctx, hnswDocContent(1), index.SearchOptions{TopK: 10})
	require.NoError(t, err)
	for _, r := range results {
		assert.NotEqual(t, "doc-1::chunk-0", r.Chunk.ID, "deleted chunk must not be returned")
	}
}

func TestSQLiteStore_DeleteDocument_HNSWMirrorPruned(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	uploadHNSWCorpus(t, s)
	for i := index.HNSWThreshold + 1; i <= index.HNSWThreshold+2; i++ {
		doc := core.NewDocument(fmt.Sprintf("doc-%d", i), fmt.Sprintf("Doc %d", i), "t.txt")
		require.NoError(t, s.Upload(ctx, doc, hnswDocContent(i)))
	}

	require.NoError(t, s.DeleteDocument(ctx, "doc-2"))
	assert.Equal(t, index.HNSWThreshold+2, s.Count(), "deleted document chunk should not be counted")

	results, err := s.Search(ctx, hnswDocContent(2), index.SearchOptions{TopK: 10})
	require.NoError(t, err)
	for _, r := range results {
		assert.NotEqual(t, "doc-2::chunk-0", r.Chunk.ID, "deleted document chunk must not be returned")
	}
}
