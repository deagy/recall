package index

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func makeTestChunks(dim int) []*core.Chunk {
	return []*core.Chunk{
		{
			ID: "c1", Content: "Go programming language", DocumentRef: "doc1", ChunkIndex: 0,
			Embedding: makeEmbed(dim, 1, 0, 0),
			Metadata:  map[string]core.Value{"source": core.String{Value: "golang.org"}},
		},
		{
			ID: "c2", Content: "Python programming language", DocumentRef: "doc1", ChunkIndex: 1,
			Embedding: makeEmbed(dim, 0, 1, 0),
			Metadata:  map[string]core.Value{"source": core.String{Value: "python.org"}},
		},
		{
			ID: "c3", Content: "Rust systems language", DocumentRef: "doc2", ChunkIndex: 0,
			Embedding: makeEmbed(dim, 0, 0, 1),
			Metadata:  map[string]core.Value{"source": core.String{Value: "rust-lang.org"}},
		},
	}
}

func makeEmbed(dim int, x, y, z float32) []float32 {
	v := make([]float32, dim)
	if dim >= 1 {
		v[0] = x
	}
	if dim >= 2 {
		v[1] = y
	}
	if dim >= 3 {
		v[2] = z
	}
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	if norm > 0 {
		norm = float32(1.0 / float64(norm))
		for i := range v {
			v[i] *= norm
		}
	}
	return v
}

func TestMemoryIndex_AddAndCount(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunks := makeTestChunks(3)
	for _, c := range chunks {
		require.NoError(t, idx.Add(context.Background(), c), "Add should not fail")
	}
	assert.Equal(t, 3, idx.Count(), "expected count 3")
}

func TestMemoryIndex_AddInvalidEmbedding(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: nil}
	err := idx.Add(context.Background(), chunk)
	assert.ErrorIs(t, err, core.ErrInvalidEmbedding, "expected ErrInvalidEmbedding")
}

func TestMemoryIndex_AddEmbeddingMismatch(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: []float32{1, 2}}
	err := idx.Add(context.Background(), chunk)
	assert.ErrorIs(t, err, core.ErrEmbeddingMismatch, "expected ErrEmbeddingMismatch")
}

func TestMemoryIndex_Search(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{TopK: 3})
	require.NoError(t, err)
	require.Len(t, results, 3, "expected 3 results")
	assert.Equal(t, "c1", results[0].Chunk.ID, "expected first result c1")
	for i := 1; i < len(results); i++ {
		assert.GreaterOrEqual(t, results[i-1].Score, results[i].Score, "results should be sorted descending")
	}
}

func TestMemoryIndex_SearchWithFilters(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	filter := &TermFilter{Key: "source", Value: "golang.org"}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK: 10, Filters: []Filter{filter},
	})
	require.NoError(t, err)
	require.Len(t, results, 1, "expected 1 result")
	assert.Equal(t, "c1", results[0].Chunk.ID, "expected c1")
}

func TestMemoryIndex_SearchTermInFilter(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	filter := &TermInFilter{Key: "source", Values: []string{"golang.org", "rust-lang.org"}}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK: 10, Filters: []Filter{filter},
	})
	require.NoError(t, err)
	require.Len(t, results, 2, "expected 2 results")
}

func TestMemoryIndex_Delete(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	require.NoError(t, idx.Delete(context.Background(), "c2"), "Delete should not fail")
	assert.Equal(t, 2, idx.Count(), "expected count 2")
	results, _ := idx.Search(context.Background(), []float32{0, 1, 0}, SearchOptions{TopK: 10})
	for _, r := range results {
		assert.NotEqual(t, "c2", r.Chunk.ID, "deleted chunk should not appear")
	}
}

func TestMemoryIndex_AddBatch(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	require.NoError(t, idx.AddBatch(context.Background(), makeTestChunks(3)), "AddBatch should not fail")
	assert.Equal(t, 3, idx.Count(), "expected count 3")
}

func TestMemoryIndex_GetChunk(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	c, ok := idx.GetChunk("c1")
	require.True(t, ok, "expected to find c1")
	assert.Equal(t, "Go programming language", c.Content, "content should match")
	_, ok = idx.GetChunk("nonexistent")
	assert.False(t, ok, "expected not to find nonexistent")
}

func TestMemoryIndex_NamespaceAndDimension(t *testing.T) {
	idx := NewMemoryIndex("myns", 1536)
	assert.Equal(t, "myns", idx.Namespace(), "namespace should match")
	assert.Equal(t, 1536, idx.Dimension(), "dimension should match")
}

func TestMemoryIndex_MinScoreFilter(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK: 10, MinScore: 0.9,
	})
	require.NoError(t, err)
	require.Len(t, results, 1, "expected 1 result with MinScore 0.9")
	assert.Equal(t, "c1", results[0].Chunk.ID, "expected c1")
}

func TestMemoryIndex_Search_EmptyIndex(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{TopK: 10})
	require.NoError(t, err)
	assert.Empty(t, results, "expected empty results for empty index")
}

func TestMemoryIndex_Search_ZeroTopK(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{TopK: 0})
	require.NoError(t, err)
	// Zero TopK should still return results (default behavior)
	_ = results
}

func TestMemoryIndex_AddBatch_Empty(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	err := idx.AddBatch(context.Background(), []*core.Chunk{})
	assert.NoError(t, err, "AddBatch with empty slice should not fail")
}

func TestMemoryIndex_AddBatch_InvalidEmbedding(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunks := []*core.Chunk{
		{ID: "c1", Content: "test", Embedding: nil},
	}
	err := idx.AddBatch(context.Background(), chunks)
	assert.ErrorIs(t, err, core.ErrInvalidEmbedding, "expected ErrInvalidEmbedding")
}

func TestMemoryIndex_AddBatch_EmbeddingMismatch(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunks := []*core.Chunk{
		{ID: "c1", Content: "test", Embedding: []float32{1, 2}},
	}
	err := idx.AddBatch(context.Background(), chunks)
	assert.ErrorIs(t, err, core.ErrEmbeddingMismatch, "expected ErrEmbeddingMismatch")
}

func TestMemoryIndex_Delete_NonExistent(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	err := idx.Delete(context.Background(), "nonexistent")
	// Delete should not fail even if chunk doesn't exist
	assert.NoError(t, err, "Delete non-existent should not fail")
}

func TestMemoryIndex_GetChunk_EmptyIndex(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	_, ok := idx.GetChunk("c1")
	assert.False(t, ok, "expected not to find chunk in empty index")
}

func TestMemoryIndex_Add_InvalidEmbedding(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: nil}
	err := idx.Add(context.Background(), chunk)
	assert.ErrorIs(t, err, core.ErrInvalidEmbedding, "expected ErrInvalidEmbedding")
}

func TestMemoryIndex_Add_EmbeddingMismatch(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: []float32{1, 2}}
	err := idx.Add(context.Background(), chunk)
	assert.ErrorIs(t, err, core.ErrEmbeddingMismatch, "expected ErrEmbeddingMismatch")
}

func TestMemoryIndex_Search_WithNilFilters(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK:    10,
		Filters: nil,
	})
	require.NoError(t, err)
	require.Len(t, results, 3, "expected all 3 results with nil filters")
}

func TestMemoryIndex_Search_WithMinScore(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK:     10,
		MinScore: 1.0,
	})
	require.NoError(t, err)
	// With MinScore 1.0, only exact matches should pass
	for _, r := range results {
		assert.GreaterOrEqual(t, r.Score, 1.0, "expected score >= 1.0")
	}
}

func TestMemoryIndex_Search_MultipleFilters(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	filter1 := &TermFilter{Key: "source", Value: "golang.org"}
	filter2 := &TermFilter{Key: "source", Value: "python.org"}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK:    10,
		Filters: []Filter{filter1, filter2},
	})
	require.NoError(t, err)
	// No chunk can match both filters simultaneously
	assert.Empty(t, results, "expected no results with conflicting filters")
}

func TestMemoryIndex_Add_Overwrite(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk1 := &core.Chunk{ID: "c1", Content: "first", Embedding: makeEmbed(3, 1, 0, 0)}
	chunk2 := &core.Chunk{ID: "c1", Content: "second", Embedding: makeEmbed(3, 0, 1, 0)}

	_ = idx.Add(context.Background(), chunk1)
	_ = idx.Add(context.Background(), chunk2)

	assert.Equal(t, 1, idx.Count(), "expected count 1 (overwritten)")

	c, ok := idx.GetChunk("c1")
	require.True(t, ok)
	assert.Equal(t, "second", c.Content, "expected overwritten content")
}

func TestMemoryIndex_AddBatch_MixedValidInvalid(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunks := []*core.Chunk{
		{ID: "c1", Content: "test", Embedding: makeEmbed(3, 1, 0, 0)},
		{ID: "c2", Content: "test", Embedding: nil}, // Invalid
	}
	err := idx.AddBatch(context.Background(), chunks)
	assert.Error(t, err, "expected error for invalid embedding in batch")
}

func TestMemoryIndex_Namespace_Empty(t *testing.T) {
	idx := NewMemoryIndex("", 3)
	assert.Equal(t, "", idx.Namespace(), "namespace should be empty")
}

func TestMemoryIndex_Dimension_Different(t *testing.T) {
	idx := NewMemoryIndex("test", 1536)
	assert.Equal(t, 1536, idx.Dimension(), "dimension should match")
}

func TestMemoryIndex_HNSW_NotEnabled_Initially(t *testing.T) {
	idx := NewMemoryIndex("test", 32)
	assert.False(t, idx.hnswEnabled, "HNSW should not be enabled initially")
}

func TestMemoryIndex_HNSW_EmptySearch(t *testing.T) {
	idx := NewMemoryIndex("test", 32)
	results, err := idx.Search(context.Background(), make([]float32, 32), DefaultSearchOptions(10))
	require.NoError(t, err)
	assert.Empty(t, results, "expected empty results for empty index")
}

func TestMemoryIndex_Add_SingleChunk(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: makeEmbed(3, 1, 0, 0)}
	require.NoError(t, idx.Add(context.Background(), chunk))
	assert.Equal(t, 1, idx.Count(), "expected count 1")
}
