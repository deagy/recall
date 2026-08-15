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
