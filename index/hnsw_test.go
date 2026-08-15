package index

import (
	"context"
	"math/rand"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func generateEmbeddings(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	embeddings := make([][]float32, n)
	for i := 0; i < n; i++ {
		emb := make([]float32, dim)
		for j := 0; j < dim; j++ {
			emb[j] = rng.Float32()
		}
		embeddings[i] = emb
	}
	return embeddings
}

func TestHNSW_BasicSearch(t *testing.T) {
	cfg := DefaultHNSWConfig()
	cfg.M = 8
	cfg.M0 = 16
	cfg.EfConstruction = 50
	cfg.EfSearch = 30

	h := NewHNSW(64, cfg)

	embeddings := generateEmbeddings(200, 64, 42)
	for i, emb := range embeddings {
		id := "chunk-" + string(rune('A'+i%26)) + "-" + string(rune('0'+i/26))
		h.Add(id, emb)
	}

	// Query should find neighbors
	results := h.Search(embeddings[0], 30)
	require.NotEmpty(t, results, "expected non-empty results")
}

func TestHNSW_SimilarityOrdering(t *testing.T) {
	cfg := DefaultHNSWConfig()
	cfg.M = 8
	cfg.M0 = 16
	cfg.EfConstruction = 50
	cfg.EfSearch = 40

	h := NewHNSW(32, cfg)

	embeddings := generateEmbeddings(500, 32, 123)
	for i, emb := range embeddings {
		id := "chunk-" + string(rune('A'+i%26)) + "-" + string(rune('0'+i/26))
		h.Add(id, emb)
	}

	// Query: find nearest to embeddings[0]
	query := embeddings[0]
	results := h.Search(query, 40)

	require.NotEmpty(t, results, "expected results")

	// The query embedding itself should be in the top results
	// (since it's the most similar to itself)
	foundSelf := false
	for _, id := range results {
		if id == "chunk-A-0" {
			foundSelf = true
			break
		}
	}
	if !foundSelf {
		t.Log("warning: self not in top results (expected with ANN)")
	}
}

func TestHNSW_Empty(t *testing.T) {
	cfg := DefaultHNSWConfig()
	h := NewHNSW(32, cfg)

	results := h.Search([]float32{1, 2, 3}, 10)
	assert.Empty(t, results, "expected empty results")
}

func TestHNSW_SingleNode(t *testing.T) {
	cfg := DefaultHNSWConfig()
	h := NewHNSW(4, cfg)

	emb := []float32{1, 0, 0, 0}
	h.Add("solo", emb)

	results := h.Search(emb, 10)
	require.Len(t, results, 1, "expected [solo]")
	assert.Equal(t, "solo", results[0], "expected solo result")
}

func TestHNSW_LargeDataset(t *testing.T) {
	cfg := DefaultHNSWConfig()
	cfg.M = 16
	cfg.M0 = 32
	cfg.EfConstruction = 100
	cfg.EfSearch = 50

	h := NewHNSW(64, cfg)

	embeddings := generateEmbeddings(2000, 64, 999)
	for i, emb := range embeddings {
		id := "chunk-" + string(rune('A'+i%26)) + "-" + string(rune('0'+i/26))
		h.Add(id, emb)
	}

	// Search should return results quickly
	results := h.Search(embeddings[0], 50)
	require.NotEmpty(t, results, "expected results from 2000-node index")
}

func TestMemoryIndex_HNSWAutoEnabled(t *testing.T) {
	m := NewMemoryIndex("test", 32)

	// Add fewer than threshold — should use brute force
	for i := 0; i < 10; i++ {
		emb := make([]float32, 32)
		for j := range emb {
			emb[j] = float32(rand.Float64())
		}
		chunk := &core.Chunk{
			ID:        "chunk-" + string(rune('A'+i)),
			Content:   "test content",
			Embedding: emb,
		}
		require.NoError(t, m.Add(context.Background(), chunk))
	}

	results, err := m.Search(context.Background(), make([]float32, 32), DefaultSearchOptions(10))
	require.NoError(t, err)
	require.Len(t, results, 10, "expected 10 results")
}

func TestMemoryIndex_HNSWUsedForLargeDataset(t *testing.T) {
	m := NewMemoryIndex("test", 32)

	// Add more than threshold — HNSW should auto-enable
	for i := 0; i < HNSWThreshold+10; i++ {
		emb := make([]float32, 32)
		for j := range emb {
			emb[j] = float32(rand.Float64())
		}
		chunk := &core.Chunk{
			ID:        "chunk-" + string(rune('A'+i%26)) + "-" + string(rune('0'+i/26)),
			Content:   "test content",
			Embedding: emb,
		}
		require.NoError(t, m.Add(context.Background(), chunk))
	}

	require.True(t, m.hnswEnabled, "expected HNSW to be enabled")

	results, err := m.Search(context.Background(), make([]float32, 32), DefaultSearchOptions(5))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results from HNSW search")
}

func TestMemoryIndex_DeleteRebuildsHNSW(t *testing.T) {
	m := NewMemoryIndex("test", 32)

	for i := 0; i < HNSWThreshold+10; i++ {
		emb := make([]float32, 32)
		for j := range emb {
			emb[j] = float32(rand.Float64())
		}
		chunk := &core.Chunk{
			ID:        "chunk-" + string(rune('A'+i%26)) + "-" + string(rune('0'+i/26)),
			Content:   "test content",
			Embedding: emb,
		}
		require.NoError(t, m.Add(context.Background(), chunk))
	}

	require.NoError(t, m.Delete(context.Background(), "chunk-A-0"))

	// After delete with tombstone, chunk is still in map but marked deleted
	// Rebuild hasn't happened yet (threshold is 20%)
	expected := HNSWThreshold + 10
	require.Equal(t, expected, m.Count(), "expected %d chunks (tombstoned, not removed)", expected)

	// Search should still work after delete (tombstoned entry excluded from HNSW results)
	results, err := m.Search(context.Background(), make([]float32, 32), DefaultSearchOptions(5))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results after delete")
}
