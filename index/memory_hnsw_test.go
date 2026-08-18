package index

import (
	"context"
	"fmt"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// addHNSWCorpus loads just enough chunks to activate the HNSW graph.
func addHNSWCorpus(t *testing.T, m *MemoryIndex, dim, seed int) []core.Chunk {
	t.Helper()
	ctx := context.Background()
	emb := generateEmbeddings(HNSWThreshold+1, dim, int64(seed))
	chunks := make([]core.Chunk, len(emb))
	for i, e := range emb {
		chunks[i] = core.Chunk{
			ID:        fmt.Sprintf("chunk-%d", i),
			Content:   fmt.Sprintf("corpus document %d", i),
			Embedding: e,
		}
		require.NoError(t, m.Add(ctx, &chunks[i]), "adding chunk %d should not fail", i)
	}
	require.True(t, m.hnswEnabled, "HNSW should be enabled after crossing the threshold")
	return chunks
}

func TestMemoryIndex_Add_HNSWIncremental(t *testing.T) {
	const dim = 16
	m := NewMemoryIndex("test", dim)

	addHNSWCorpus(t, m, dim, 7)

	// A chunk added after HNSW activation must be inserted into the graph
	// and returned by search.
	newEmb := makeEmbed(dim, 1, 0, 0)
	require.NoError(t, m.Add(context.Background(), &core.Chunk{
		ID: "chunk-new", Content: "arrived after activation", Embedding: newEmb,
	}))
	require.Equal(t, HNSWThreshold+2, m.Count())

	// Query with a generous ef: on the flat random-uniform landscape an ef of 5
	// gives no recall guarantee, but the new chunk must be retrievable at all.
	results, err := m.Search(context.Background(), newEmb, DefaultSearchOptions(50))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results from HNSW search")
	found := false
	for _, r := range results {
		if r.Chunk.ID == "chunk-new" {
			found = true
			break
		}
	}
	assert.True(t, found, "chunk added after activation should be found by searching its own embedding")
}

func TestMemoryIndex_AddBatch_HNSWIncremental(t *testing.T) {
	const dim = 16
	m := NewMemoryIndex("test", dim)

	addHNSWCorpus(t, m, dim, 11)

	batchEmb := generateEmbeddings(3, dim, 23)
	batch := make([]*core.Chunk, len(batchEmb))
	for i, e := range batchEmb {
		batch[i] = &core.Chunk{
			ID: fmt.Sprintf("batch-%d", i), Content: fmt.Sprintf("batch document %d", i), Embedding: e,
		}
	}
	require.NoError(t, m.AddBatch(context.Background(), batch))
	require.Equal(t, HNSWThreshold+4, m.Count())

	// The last chunk of the batch must be searchable through the HNSW graph.
	results, err := m.Search(context.Background(), batch[2].Embedding, DefaultSearchOptions(5))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results from HNSW search")
	assert.Equal(t, "batch-2", results[0].Chunk.ID,
		"batch chunk added after activation should be the top match for its own embedding")
}

func TestMemoryIndex_Delete_HNSWTombstoneFiltered(t *testing.T) {
	const dim = 16
	m := NewMemoryIndex("test", dim)

	// Five extra chunks keep the index above the threshold (and thus on the
	// HNSW code path) even after a deletion.
	corpus := addHNSWCorpus(t, m, dim, 5)
	require.NoError(t, m.Add(context.Background(), &core.Chunk{
		ID: "chunk-keep", Content: "kept", Embedding: makeEmbed(dim, 0, 1, 0),
	}))
	require.NoError(t, m.Add(context.Background(), &core.Chunk{
		ID: "chunk-keep2", Content: "kept too", Embedding: makeEmbed(dim, 0, 0, 1),
	}))

	require.NoError(t, m.Delete(context.Background(), "chunk-0"))

	assert.Equal(t, HNSWThreshold+2, m.Count(), "deleted chunk should not be counted")
	_, ok := m.GetChunk("chunk-0")
	assert.False(t, ok, "deleted chunk should not be retrievable")

	// Search must not surface the deleted chunk even though its node may
	// still exist in the HNSW graph.
	results, err := m.Search(context.Background(), corpus[0].Embedding, DefaultSearchOptions(10))
	require.NoError(t, err)
	for _, r := range results {
		assert.NotEqual(t, "chunk-0", r.Chunk.ID, "deleted chunk must not be returned by HNSW search")
	}
}

func TestMemoryIndex_Delete_HNSWRebuildAtTombstoneRatio(t *testing.T) {
	const dim = 16
	m := NewMemoryIndex("test", dim)

	corpus := addHNSWCorpus(t, m, dim, 9)

	// Delete well over the 20% tombstone threshold to force a rebuild.
	toDelete := (HNSWThreshold + 1) * 3 / 10
	deleted := make(map[string]bool, toDelete)
	for i := 0; i < toDelete; i++ {
		require.NoError(t, m.Delete(context.Background(), corpus[i].ID))
		deleted[corpus[i].ID] = true
	}

	assert.Equal(t, HNSWThreshold+1-toDelete, m.Count(), "all deleted chunks should be excluded from the count")
	assert.True(t, m.hnswEnabled, "HNSW should stay enabled after rebuild")

	results, err := m.Search(context.Background(), corpus[len(corpus)-1].Embedding, DefaultSearchOptions(10))
	require.NoError(t, err)
	require.NotEmpty(t, results, "expected results after tombstone rebuild")
	for _, r := range results {
		assert.False(t, deleted[r.Chunk.ID], "deleted chunk must not be returned after rebuild")
	}
}
