package store

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/fuse"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// keywordOnlyTargetContent contains rare tokens that only appear in the
// "keyword-only" document, so BM25 ranks it first while the hash-based
// MockEmbedder gives it no vector correlation with the query.
const (
	keywordOnlyTargetContent = "the quantum fluoroscope zephyr apparatus emits rare isotopes"
	keywordOnlyQuery         = "quantum fluoroscope zephyr apparatus"
)

func hybridTargetID(results []index.SearchResult, marker string) string {
	for _, r := range results {
		if strings.Contains(r.Chunk.Content, marker) {
			return r.Chunk.ID
		}
	}
	return ""
}

func resScore(results []index.SearchResult, id string) float64 {
	for _, r := range results {
		if r.Chunk.ID == id {
			return r.Score
		}
	}
	return 0
}

// --- fuseMap (MemoryStore fusion core) ---

func TestFuseMap_IncludesBM25OnlyChunks(t *testing.T) {
	vecResults := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", Content: "one"}, Score: 0.9},
		{Chunk: &core.Chunk{ID: "c2", Content: "two"}, Score: 0.5},
	}
	// "c3" is only in the BM25 results: it must be resolved via lookup.
	bm25Scores := map[string]float64{"c1": 0.2, "c3": 1.5}
	lookup := func(id string) *core.Chunk {
		if id == "c3" {
			return &core.Chunk{ID: "c3", Content: "three"}
		}
		return nil
	}

	res := fuseMap(vecResults, bm25Scores, lookup, index.SearchOptions{BM25Weight: 0.5})

	// fuseMap does not sort (callers do); compare by ID.
	require.Len(t, res, 3, "both vector and BM25-only chunks must be fused")
	byID := map[string]*core.Chunk{}
	for _, r := range res {
		byID[r.Chunk.ID] = r.Chunk
	}
	assert.Equal(t, "three", byID["c3"].Content, "lookup must resolve the full BM25-only chunk")
	assert.InDelta(t, 0.75, resScore(res, "c3"), 1e-9, "BM25-only chunk fused from 0.5 * 1.5")
}

func TestFuseMap_SkipsBM25OnlyChunkWhenNotInIndex(t *testing.T) {
	vecResults := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", Content: "one"}, Score: 0.9},
	}
	bm25Scores := map[string]float64{"deleted": 2.0}

	res := fuseMap(vecResults, bm25Scores, func(id string) *core.Chunk { return nil }, index.SearchOptions{BM25Weight: 0.5})

	require.Len(t, res, 1)
	assert.Equal(t, "c1", res[0].Chunk.ID, "chunks missing from the index (e.g. deleted) must not be returned")
}

func TestFuseMap_PureVectorIgnoresBM25(t *testing.T) {
	vecResults := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", Content: "one"}, Score: 0.9},
	}
	bm25Scores := map[string]float64{"c1": 0.2, "c3": 1.5}

	res := fuseMap(vecResults, bm25Scores, func(id string) *core.Chunk { return nil }, index.SearchOptions{BM25Weight: 0})

	require.Len(t, res, 1)
	assert.Equal(t, 0.9, res[0].Score, "BM25Weight=0 must be pure vector")
}

func TestFuseMap_CustomFusionIncludesBM25OnlyChunks(t *testing.T) {
	vecResults := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", Content: "one"}, Score: 0.9},
	}
	bm25Scores := map[string]float64{"c3": 1.5}
	lookup := func(id string) *core.Chunk {
		if id == "c3" {
			return &core.Chunk{ID: "c3", Content: "three"}
		}
		return nil
	}

	opts := index.SearchOptions{BM25Weight: 0.5, Fusion: fuse.NewRRFFusion(60)}
	res := fuseMap(vecResults, bm25Scores, lookup, opts)

	require.Len(t, res, 2)
	byID := map[string]float64{}
	for _, r := range res {
		byID[r.Chunk.ID] = r.Score
	}
	// RRF: c1 is rank 1 of the vector list, c3 is rank 1 of the BM25 list.
	assert.InDelta(t, 1.0/61.0, byID["c1"], 1e-9)
	assert.InDelta(t, 1.0/61.0, byID["c3"], 1e-9)
}

// --- MemoryStore.SearchHybrid end-to-end ---

func uploadHybridCorpus(t *testing.T, s *MemoryStore, distractors int) {
	t.Helper()
	ctx := context.Background()
	require.NoError(t, s.Upload(ctx, core.NewDocument("doc-target", "Target", "t.md"), keywordOnlyTargetContent))
	for i := 0; i < distractors; i++ {
		content := fmt.Sprintf("distractor number %d about mundane everyday topics and common ordinary phrases", i)
		require.NoError(t, s.Upload(ctx, core.NewDocument(fmt.Sprintf("doc-dist%d", i), "Dist", "d.md"), content))
	}
}

func TestMemoryStore_SearchHybrid_PureBM25WeightRanksKeywordMatchFirst(t *testing.T) {
	s := newTestStore(t)
	uploadHybridCorpus(t, s, 5)

	opts := index.SearchOptions{TopK: 10, BM25Weight: 1.0}
	results, err := s.SearchHybrid(context.Background(), keywordOnlyQuery, opts)
	require.NoError(t, err)
	require.NotEmpty(t, results)

	require.Equal(t, "doc-target::chunk-0", hybridTargetID(results, "quantum fluoroscope"),
		"BM25Weight=1 must rank the keyword-only match first (pure BM25)")
}

func TestMemoryStore_SearchHybrid_ReturnsBM25OnlyChunks(t *testing.T) {
	s := newTestStore(t)
	uploadHybridCorpus(t, s, 30)

	ctx := context.Background()
	// Sanity: with the hash-based mock embedder the target is not a strong
	// vector hit, so this exercises the BM25-only lookup path.
	vecResults, err := s.Search(ctx, keywordOnlyQuery, index.SearchOptions{TopK: 10})
	require.NoError(t, err)
	assert.Empty(t, hybridTargetID(vecResults, "quantum fluoroscope"),
		"test precondition: target must not be in vector Top-10")

	opts := index.SearchOptions{TopK: 10, BM25Weight: 0.5}
	results, err := s.SearchHybrid(ctx, keywordOnlyQuery, opts)
	require.NoError(t, err)

	target := hybridTargetID(results, "quantum fluoroscope")
	assert.NotEmpty(t, target, "keyword-only chunk must appear in hybrid results")
}

func TestMemoryStore_SearchHybrid_CustomFusionIncludesBM25OnlyChunks(t *testing.T) {
	s := newTestStore(t)
	uploadHybridCorpus(t, s, 5)

	opts := index.SearchOptions{TopK: 10, BM25Weight: 0.5, Fusion: fuse.NewRRFFusion(60)}
	results, err := s.SearchHybrid(context.Background(), keywordOnlyQuery, opts)
	require.NoError(t, err)

	assert.NotEmpty(t, hybridTargetID(results, "quantum fluoroscope"),
		"custom fusion must include keyword-only chunks")
}

func TestMemoryStore_SearchHybrid_DeletedChunkNotReturned(t *testing.T) {
	s := newTestStore(t)
	uploadHybridCorpus(t, s, 3)

	// The target is a strong BM25 hit; delete it and verify hybrid search
	// no longer resurrects it through the keyword side.
	require.NoError(t, s.DeleteChunk(context.Background(), "doc-target::chunk-0"))

	results, err := s.SearchHybrid(context.Background(), keywordOnlyQuery, index.SearchOptions{TopK: 10, BM25Weight: 1.0})
	require.NoError(t, err)

	assert.Empty(t, hybridTargetID(results, "quantum fluoroscope"),
		"deleted chunk must not be returned by hybrid search")
}

// --- SQLiteStore.SearchHybrid end-to-end ---

func uploadHybridCorpusSQLite(t *testing.T, s *SQLiteStore, distractors int) {
	t.Helper()
	ctx := context.Background()
	require.NoError(t, s.Upload(ctx, core.NewDocument("doc-target", "Target", "t.md"), keywordOnlyTargetContent))
	for i := 0; i < distractors; i++ {
		content := fmt.Sprintf("distractor number %d about mundane everyday topics and common ordinary phrases", i)
		require.NoError(t, s.Upload(ctx, core.NewDocument(fmt.Sprintf("doc-dist%d", i), "Dist", "d.md"), content))
	}
}

func TestSQLiteStore_SearchHybrid_PureBM25WeightRanksKeywordMatchFirst(t *testing.T) {
	s := newTestSQLiteStore(t)
	uploadHybridCorpusSQLite(t, s, 5)

	opts := index.SearchOptions{TopK: 10, BM25Weight: 1.0}
	results, err := s.SearchHybrid(context.Background(), keywordOnlyQuery, opts)
	require.NoError(t, err)
	require.NotEmpty(t, results)

	require.Equal(t, "doc-target::chunk-0", hybridTargetID(results, "quantum fluoroscope"),
		"BM25Weight=1 must rank the keyword-only match first (pure BM25)")
}

func TestSQLiteStore_SearchHybrid_PureVectorWeightMatchesVectorSearch(t *testing.T) {
	s := newTestSQLiteStore(t)
	uploadHybridCorpusSQLite(t, s, 5)

	ctx := context.Background()
	vecResults, err := s.Search(ctx, keywordOnlyQuery, index.SearchOptions{TopK: 10})
	require.NoError(t, err)
	require.NotEmpty(t, vecResults)

	opts := index.SearchOptions{TopK: 10, BM25Weight: 0.0}
	results, err := s.SearchHybrid(ctx, keywordOnlyQuery, opts)
	require.NoError(t, err)
	require.Len(t, results, len(vecResults))

	assert.Equal(t, vecResults[0].Chunk.ID, results[0].Chunk.ID,
		"BM25Weight=0 must behave as pure vector search")
}

func TestSQLiteStore_SearchHybrid_RRFIncludesFTSOnlyMatch(t *testing.T) {
	s := newTestSQLiteStore(t)
	uploadHybridCorpusSQLite(t, s, 5)

	opts := index.SearchOptions{TopK: 10, Fusion: fuse.NewRRFFusion(60)}
	results, err := s.SearchHybrid(context.Background(), keywordOnlyQuery, opts)
	require.NoError(t, err)

	assert.NotEmpty(t, hybridTargetID(results, "quantum fluoroscope"),
		"RRF fusion must include FTS keyword-only matches")
}

// --- Bug 6: single keyword index per namespace, pruned on delete ---
//
// MemoryStore used to keep a second, store-level BM25 instance per
// namespace that DeleteChunk/DeleteDocument never pruned, so deleted
// chunks kept scoring in hybrid search forever. The fix removes the
// duplicate: each index's internal BM25 is the single keyword source
// and is pruned by MemoryIndex.Delete.

func TestMemoryStore_DeleteDocumentPrunesKeywordIndex(t *testing.T) {
	s := newTestStore(t)
	uploadHybridCorpus(t, s, 3)

	idx := s.indexes["test"]
	require.NotEmpty(t, idx.SearchBM25(keywordOnlyQuery),
		"keyword index must contain the target before deletion")

	require.NoError(t, s.DeleteDocument(context.Background(), "doc-target"))
	assert.Empty(t, idx.SearchBM25(keywordOnlyQuery),
		"DeleteDocument must prune the keyword index")

	// And hybrid search must not resurrect the deleted document.
	results, err := s.SearchHybrid(context.Background(), keywordOnlyQuery,
		index.SearchOptions{TopK: 10, BM25Weight: 1.0})
	require.NoError(t, err)
	assert.Empty(t, hybridTargetID(results, "quantum fluoroscope"),
		"deleted document must not match keyword search")
}

func TestMemoryStore_DeleteChunkPrunesKeywordIndex(t *testing.T) {
	s := newTestStore(t)
	uploadHybridCorpus(t, s, 2)

	idx := s.indexes["test"]
	require.NotEmpty(t, idx.SearchBM25(keywordOnlyQuery))

	require.NoError(t, s.DeleteChunk(context.Background(), "doc-target::chunk-0"))
	assert.Empty(t, idx.SearchBM25(keywordOnlyQuery),
		"DeleteChunk must prune the keyword index")
}
