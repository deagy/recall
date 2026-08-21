package store

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// parityCorpus is uploaded identically to both stores. Each document is
// short enough to chunk into exactly one chunk, so chunk IDs are
// predictable (<docID>::chunk-0) and chunk content equals the document body.
var parityCorpus = map[string]string{
	"doc-go":   "Go programming language for building scalable network services.",
	"doc-py":   "Python scripting language for data analysis and automation.",
	"doc-rust": "Rust systems programming language for safe concurrent code.",
	"doc-db":   "SQLite embedded database engine for persistent storage.",
}

const parityQuery = "programming language for systems"

// refSims computes the true cosine similarity between the query embedding
// and each chunk embedding (known single-chunk corpus).
func refSims(t *testing.T, emb embedder.Embedder) map[string]float64 {
	t.Helper()
	ctx := context.Background()
	queryVec, err := emb.Embed(ctx, parityQuery)
	require.NoError(t, err)
	sims := make(map[string]float64, len(parityCorpus))
	for docID, content := range parityCorpus {
		vec, err := emb.Embed(ctx, content)
		require.NoError(t, err)
		sims[docID+"::chunk-0"] = embedder.CosineSimilarity(queryVec, vec)
	}
	return sims
}

func resultIDs(results []index.SearchResult) map[string]bool {
	set := make(map[string]bool, len(results))
	for _, r := range results {
		set[r.Chunk.ID] = true
	}
	return set
}

// TestMinScoreParity_Vector is the table-driven Memory vs SQLite parity test
// for vector search: for every MinScore value both stores must return
// exactly the chunks whose true similarity passes the threshold.
func TestMinScoreParity_Vector(t *testing.T) {
	ctx := context.Background()
	emb := embedder.NewMockEmbedder(384)

	mem, err := NewMemoryStore(Config{Namespace: "test", Embedder: emb})
	require.NoError(t, err)
	defer mem.Close()

	sq, err := NewSQLiteStore(Config{Namespace: "test", Embedder: emb}, ":memory:")
	require.NoError(t, err)
	defer sq.Close()

	for id, content := range parityCorpus {
		doc := core.NewDocument(id, "Test", "test.txt")
		require.NoError(t, mem.Upload(ctx, doc, content))
		require.NoError(t, sq.Upload(ctx, doc, content))
	}

	type vectorSearcher interface {
		Search(context.Context, string, index.SearchOptions) ([]index.SearchResult, error)
	}
	stores := map[string]vectorSearcher{"memory": mem, "sqlite": sq}

	sims := refSims(t, emb)
	for _, tc := range []struct {
		name     string
		minScore float64
	}{
		{"no threshold", 0},
		{"mid threshold", 0.75},
		{"high threshold", 0.9},
		{"impossible", 1.0001},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := index.DefaultSearchOptions(10)
			opts.MinScore = tc.minScore
			want := make(map[string]bool)
			for id, sim := range sims {
				if sim >= tc.minScore {
					want[id] = true
				}
			}

			for name, s := range stores {
				results, err := s.Search(ctx, parityQuery, opts)
				require.NoError(t, err, "%s: search", name)
				assert.Equal(t, want, resultIDs(results),
					"%s: MinScore %v must keep exactly the passing chunks", name, tc.minScore)
				for _, r := range results {
					assert.GreaterOrEqual(t, r.Score, tc.minScore,
						"%s: returned a score below MinScore", name)
				}
			}
		})
	}
}

// TestMinScoreParity_Hybrid verifies that hybrid search in both stores drops
// every result whose *fused* score is below MinScore. The exact surviving
// set is derived from each store's own MinScore=0 baseline, because the two
// stores score keywords with different engines (bm25 package vs FTS5) and
// therefore cannot be expected to produce identical fused scores.
func TestMinScoreParity_Hybrid(t *testing.T) {
	ctx := context.Background()
	emb := embedder.NewMockEmbedder(384)

	mem, err := NewMemoryStore(Config{Namespace: "test", Embedder: emb})
	require.NoError(t, err)
	defer mem.Close()

	sq, err := NewSQLiteStore(Config{Namespace: "test", Embedder: emb}, ":memory:")
	require.NoError(t, err)
	defer sq.Close()

	for id, content := range parityCorpus {
		doc := core.NewDocument(id, "Test", "test.txt")
		require.NoError(t, mem.Upload(ctx, doc, content))
		require.NoError(t, sq.Upload(ctx, doc, content))
	}

	type hybridSearcher interface {
		SearchHybrid(context.Context, string, index.SearchOptions) ([]index.SearchResult, error)
	}
	stores := map[string]hybridSearcher{"memory": mem, "sqlite": sq}

	baseOpts := index.DefaultSearchOptions(10)
	baseOpts.Hybrid = true
	baseOpts.BM25Weight = 0.5

	for name, s := range stores {
		t.Run(name, func(t *testing.T) {
			base, err := s.SearchHybrid(ctx, parityQuery, baseOpts)
			require.NoError(t, err)
			require.NotEmpty(t, base, "baseline hybrid search must return results")

			baseByID := make(map[string]float64, len(base))
			var minScore, maxScore float64
			for i, r := range base {
				baseByID[r.Chunk.ID] = r.Score
				if i == 0 || r.Score < minScore {
					minScore = r.Score
				}
				if i == 0 || r.Score > maxScore {
					maxScore = r.Score
				}
			}
			if maxScore <= minScore {
				t.Skip("degenerate single-score baseline; no separating threshold derivable")
			}
			threshold := (minScore + maxScore) / 2

			opts := index.DefaultSearchOptions(10)
			opts.Hybrid = true
			opts.BM25Weight = 0.5
			opts.MinScore = threshold

			filtered, err := s.SearchHybrid(ctx, parityQuery, opts)
			require.NoError(t, err)

			want := make(map[string]bool)
			for id, score := range baseByID {
				if score >= threshold {
					want[id] = true
				}
			}
			assert.Equal(t, want, resultIDs(filtered),
				"results with fused score below MinScore must be dropped")
			for _, r := range filtered {
				assert.GreaterOrEqual(t, r.Score, threshold,
					"returned a fused score below MinScore")
			}
			assert.NotEmpty(t, filtered, "at least the top result must survive the midpoint threshold")
		})
	}
}
