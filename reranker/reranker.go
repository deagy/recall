// Package reranker implements fine-ranking (reranking) of coarse retrieval
// results to improve top-k precision. All rerankers are pure Go with no CGO
// or external model downloads: the cross-encoder runs on the bundled ONNX
// runtime, the sparse reranker reuses the BM25 package, the LLM reranker
// consumes an injected llm.Backend, and the ensemble fuses several
// rerankers via the fuse package.
package reranker

import (
	"context"
	"sort"

	"github.com/deagy/recall/index"
)

// Reranker is the interface satisfied by all fine-ranking strategies.
//
// Rerank takes a query and an already-retrieved (coarse) list of results and
// returns a new, reordered list. Implementations MUST:
//   - preserve the same set of chunks (no additions, no drops),
//   - set each result's RerankScore, RerankRank, and Reranker fields,
//   - return results sorted by RerankScore descending (RerankRank 1 = best).
//
// The original Score (coarse retrieval score) is left untouched so callers
// can compare coarse vs fine ranking.
type Reranker interface {
	// Rerank reorders and scores the given results for the query.
	Rerank(ctx context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error)

	// Name returns a stable identifier for this reranker, recorded in
	// SearchResult.Reranker for score attribution.
	Name() string
}

// finalize sorts results by RerankScore descending, assigns 1-based
// RerankRank, and stamps the reranker name on every result. It returns a new
// slice; the input slice is not mutated in place.
func finalize(name string, results []index.SearchResult) []index.SearchResult {
	out := make([]index.SearchResult, len(results))
	copy(out, results)

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].RerankScore != out[j].RerankScore {
			return out[i].RerankScore > out[j].RerankScore
		}
		// Tie-break by coarse score, then ID, for determinism.
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Chunk != nil && out[j].Chunk != nil && out[i].Chunk.ID < out[j].Chunk.ID
	})

	for i := range out {
		out[i].RerankRank = i + 1
		out[i].Reranker = name
	}
	return out
}

// normalizeToUnit maps a slice of scores into [0,1] using min-max scaling.
// When all scores are equal (or the slice is short) every score maps to 1.
func normalizeToUnit(scores []float64) []float64 {
	n := make([]float64, len(scores))
	if len(scores) == 0 {
		return n
	}
	min, max := scores[0], scores[0]
	for _, s := range scores[1:] {
		if s < min {
			min = s
		}
		if s > max {
			max = s
		}
	}
	span := max - min
	for i, s := range scores {
		if span == 0 {
			n[i] = 1
		} else {
			n[i] = (s - min) / span
		}
	}
	return n
}

// clampScore clamps v into [0, max].
func clampScore(v, max float64) float64 {
	if v < 0 {
		return 0
	}
	if v > max {
		return max
	}
	return v
}
