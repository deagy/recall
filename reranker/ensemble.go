package reranker

import (
	"context"
	"fmt"

	"github.com/deagy/recall/fuse"
	"github.com/deagy/recall/index"
)

// EnsembleReranker combines several rerankers into one by fusing their
// individual scores. This is useful when different rerankers are strong in
// different regimes (e.g. a cross-encoder for semantic fit plus a sparse
// reranker for exact-term matches).
type EnsembleReranker struct {
	// Rerankers is the set of sub-rerankers to run (at least one required).
	Rerankers []Reranker

	// Fusion, when set, combines the per-reranker score maps via the given
	// fusion strategy. When nil, Weights (if set) drive a weighted mean,
	// otherwise Reciprocal Rank Fusion is used as a robust default.
	Fusion fuse.Fusion

	// Weights optionally supplies one weight per sub-reranker for a weighted
	// mean of min-max-normalized scores. It is only used when Fusion is nil.
	Weights []float64

	// RRFK is the Reciprocal Rank Fusion constant used when no Fusion or
	// Weights is supplied. Defaults to 60.
	RRFK int
}

// NewEnsembleReranker creates an ensemble from the given rerankers.
func NewEnsembleReranker(rerankers ...Reranker) (*EnsembleReranker, error) {
	if len(rerankers) == 0 {
		return nil, fmt.Errorf("reranker: ensemble requires at least one reranker")
	}
	return &EnsembleReranker{Rerankers: rerankers, RRFK: 60}, nil
}

// Name implements Reranker.
func (r *EnsembleReranker) Name() string {
	if len(r.Rerankers) == 1 {
		return r.Rerankers[0].Name()
	}
	return "ensemble"
}

// Rerank runs every sub-reranker over the same candidate set, fuses their
// per-chunk scores, and returns the results ordered by the fused score.
func (r *EnsembleReranker) Rerank(ctx context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	// Run each sub-reranker and collect its per-chunk score map.
	maps := make([]map[string]float64, len(r.Rerankers))
	for i, rr := range r.Rerankers {
		scored, err := rr.Rerank(ctx, query, results)
		if err != nil {
			return nil, fmt.Errorf("reranker: ensemble: sub-reranker %d (%s): %w", i, rr.Name(), err)
		}
		m := make(map[string]float64, len(scored))
		for _, sr := range scored {
			if sr.Chunk == nil {
				continue
			}
			m[sr.Chunk.ID] = sr.RerankScore
		}
		maps[i] = m
	}

	fused := r.fuse(maps, results)

	out := make([]index.SearchResult, 0, len(results))
	for _, res := range results {
		if res.Chunk == nil {
			return nil, fmt.Errorf("reranker: ensemble: result without chunk")
		}
		r2 := res
		r2.RerankScore = fused[res.Chunk.ID]
		out = append(out, r2)
	}
	return finalize(r.Name(), out), nil
}

// fuse produces the final per-chunk score from the per-reranker maps.
func (r *EnsembleReranker) fuse(maps []map[string]float64, results []index.SearchResult) map[string]float64 {
	// Ensure every chunk ID is present in every map (filled with 0) so the
	// fusion sees a uniform universe and nothing is silently dropped.
	for _, m := range maps {
		for _, res := range results {
			if res.Chunk == nil {
				continue
			}
			if _, ok := m[res.Chunk.ID]; !ok {
				m[res.Chunk.ID] = 0
			}
		}
	}

	if r.Fusion != nil {
		return r.Fusion.Fuse(maps...)
	}
	if len(r.Weights) == len(maps) {
		return r.weightedMean(maps, results)
	}
	rrfK := r.RRFK
	if rrfK <= 0 {
		rrfK = 60
	}
	return fuse.NewRRFFusion(rrfK).Fuse(maps...)
}

// weightedMean returns a weighted sum of each reranker's min-max-normalized
// scores, using r.Weights as the weights.
func (r *EnsembleReranker) weightedMean(maps []map[string]float64, results []index.SearchResult) map[string]float64 {
	// Normalize each map independently into [0,1].
	normalized := make([]map[string]float64, len(maps))
	for i, m := range maps {
		keys := make([]string, 0, len(m))
		vals := make([]float64, 0, len(m))
		for k, v := range m {
			keys = append(keys, k)
			vals = append(vals, v)
		}
		nv := normalizeToUnit(vals)
		normalized[i] = make(map[string]float64, len(m))
		for j, k := range keys {
			normalized[i][k] = nv[j]
		}
	}

	out := make(map[string]float64, len(results))
	for _, res := range results {
		if res.Chunk == nil {
			continue
		}
		id := res.Chunk.ID
		var sum, wsum float64
		for i, m := range normalized {
			w := r.Weights[i]
			sum += w * m[id]
			wsum += w
		}
		if wsum > 0 {
			out[id] = sum / wsum
		}
	}
	return out
}
