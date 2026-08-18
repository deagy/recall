package reranker

import (
	"context"
	"fmt"

	"github.com/deagy/recall/bm25"
	"github.com/deagy/recall/index"
)

// SparseReranker is a BM25-based reranker that re-scores coarse results by
// keyword relevance. It builds a BM25 index over just the candidate chunks
// and uses BM25 scores as the fine-rank score, which tends to promote
// exact-term matches that dense vectors can under-rank.
type SparseReranker struct {
	// Config are the BM25 saturation / length parameters.
	Config bm25.Config

	// KeepUnscored leaves chunks that scored zero (no keyword overlap) at
	// the end of the list, ranked by their coarse score. When false, only
	// chunks with a positive BM25 score are kept.
	KeepUnscored bool
}

// NewSparseReranker creates a SparseReranker with default BM25 parameters.
func NewSparseReranker() *SparseReranker {
	return &SparseReranker{Config: bm25.DefaultConfig(), KeepUnscored: true}
}

// Name implements Reranker.
func (r *SparseReranker) Name() string { return "sparse-bm25" }

// Rerank re-scores the given results with BM25 and returns them ordered by
// the resulting score. Results are never dropped when KeepUnscored is true;
// otherwise chunks with zero keyword overlap are filtered out.
func (r *SparseReranker) Rerank(_ context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	idx := bm25.New(r.Config)
	for _, res := range results {
		if res.Chunk == nil {
			continue
		}
		idx.AddDocument(res.Chunk.ID, res.Chunk.Content)
	}

	bm25Scores := map[string]float64{}
	if idx.Count() > 0 {
		for _, sr := range idx.Search(query) {
			bm25Scores[sr.DocID] = sr.Score
		}
	}

	out := make([]index.SearchResult, 0, len(results))
	for _, res := range results {
		if res.Chunk == nil {
			return nil, fmt.Errorf("reranker: sparse: result without chunk")
		}
		score := bm25Scores[res.Chunk.ID]
		if score <= 0 && !r.KeepUnscored {
			continue
		}
		// Retain the coarse score as a tie-breaker in RerankScore when there
		// is no keyword signal.
		if score <= 0 {
			score = res.Score
		}
		r2 := res
		r2.RerankScore = score
		out = append(out, r2)
	}

	return finalize(r.Name(), out), nil
}
