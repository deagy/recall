package pipeline

import (
	"context"

	"github.com/deagy/recall/index"
)

// Reranker is the fine-ranking interface the RAG pipeline uses to reorder
// coarse retrieval results. It is deliberately redeclared here (a structural
// type) so the pipeline package does not import the concrete reranker
// package. Any value implementing reranker.Reranker satisfies this interface
// without any adapter.
type Reranker interface {
	// Rerank reorders and scores the given results for the query.
	Rerank(ctx context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error)

	// Name identifies the reranker for score attribution.
	Name() string
}

// WithReranker configures a fine-ranking stage. When set, Query and
// QueryHybrid run a two-stage retrieval: a coarse pass retrieves candidates,
// then the reranker refines their order. Returns the pipeline for chaining.
func (p *RAGPipeline) WithReranker(r Reranker) *RAGPipeline {
	if r != nil {
		p.reranker = r
	}
	return p
}

// WithCoarseTopK sets how many candidates the coarse retrieval pass pulls
// before reranking. It only takes effect when a reranker is configured and
// k > 0; otherwise topK is used. A larger coarseTopK gives the reranker more
// candidates to promote.
func (p *RAGPipeline) WithCoarseTopK(k int) *RAGPipeline {
	if k > 0 {
		p.coarseTopK = k
	}
	return p
}

// WithRerankTopK sets how many results survive the reranking stage. When > 0
// and a reranker is configured, the reranked list is truncated to this size.
func (p *RAGPipeline) WithRerankTopK(k int) *RAGPipeline {
	if k > 0 {
		p.rerankTopK = k
	}
	return p
}
