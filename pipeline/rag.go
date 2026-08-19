// Package pipeline implements the RAG workflow: retrieve relevant chunks,
// assemble context, and render a prompt for LLM consumption.
package pipeline

import (
	"context"
	"fmt"

	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// RAGResponse contains the result of a RAG query.
type RAGResponse struct {
	// Answer is the rendered prompt ready for LLM consumption.
	Answer string

	// Context is the assembled context string from retrieved chunks.
	Context string

	// Sources are the retrieved chunks with their relevance scores.
	Sources []index.SearchResult

	// Tokens is the approximate token count of the context.
	Tokens int

	// Citations are the ranked references to source chunks. Populated only
	// when the pipeline is configured with WithCitations.
	Citations []Citation
}

// RAGPipeline is the main RAG pipeline that retrieves and assembles context.
type RAGPipeline struct {
	store            store.Store
	template         *Template
	topK             int
	minScore         float64
	maxContextTokens int
	reranker         Reranker
	coarseTopK       int
	rerankTopK       int
	citations        bool
	smartContext     bool
	filters          []index.Filter
}

// NewRAGPipeline creates a new RAG pipeline with the given store and template.
func NewRAGPipeline(s store.Store, t *Template) *RAGPipeline {
	if t == nil {
		t = DefaultTemplate()
	}
	return &RAGPipeline{
		store:            s,
		template:         t,
		topK:             10,
		minScore:         0.0,
		maxContextTokens: 4096,
	}
}

// WithTopK sets the number of chunks to retrieve.
func (p *RAGPipeline) WithTopK(k int) *RAGPipeline {
	if k > 0 {
		p.topK = k
	}
	return p
}

// WithMinScore sets the minimum relevance score for retrieved chunks.
func (p *RAGPipeline) WithMinScore(score float64) *RAGPipeline {
	if score >= 0 {
		p.minScore = score
	}
	return p
}

// WithMaxTokens sets the maximum token limit for the context window.
func (p *RAGPipeline) WithMaxTokens(tokens int) *RAGPipeline {
	if tokens > 0 {
		p.maxContextTokens = tokens
	}
	return p
}

// WithCitations enables citation tracking: the response's Citations field is
// populated with ranked references to the source chunks.
func (p *RAGPipeline) WithCitations() *RAGPipeline {
	p.citations = true
	return p
}

// WithSmartContext enables priority-based context selection: chunks are
// included by relevance score within the token budget (via SmartContextWindow)
// rather than strictly in retrieval order.
func (p *RAGPipeline) WithSmartContext() *RAGPipeline {
	p.smartContext = true
	return p
}

// WithSearchFilters appends metadata filters applied during retrieval: a
// chunk must match every filter to be retrieved (see index.Filter). Use
// Clone() first when configuring a shared pipeline for a single request, so
// the shared instance is not mutated.
func (p *RAGPipeline) WithSearchFilters(filters ...index.Filter) *RAGPipeline {
	p.filters = append(append([]index.Filter(nil), p.filters...), filters...)
	return p
}

// Clone returns a copy of the pipeline with an independent search-filter
// slice. Use it to derive request-specific pipelines (e.g. with additional
// filters) from a shared instance without data races.
func (p *RAGPipeline) Clone() *RAGPipeline {
	cp := *p
	if p.filters != nil {
		cp.filters = append([]index.Filter(nil), p.filters...)
	}
	return &cp
}

// Query performs a RAG query: retrieves relevant chunks and assembles a prompt.
// When a reranker is configured, retrieval is two-stage: a coarse vector pass
// pulls coarseTopK (or topK) candidates, then the reranker refines them.
func (p *RAGPipeline) Query(ctx context.Context, question string) (*RAGResponse, error) {
	results, err := p.retrieve(ctx, question, false)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	return p.buildResponse(question, results), nil
}

// QueryHybrid performs a RAG query using hybrid search (vector + BM25), with
// the same optional two-stage reranking as Query.
func (p *RAGPipeline) QueryHybrid(ctx context.Context, question string) (*RAGResponse, error) {
	results, err := p.retrieve(ctx, question, true)
	if err != nil {
		return nil, fmt.Errorf("hybrid search failed: %w", err)
	}
	return p.buildResponse(question, results), nil
}

// retrieve performs the coarse search pass and, when a reranker is set, the
// fine reranking pass. It honors coarseTopK during retrieval and rerankTopK
// after reranking.
func (p *RAGPipeline) retrieve(ctx context.Context, question string, hybrid bool) ([]index.SearchResult, error) {
	coarseK := p.topK
	if p.reranker != nil && p.coarseTopK > 0 {
		coarseK = p.coarseTopK
	}

	opts := index.DefaultSearchOptions(coarseK)
	opts.MinScore = p.minScore
	opts.Filters = append(opts.Filters, p.filters...)
	if hybrid {
		opts.Hybrid = true
		opts.BM25Weight = 0.5
	}

	var results []index.SearchResult
	var err error
	if hybrid {
		results, err = p.store.SearchHybrid(ctx, question, opts)
	} else {
		results, err = p.store.Search(ctx, question, opts)
	}
	if err != nil {
		return nil, err
	}

	if p.reranker != nil {
		results, err = p.reranker.Rerank(ctx, question, results)
		if err != nil {
			return nil, fmt.Errorf("rerank failed: %w", err)
		}
		if p.rerankTopK > 0 && len(results) > p.rerankTopK {
			results = results[:p.rerankTopK]
		}
	}
	return results, nil
}

// buildResponse assembles the context window and renders the prompt from a
// final, ordered result list.
func (p *RAGPipeline) buildResponse(question string, results []index.SearchResult) *RAGResponse {
	cw := NewContextWindow(p.maxContextTokens)
	if p.smartContext {
		candidates := make([]ScoredChunk, 0, len(results))
		for _, r := range results {
			if r.Chunk == nil {
				continue
			}
			candidates = append(candidates, ScoredChunk{Chunk: *r.Chunk, Score: r.Score})
		}
		for _, chunk := range NewSmartContextWindow(p.maxContextTokens).Select(candidates) {
			cw.AddChunk(chunk)
		}
	} else {
		for _, r := range results {
			cw.AddChunk(*r.Chunk)
		}
	}

	resp := &RAGResponse{
		Context: cw.String(),
		Sources: results,
		Tokens:  cw.Tokens(),
	}

	if p.citations {
		resp.Citations = TrackCitations(results)
	}

	resp.Answer = p.template.Render(map[string]interface{}{
		"Context":  resp.Context,
		"Question": question,
	})
	return resp
}
