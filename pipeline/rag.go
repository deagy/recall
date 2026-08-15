// RAGPipeline orchestrates the RAG workflow: retrieve relevant chunks,
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
}

// RAGPipeline is the main RAG pipeline that retrieves and assembles context.
type RAGPipeline struct {
	store            store.Store
	template         *Template
	topK             int
	minScore         float64
	maxContextTokens int
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

// Query performs a RAG query: retrieves relevant chunks and assembles a prompt.
func (p *RAGPipeline) Query(ctx context.Context, question string) (*RAGResponse, error) {
	// Retrieve relevant chunks
	opts := index.DefaultSearchOptions(p.topK)
	opts.MinScore = p.minScore

	results, err := p.store.Search(ctx, question, opts)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Assemble context
	cw := NewContextWindow(p.maxContextTokens)
	for _, r := range results {
		cw.AddChunk(*r.Chunk)
	}

	// Build response
	resp := &RAGResponse{
		Context: cw.String(),
		Sources: results,
		Tokens:  cw.Tokens(),
	}

	// Render prompt
	vars := map[string]interface{}{
		"Context":  resp.Context,
		"Question": question,
	}
	resp.Answer = p.template.Render(vars)

	return resp, nil
}

// QueryHybrid performs a RAG query using hybrid search (vector + BM25).
func (p *RAGPipeline) QueryHybrid(ctx context.Context, question string) (*RAGResponse, error) {
	opts := index.DefaultSearchOptions(p.topK)
	opts.MinScore = p.minScore
	opts.Hybrid = true
	opts.BM25Weight = 0.5

	results, err := p.store.SearchHybrid(ctx, question, opts)
	if err != nil {
		return nil, fmt.Errorf("hybrid search failed: %w", err)
	}

	// Assemble context
	cw := NewContextWindow(p.maxContextTokens)
	for _, r := range results {
		cw.AddChunk(*r.Chunk)
	}

	// Build response
	resp := &RAGResponse{
		Context: cw.String(),
		Sources: results,
		Tokens:  cw.Tokens(),
	}

	vars := map[string]interface{}{
		"Context":  resp.Context,
		"Question": question,
	}
	resp.Answer = p.template.Render(vars)

	return resp, nil
}
