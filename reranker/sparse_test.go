package reranker

import (
	"context"
	"testing"

	"github.com/deagy/recall/index"
)

func TestSparseReranker_PromotesKeywordMatches(t *testing.T) {
	rr := NewSparseReranker()
	results := []index.SearchResult{
		// Dense score ranks the generic passage first, but it has no keyword
		// overlap with the query.
		mkResult("generic", "the quick fox jumps over the lazy dog", 0.99),
		// This passage contains the query's rare term, so BM25 should win.
		mkResult("match", "retrieval augmented generation improves answers", 0.40),
	}
	out, err := rr.Rerank(context.Background(), "retrieval augmented generation", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 results, got %d", len(out))
	}
	if out[0].Chunk.ID != "match" {
		t.Errorf("top result = %s, want match", out[0].Chunk.ID)
	}
	if out[0].RerankRank != 1 || out[1].RerankRank != 2 {
		t.Errorf("ranks = %d,%d, want 1,2", out[0].RerankRank, out[1].RerankRank)
	}
	if out[0].Reranker != "sparse-bm25" {
		t.Errorf("reranker = %q", out[0].Reranker)
	}
	// The generic chunk has zero keyword overlap; with KeepUnscored it keeps
	// the coarse score as its RerankScore.
	if out[1].RerankScore != 0.99 {
		t.Errorf("unscored RerankScore = %f, want coarse 0.99", out[1].RerankScore)
	}
}

func TestSparseReranker_DropsUnscored(t *testing.T) {
	rr := &SparseReranker{KeepUnscored: false}
	results := []index.SearchResult{
		mkResult("generic", "the quick fox jumps over the lazy dog", 0.99),
		mkResult("match", "retrieval augmented generation improves answers", 0.40),
	}
	out, err := rr.Rerank(context.Background(), "retrieval augmented", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 result, got %d", len(out))
	}
	if out[0].Chunk.ID != "match" {
		t.Errorf("top = %s, want match", out[0].Chunk.ID)
	}
}

func TestSparseReranker_Empty(t *testing.T) {
	rr := NewSparseReranker()
	out, err := rr.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty, got %d", len(out))
	}
}

func TestSparseReranker_PreservesAll(t *testing.T) {
	rr := NewSparseReranker()
	in := []index.SearchResult{
		mkResult("a", "alpha beta", 0.9),
		mkResult("b", "gamma delta", 0.8),
		mkResult("c", "epsilon zeta", 0.7),
	}
	out, err := rr.Rerank(context.Background(), "alpha gamma epsilon", in)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("expected all 3 preserved, got %d", len(out))
	}
}
