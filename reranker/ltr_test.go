package reranker

import (
	"context"
	"testing"

	"github.com/deagy/recall/index"
)

func TestLTRanker_UnfittedPreservesCoarse(t *testing.T) {
	rr := NewLTRanker(LTRConfig{})
	if rr.Fitted() {
		t.Fatal("should not be fitted yet")
	}
	results := []index.SearchResult{
		mkResult("low", "x", 0.3),
		mkResult("high", "y", 0.9),
	}
	out, err := rr.Rerank(context.Background(), "q", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if out[0].Chunk.ID != "high" {
		t.Errorf("top = %s, want high (coarse order)", out[0].Chunk.ID)
	}
	if out[0].RerankScore != 0.9 {
		t.Errorf("unfitted score = %f, want coarse 0.9", out[0].RerankScore)
	}
}

func TestLTRanker_LearnsToCorrectCoarse(t *testing.T) {
	// Training set: keyword overlap is the true relevance signal, while the
	// coarse score is deliberately misleading (irrelevant doc ranks higher).
	examples := []LTRExample{
		{Query: "retrieval", Result: mkResult("g1", "a retrieval system for documents", 0.4), Label: 1},
		{Query: "retrieval", Result: mkResult("g2", "retrieval augmented generation", 0.5), Label: 1},
		{Query: "retrieval", Result: mkResult("b1", "the weather outside is sunny", 0.9), Label: 0},
		{Query: "retrieval", Result: mkResult("b2", "cooking pasta with garlic", 0.8), Label: 0},
	}
	rr := NewLTRanker(LTRConfig{Epochs: 300, LearningRate: 0.5, L2: 0.001})
	if err := rr.Fit(context.Background(), examples); err != nil {
		t.Fatalf("fit: %v", err)
	}
	if !rr.Fitted() {
		t.Fatal("should be fitted after Fit")
	}

	// Candidate whose coarse score is high but which lacks the keyword.
	candidates := []index.SearchResult{
		mkResult("irrel", "the weather outside is sunny", 0.95),
		mkResult("rel", "a retrieval system for documents", 0.5),
	}
	out, err := rr.Rerank(context.Background(), "retrieval", candidates)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if out[0].Chunk.ID != "rel" {
		t.Errorf("top = %s, want rel (learned signal beats misleading coarse)", out[0].Chunk.ID)
	}
	if out[0].RerankScore <= out[1].RerankScore {
		t.Errorf("scores not ordered: %f !<= %f", out[0].RerankScore, out[1].RerankScore)
	}
}

func TestLTRanker_CustomFeatures(t *testing.T) {
	// Single-feature model: only the coarse score, so it should learn to
	// rank by it.
	rr := NewLTRanker(LTRConfig{
		Features: func(query string, res index.SearchResult) []float64 { return []float64{res.Score} },
		Epochs:   200,
	})
	examples := []LTRExample{
		{Query: "q", Result: mkResult("a", "a", 0.9), Label: 1},
		{Query: "q", Result: mkResult("b", "b", 0.1), Label: 0},
	}
	if err := rr.Fit(context.Background(), examples); err != nil {
		t.Fatalf("fit: %v", err)
	}
	out, _ := rr.Rerank(context.Background(), "q", []index.SearchResult{
		mkResult("b", "b", 0.1),
		mkResult("a", "a", 0.9),
	})
	if out[0].Chunk.ID != "a" {
		t.Errorf("top = %s, want a", out[0].Chunk.ID)
	}
}

func TestLTRanker_FitEmpty(t *testing.T) {
	rr := NewLTRanker(LTRConfig{})
	if err := rr.Fit(context.Background(), nil); err == nil {
		t.Error("expected error for empty training set")
	}
}

func TestDefaultFeatures(t *testing.T) {
	f := DefaultFeatures("retrieval generation", mkResult("x", "retrieval system for generation tasks", 0.7))
	if len(f) != 3 {
		t.Fatalf("expected 3 features, got %d", len(f))
	}
	if f[0] != 0.7 {
		t.Errorf("feature[0] = %f, want coarse 0.7", f[0])
	}
	if f[1] != 1.0 {
		t.Errorf("feature[1] = %f, want full overlap 1.0", f[1])
	}
}
