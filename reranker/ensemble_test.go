package reranker

import (
	"context"
	"sort"
	"testing"

	"github.com/deagy/recall/fuse"
	"github.com/deagy/recall/index"
)

// fakeReranker scores each result with a fixed content-based function, giving
// deterministic control over per-reranker ordering in ensemble tests.
type fakeReranker struct {
	name  string
	score func(content string) float64
}

func (f fakeReranker) Name() string { return f.name }

func (f fakeReranker) Rerank(_ context.Context, _ string, results []index.SearchResult) ([]index.SearchResult, error) {
	out := make([]index.SearchResult, len(results))
	for i, r := range results {
		out[i] = r
		out[i].RerankScore = f.score(r.Chunk.Content)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].RerankScore > out[j].RerankScore })
	return out, nil
}

func TestEnsembleReranker_RRF(t *testing.T) {
	// Two rerankers rank "a" first by different signals; "c" is best on both.
	alpha := fakeReranker{name: "alpha", score: func(c string) float64 {
		if c == "a" {
			return 1
		}
		if c == "c" {
			return 0.9
		}
		return 0.1
	}}
	beta := fakeReranker{name: "beta", score: func(c string) float64 {
		if c == "c" {
			return 1
		}
		if c == "b" {
			return 0.8
		}
		return 0.1
	}}
	ens, err := NewEnsembleReranker(alpha, beta)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	results := []index.SearchResult{
		mkResult("a", "a", 0.9),
		mkResult("b", "b", 0.8),
		mkResult("c", "c", 0.7),
	}
	out, err := ens.Rerank(context.Background(), "q", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	// "c" is top on both -> should be overall top.
	if out[0].Chunk.ID != "c" {
		t.Errorf("top = %s, want c", out[0].Chunk.ID)
	}
	if out[0].Reranker != "ensemble" {
		t.Errorf("reranker name = %q, want ensemble", out[0].Reranker)
	}
	if out[0].RerankRank != 1 {
		t.Errorf("top rank = %d, want 1", out[0].RerankRank)
	}
}

func TestEnsembleReranker_WeightedMean(t *testing.T) {
	alpha := fakeReranker{name: "alpha", score: func(c string) float64 {
		if c == "a" {
			return 1
		}
		return 0
	}}
	beta := fakeReranker{name: "beta", score: func(c string) float64 {
		if c == "b" {
			return 1
		}
		return 0
	}}
	ens, _ := NewEnsembleReranker(alpha, beta)
	ens.Weights = []float64{1, 1} // equal weights
	results := []index.SearchResult{
		mkResult("a", "a", 0.9),
		mkResult("b", "b", 0.8),
	}
	out, err := ens.Rerank(context.Background(), "q", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	// a: alpha norm 1, beta norm 0 -> 0.5 ; b: 0, 1 -> 0.5. Tie broken by
	// coarse score: a (0.9) > b (0.8).
	if out[0].Chunk.ID != "a" {
		t.Errorf("top = %s, want a (coarse tie-break)", out[0].Chunk.ID)
	}
}

func TestEnsembleReranker_CustomFusion(t *testing.T) {
	alpha := fakeReranker{name: "alpha", score: func(c string) float64 {
		if c == "a" {
			return 1
		}
		return 0.1
	}}
	beta := fakeReranker{name: "beta", score: func(c string) float64 {
		if c == "b" {
			return 1
		}
		return 0.1
	}}
	ens, _ := NewEnsembleReranker(alpha, beta)
	// Weighted 90% alpha -> "a" should win.
	ens.Fusion = fuse.NewWeightedFusion(0.9)
	results := []index.SearchResult{
		mkResult("a", "a", 0.5),
		mkResult("b", "b", 0.5),
	}
	out, err := ens.Rerank(context.Background(), "q", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if out[0].Chunk.ID != "a" {
		t.Errorf("top = %s, want a under 0.9 alpha weight", out[0].Chunk.ID)
	}
}

func TestEnsembleReranker_Single(t *testing.T) {
	single := fakeReranker{name: "solo", score: func(c string) float64 {
		if c == "z" {
			return 1
		}
		return 0.2
	}}
	ens, err := NewEnsembleReranker(single)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if ens.Name() != "solo" {
		t.Errorf("single name = %q, want solo", ens.Name())
	}
	results := []index.SearchResult{
		mkResult("x", "x", 0.9),
		mkResult("z", "z", 0.3),
	}
	out, _ := ens.Rerank(context.Background(), "q", results)
	if out[0].Chunk.ID != "z" {
		t.Errorf("top = %s, want z", out[0].Chunk.ID)
	}
}

func TestEnsembleReranker_RequiresOne(t *testing.T) {
	if _, err := NewEnsembleReranker(); err == nil {
		t.Error("expected error for empty ensemble")
	}
}
