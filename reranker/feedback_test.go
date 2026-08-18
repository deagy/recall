package reranker

import (
	"context"
	"sync"
	"testing"

	"github.com/deagy/recall/index"
)

func TestAdaptiveLTRanker_RefitsAtThreshold(t *testing.T) {
	// True signal: keyword overlap, coarse score misleading.
	base := []LTRExample{
		{Query: "retrieval", Result: mkResult("g1", "a retrieval system for documents", 0.4), Label: 1},
		{Query: "retrieval", Result: mkResult("g2", "retrieval augmented generation", 0.5), Label: 1},
		{Query: "retrieval", Result: mkResult("b1", "the weather outside is sunny", 0.9), Label: 0},
	}
	rr := NewAdaptiveLTRanker(AdaptiveConfig{
		LTR:            LTRConfig{Epochs: 300, LearningRate: 0.5},
		RefitThreshold: 2,
	})
	if rr.Fitted() {
		t.Fatal("should not be fitted before Fit")
	}
	if err := rr.Fit(context.Background(), base); err != nil {
		t.Fatalf("fit: %v", err)
	}
	if got := rr.ExamplesFitted(); got != 3 {
		t.Fatalf("ExamplesFitted = %d, want 3", got)
	}
	if got := rr.FeedbackRecorded(); got != 0 {
		t.Fatalf("FeedbackRecorded = %d, want 0", got)
	}

	// One example below the threshold: no refit.
	n, err := rr.RecordFeedback(context.Background(), []LTRExample{
		{Query: "retrieval", Result: mkResult("b2", "cooking pasta with garlic", 0.8), Label: 0},
	})
	if err != nil {
		t.Fatalf("record: %v", err)
	}
	if n != 3 {
		t.Errorf("fitted n after 1 feedback = %d, want 3 (no refit yet)", n)
	}
	if got := rr.ExamplesFitted(); got != 3 {
		t.Errorf("ExamplesFitted = %d, want 3 (buffer not full)", got)
	}

	// Second example reaches the threshold: automatic refit on 5 examples.
	n, err = rr.RecordFeedback(context.Background(), []LTRExample{
		{Query: "retrieval", Result: mkResult("b3", "a sunny day at the beach", 0.7), Label: 0},
	})
	if err != nil {
		t.Fatalf("record: %v", err)
	}
	if n != 5 {
		t.Errorf("fitted n after threshold = %d, want 5 (refit on base+feedback)", n)
	}
	if got := rr.ExamplesFitted(); got != 5 {
		t.Errorf("ExamplesFitted = %d, want 5", got)
	}
	if got := rr.FeedbackRecorded(); got != 2 {
		t.Errorf("FeedbackRecorded = %d, want 2", got)
	}

	// The learned model must prefer the keyword-matching chunk over a
	// high-scoring irrelevant one.
	out, err := rr.Rerank(context.Background(), "retrieval", []index.SearchResult{
		mkResult("irrel", "the weather outside is sunny", 0.95),
		mkResult("rel", "a retrieval system for documents", 0.5),
	})
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if out[0].Chunk.ID != "rel" {
		t.Errorf("top = %s, want rel", out[0].Chunk.ID)
	}
}

func TestAdaptiveLTRanker_RefitNow(t *testing.T) {
	rr := NewAdaptiveLTRanker(AdaptiveConfig{
		LTR:            LTRConfig{Epochs: 200},
		RefitThreshold: 100, // high, so nothing refits automatically.
	})
	base := []LTRExample{
		{Query: "q", Result: mkResult("a", "alpha topic text", 0.4), Label: 1},
		{Query: "q", Result: mkResult("b", "unrelated beta text", 0.9), Label: 0},
	}
	if err := rr.Fit(context.Background(), base); err != nil {
		t.Fatalf("fit: %v", err)
	}
	if _, err := rr.RecordFeedback(context.Background(), []LTRExample{
		{Query: "q", Result: mkResult("c", "more alpha topic text", 0.5), Label: 1},
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	n, err := rr.RefitNow(context.Background())
	if err != nil {
		t.Fatalf("refit now: %v", err)
	}
	if n != 3 {
		t.Errorf("RefitNow n = %d, want 3", n)
	}
	// Buffer drained: another single example should not refit.
	n, err = rr.RecordFeedback(context.Background(), []LTRExample{
		{Query: "q", Result: mkResult("d", "unrelated delta text", 0.8), Label: 0},
	})
	if err != nil {
		t.Fatalf("record: %v", err)
	}
	if n != 3 {
		t.Errorf("n after drain = %d, want 3 (no refit)", n)
	}
}

func TestAdaptiveLTRanker_WithoutFitIsIdentity(t *testing.T) {
	rr := NewAdaptiveLTRanker(AdaptiveConfig{RefitThreshold: 1})
	if _, err := rr.RecordFeedback(context.Background(), []LTRExample{
		{Query: "q", Result: mkResult("a", "alpha", 0.9), Label: 1},
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	// With no initial Fit, the refit trains on the single example; the
	// reranker must still return a valid ordering for other inputs.
	out, err := rr.Rerank(context.Background(), "q", []index.SearchResult{
		mkResult("x", "x", 0.3),
		mkResult("y", "y", 0.9),
	})
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 2 || out[0].RerankRank != 1 || out[1].RerankRank != 2 {
		t.Errorf("invalid ranking: %+v", out)
	}
	if rr.Name() != "adaptive-ltr" {
		t.Errorf("name = %q, want adaptive-ltr", rr.Name())
	}
}

func TestAdaptiveLTRanker_Errors(t *testing.T) {
	rr := NewAdaptiveLTRanker(AdaptiveConfig{})
	if err := rr.Fit(context.Background(), nil); err == nil {
		t.Error("expected error for empty Fit")
	}
	if _, err := rr.RecordFeedback(context.Background(), nil); err == nil {
		t.Error("expected error for empty RecordFeedback")
	}
}

func TestAdaptiveLTRanker_Concurrent(t *testing.T) {
	rr := NewAdaptiveLTRanker(AdaptiveConfig{
		LTR:            LTRConfig{Epochs: 2},
		RefitThreshold: 10,
	})
	_ = rr.Fit(context.Background(), []LTRExample{
		{Query: "q", Result: mkResult("a", "alpha text", 0.5), Label: 1},
		{Query: "q", Result: mkResult("b", "beta text", 0.4), Label: 0},
	})
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx := context.Background()
			for i := 0; i < 25; i++ {
				_, _ = rr.RecordFeedback(ctx, []LTRExample{
					{Query: "q", Result: mkResult("a", "alpha text", 0.6), Label: 1},
				})
				_, _ = rr.Rerank(ctx, "q", []index.SearchResult{mkResult("a", "alpha text", 0.6)})
			}
		}()
	}
	wg.Wait()
	if got := rr.FeedbackRecorded(); got != 100 {
		t.Errorf("FeedbackRecorded = %d, want 100", got)
	}
}
