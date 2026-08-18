package reranker

import (
	"context"
	"fmt"
	"sync"

	"github.com/deagy/recall/index"
)

// AdaptiveLTRanker is a feedback-driven reranker: it wraps an LTRanker and
// adapts its weights from live relevance feedback (e.g. clicks, explicit
// thumbs up/down, or human judgments) instead of a static training set.
//
// Call Fit once with an initial labeled corpus, then feed fresh judgments
// through RecordFeedback. Each call appends the examples to an internal
// buffer; once the buffer reaches RefitThreshold the model is retrained on
// the full accumulated set. The reranker always returns a valid ranking, so
// it is safe to keep wired into a pipeline while it learns.
//
// All methods are safe for concurrent use.
type AdaptiveLTRanker struct {
	mu        sync.RWMutex
	ranker    *LTRanker
	threshold int
	initial   []LTRExample
	pending   []LTRExample
	fittedN   int
	feedbackN int
}

// AdaptiveConfig configures an AdaptiveLTRanker.
type AdaptiveConfig struct {
	// LTR configures the underlying LTRanker (features, learning rate,
	// epochs, L2). Zero values use the LTRanker defaults.
	LTR LTRConfig

	// RefitThreshold is the number of accumulated feedback examples that
	// triggers an automatic refit. Must be >= 1. Defaults to 100.
	RefitThreshold int
}

// NewAdaptiveLTRanker creates a feedback-driven LTRanker.
func NewAdaptiveLTRanker(cfg AdaptiveConfig) *AdaptiveLTRanker {
	threshold := cfg.RefitThreshold
	if threshold < 1 {
		threshold = 100
	}
	return &AdaptiveLTRanker{
		ranker:    NewLTRanker(cfg.LTR),
		threshold: threshold,
	}
}

// Name implements Reranker.
func (a *AdaptiveLTRanker) Name() string { return "adaptive-" + a.ranker.Name() }

// Fit trains the initial model on the given labeled examples. It can also be
// called again later to reset the accumulated corpus.
func (a *AdaptiveLTRanker) Fit(ctx context.Context, examples []LTRExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("reranker: adaptive: need at least one training example")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.ranker.Fit(ctx, examples); err != nil {
		return err
	}
	a.initial = append([]LTRExample(nil), examples...)
	a.fittedN = len(examples)
	return nil
}

// Fitted reports whether the underlying LTRanker has been trained.
func (a *AdaptiveLTRanker) Fitted() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.ranker.fitted
}

// RecordFeedback accepts new labeled examples observed since the last call.
// The examples are appended to the internal buffer and, once the buffer
// reaches the configured threshold, the model is retrained on the full
// accumulated set (initial Fit examples plus all recorded feedback). It
// returns the total number of examples the current model was trained on.
func (a *AdaptiveLTRanker) RecordFeedback(ctx context.Context, examples []LTRExample) (int, error) {
	if len(examples) == 0 {
		return 0, fmt.Errorf("reranker: adaptive: need at least one feedback example")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pending = append(a.pending, examples...)
	a.feedbackN += len(examples)
	if len(a.pending) >= a.threshold {
		return a.refitLocked(ctx)
	}
	return a.fittedN, nil
}

// RefitNow retrains the model immediately on the full accumulated example
// set (initial Fit examples plus all recorded feedback) and returns the
// total number of examples used.
func (a *AdaptiveLTRanker) RefitNow(ctx context.Context) (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.refitLocked(ctx)
}

// refitLocked retrains the model and drains the feedback buffer. The caller
// must hold a.mu exclusively.
func (a *AdaptiveLTRanker) refitLocked(ctx context.Context) (int, error) {
	train := make([]LTRExample, 0, len(a.initial)+len(a.pending))
	train = append(train, a.initial...)
	train = append(train, a.pending...)
	if err := a.ranker.Fit(ctx, train); err != nil {
		return a.fittedN, fmt.Errorf("reranker: adaptive: refit: %w", err)
	}
	a.fittedN = len(train)
	a.pending = a.pending[:0]
	return a.fittedN, nil
}

// ExamplesFitted returns the number of examples the current model was
// trained on.
func (a *AdaptiveLTRanker) ExamplesFitted() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.fittedN
}

// FeedbackRecorded returns the total number of feedback examples accepted
// through RecordFeedback since construction.
func (a *AdaptiveLTRanker) FeedbackRecorded() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.feedbackN
}

// Rerank implements Reranker, delegating to the current model.
func (a *AdaptiveLTRanker) Rerank(ctx context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.ranker.Rerank(ctx, query, results)
}
