// Package ingest orchestrates the full document ingestion pipeline:
// load (via a loader or connector) -> filter (incremental + dedup) ->
// validate -> transform -> upload (chunk, embed, index via the store).
//
// The pipeline is safe for concurrent use across goroutines when the
// underlying store and embedder are, and reports progress through
// thread-safe callbacks.
package ingest

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// Progress tracks ingestion counters and fires callbacks. All methods are
// safe for concurrent use.
type Progress struct {
	mu       sync.Mutex
	Phase    string
	start    time.Time
	loaded   int
	skipped  int
	uploaded int
	failed   int

	// OnDocument is called after each document with its id and outcome
	// ("uploaded", "skipped", or "failed").
	OnDocument func(id, outcome string)

	// OnPhase is called when the pipeline phase changes.
	OnPhase func(phase string)
}

// NewProgress creates a Progress ready to use.
func NewProgress() *Progress {
	return &Progress{start: time.Now()}
}

// SetPhase records a phase transition and fires OnPhase.
func (p *Progress) SetPhase(phase string) {
	p.mu.Lock()
	old := p.Phase
	p.Phase = phase
	cb := p.OnPhase
	p.mu.Unlock()
	if cb != nil && phase != old {
		cb(phase)
	}
}

// Loaded records that n documents were loaded from the source.
func (p *Progress) Loaded(n int) {
	p.mu.Lock()
	p.loaded += n
	p.mu.Unlock()
}

// Skip records one skipped document.
func (p *Progress) Skip(id string) {
	p.mu.Lock()
	p.skipped++
	cb := p.OnDocument
	p.mu.Unlock()
	if cb != nil {
		cb(id, "skipped")
	}
}

// Upload records one uploaded document.
func (p *Progress) Upload(id string) {
	p.mu.Lock()
	p.uploaded++
	cb := p.OnDocument
	p.mu.Unlock()
	if cb != nil {
		cb(id, "uploaded")
	}
}

// Fail records one failed document.
func (p *Progress) Fail(id string) {
	p.mu.Lock()
	p.failed++
	cb := p.OnDocument
	p.mu.Unlock()
	if cb != nil {
		cb(id, "failed")
	}
}

// Summary returns a one-line human-readable progress summary.
func (p *Progress) Summary() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	phase := p.Phase
	if phase == "" {
		phase = "idle"
	}
	return fmt.Sprintf("%s: %d loaded, %d uploaded, %d skipped, %d failed (%s)",
		phase, p.loaded, p.uploaded, p.skipped, p.failed, time.Since(p.start).Round(time.Millisecond))
}

// Counters returns a snapshot of the counters.
func (p *Progress) Counters() (loaded, skipped, uploaded, failed int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.loaded, p.skipped, p.uploaded, p.failed
}

// Result is the outcome of one pipeline run.
type Result struct {
	// Source is the ref that was ingested.
	Source string
	// Loaded is the number of documents the source produced.
	Loaded int
	// Uploaded is the number stored in the knowledge store.
	Uploaded int
	// Skipped is the number filtered out (incremental/dedup/validation).
	Skipped int
	// Failed documents with their errors.
	Failed []Failure
	// Duration of the run.
	Duration time.Duration
}

// Failure pairs a document id with its error.
type Failure struct {
	ID  string
	Err error
}

// Error renders the failures of a result as a single error (or nil).
func (r *Result) Error() error {
	if len(r.Failed) == 0 {
		return nil
	}
	msgs := make([]string, 0, len(r.Failed))
	for _, f := range r.Failed {
		msgs = append(msgs, f.ID+": "+f.Err.Error())
	}
	return fmt.Errorf("ingest: %d document(s) failed: %s", len(r.Failed), strings.Join(msgs, "; "))
}
