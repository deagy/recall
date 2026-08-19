package analytics

import (
	"context"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// InstrumentedAnalyticsStore wraps a store.Store and records a QueryRecord for
// each search. It is a drop-in replacement for store.Store.
type InstrumentedAnalyticsStore struct {
	inner store.Store
	log   *QueryLog
}

// NewInstrumentedAnalyticsStore wraps s, recording query analytics to log. If
// log is nil, no analytics are recorded (the wrapper still delegates).
func NewInstrumentedAnalyticsStore(s store.Store, log *QueryLog) *InstrumentedAnalyticsStore {
	return &InstrumentedAnalyticsStore{inner: s, log: log}
}

// Inner exposes the wrapped store.
func (w *InstrumentedAnalyticsStore) Inner() store.Store { return w.inner }

// Log returns the QueryLog being recorded to (may be nil).
func (w *InstrumentedAnalyticsStore) Log() *QueryLog { return w.log }

// Upload implements store.Store.
func (w *InstrumentedAnalyticsStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	return w.inner.Upload(ctx, doc, content)
}

// Search implements store.Store and records a query record.
func (w *InstrumentedAnalyticsStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	start := time.Now()
	results, err := w.inner.Search(ctx, query, opts)
	w.record("vector", query, opts, results, time.Since(start), err)
	return results, err
}

// SearchHybrid implements store.Store and records a query record.
func (w *InstrumentedAnalyticsStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	start := time.Now()
	results, err := w.inner.SearchHybrid(ctx, query, opts)
	w.record("hybrid", query, opts, results, time.Since(start), err)
	return results, err
}

func (w *InstrumentedAnalyticsStore) record(qtype, query string, opts index.SearchOptions, results []index.SearchResult, d time.Duration, err error) {
	if w.log == nil {
		return
	}
	rec := QueryRecord{
		Query:    query,
		Latency:  d,
		Results:  len(results),
		Metadata: map[string]string{"query.type": qtype},
	}
	if len(results) > 0 {
		top := results[0].Score
		for _, r := range results[1:] {
			if r.Score > top {
				top = r.Score
			}
		}
		rec.TopScore = top
	}
	if err != nil {
		rec.Error = err.Error()
	}
	if opts.Filters != nil {
		rec.Metadata["filtered"] = "true"
	}
	w.log.Record(rec)
}

// GetChunk implements store.Store.
func (w *InstrumentedAnalyticsStore) GetChunk(id string) (*core.Chunk, bool) {
	return w.inner.GetChunk(id)
}

// DeleteChunk implements store.Store.
func (w *InstrumentedAnalyticsStore) DeleteChunk(ctx context.Context, id string) error {
	return w.inner.DeleteChunk(ctx, id)
}

// DeleteDocument implements store.Store.
func (w *InstrumentedAnalyticsStore) DeleteDocument(ctx context.Context, docID string) error {
	return w.inner.DeleteDocument(ctx, docID)
}

// Count implements store.Store.
func (w *InstrumentedAnalyticsStore) Count() int { return w.inner.Count() }

// Namespaces implements store.Store.
func (w *InstrumentedAnalyticsStore) Namespaces() []string { return w.inner.Namespaces() }

// Close implements store.Store.
func (w *InstrumentedAnalyticsStore) Close() error { return w.inner.Close() }
