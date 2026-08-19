package metrics

import (
	"context"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// InstrumentedStore wraps a store.Store and records StoreMetrics around
// upload and search operations, optionally emitting structured logs. It is a
// drop-in replacement for store.Store: pass it anywhere a store is expected.
//
// It is intentionally additive and non-invasive: the wrapped store is
// delegated to unchanged, and metrics are only recorded when a Registry (and
// optionally a Logger) are provided.
type InstrumentedStore struct {
	inner  store.Store
	m      *StoreMetrics
	logger *Logger
}

// NewInstrumentedStore wraps s, recording store metrics from reg. If reg is
// nil, no metrics are recorded (the wrapper still delegates).
func NewInstrumentedStore(s store.Store, reg *Registry) *InstrumentedStore {
	var m *StoreMetrics
	if reg != nil {
		m = NewStoreMetrics(reg)
	}
	return &InstrumentedStore{inner: s, m: m}
}

// WithLogger sets the structured logger used to record upload/search events.
// The correlation ID carried in the request context is included in each log
// record. It returns the wrapper for chaining.
func (w *InstrumentedStore) WithLogger(l *Logger) *InstrumentedStore {
	w.logger = l
	return w
}

// Inner exposes the wrapped store.
func (w *InstrumentedStore) Inner() store.Store { return w.inner }

// Metrics returns the StoreMetrics being recorded (nil if constructed with a
// nil registry).
func (w *InstrumentedStore) Metrics() *StoreMetrics { return w.m }

// Upload implements store.Store.
func (w *InstrumentedStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	start := time.Now()
	err := w.inner.Upload(ctx, doc, content)
	d := time.Since(start)
	if w.m != nil {
		if err != nil {
			w.m.RecordUploadError()
		} else {
			w.m.RecordUpload(d)
			w.m.SetSize(w.inner.Count())
		}
	}
	if w.logger != nil {
		w.logger.Ctx(ctx, levelFromErr(err), "upload",
			String("document", documentID(doc)),
			Duration("duration", d),
			Error("error", err),
		)
	}
	return err
}

// Search implements store.Store.
func (w *InstrumentedStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	start := time.Now()
	results, err := w.inner.Search(ctx, query, opts)
	d := time.Since(start)
	if w.m != nil {
		if err != nil {
			w.m.RecordSearchError()
		} else {
			w.m.RecordSearch(d)
		}
	}
	if w.logger != nil {
		w.logger.Ctx(ctx, levelFromErr(err), "search",
			String("query", query),
			Int("results", len(results)),
			Duration("duration", d),
			Error("error", err),
		)
	}
	return results, err
}

// SearchHybrid implements store.Store.
func (w *InstrumentedStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	start := time.Now()
	results, err := w.inner.SearchHybrid(ctx, query, opts)
	d := time.Since(start)
	if w.m != nil {
		if err != nil {
			w.m.RecordSearchError()
		} else {
			w.m.RecordSearch(d)
		}
	}
	if w.logger != nil {
		w.logger.Ctx(ctx, levelFromErr(err), "hybrid_search",
			String("query", query),
			Int("results", len(results)),
			Duration("duration", d),
			Error("error", err),
		)
	}
	return results, err
}

// GetChunk implements store.Store.
func (w *InstrumentedStore) GetChunk(id string) (*core.Chunk, bool) {
	return w.inner.GetChunk(id)
}

// DeleteChunk implements store.Store.
func (w *InstrumentedStore) DeleteChunk(ctx context.Context, id string) error {
	return w.inner.DeleteChunk(ctx, id)
}

// DeleteDocument implements store.Store.
func (w *InstrumentedStore) DeleteDocument(ctx context.Context, docID string) error {
	return w.inner.DeleteDocument(ctx, docID)
}

// Count implements store.Store and refreshes the store-size gauge.
func (w *InstrumentedStore) Count() int {
	n := w.inner.Count()
	if w.m != nil {
		w.m.SetSize(n)
	}
	return n
}

// Namespaces implements store.Store.
func (w *InstrumentedStore) Namespaces() []string {
	return w.inner.Namespaces()
}

// Close implements store.Store.
func (w *InstrumentedStore) Close() error {
	return w.inner.Close()
}

// levelFromErr maps an error to a log level.
func levelFromErr(err error) Level {
	if err != nil {
		return LevelError
	}
	return LevelInfo
}

// documentID returns the document's ID, or "" for a nil document.
func documentID(doc *core.Document) string {
	if doc == nil {
		return ""
	}
	return doc.ID
}
