package tracing

import (
	"context"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// InstrumentedTracingStore wraps a store.Store and creates trace spans for its
// operations. Upload, Search, and SearchHybrid each create a span (child of any
// span already in the context), and a successful search additionally records a
// child "retrieve" span so the upload → search → retrieve path is visible.
// Standard attributes (document ID, namespace, query, top-k, result count,
// status, duration) are set on each span.
//
// Spans are reported to the process-wide default tracer; call SetDefaultTracer
// to route them to a processor (e.g. an InMemoryProcessor).
type InstrumentedTracingStore struct {
	inner store.Store
}

// NewInstrumentedTracingStore wraps s for tracing.
func NewInstrumentedTracingStore(s store.Store) *InstrumentedTracingStore {
	return &InstrumentedTracingStore{inner: s}
}

// Inner exposes the wrapped store.
func (w *InstrumentedTracingStore) Inner() store.Store { return w.inner }

// Upload implements store.Store, tracing the upload pipeline.
func (w *InstrumentedTracingStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	ctx, span := StartSpan(ctx, "store.upload", WithKind(SpanKindInternal))
	defer span.End()
	span.SetAttribute("store", "recall")
	if doc != nil {
		span.SetAttribute("document.id", doc.ID)
		if doc.Namespace != "" {
			span.SetAttribute("namespace", doc.Namespace)
		}
	}
	span.SetAttribute("content.bytes", len(content))

	err := w.inner.Upload(ctx, doc, content)
	if err != nil {
		span.SetStatus(StatusError, err.Error())
		span.AddEvent("upload.error", map[string]interface{}{"error": err.Error()})
	} else {
		span.SetStatus(StatusOK, "")
		span.AddEvent("upload.completed", map[string]interface{}{"store.count": w.inner.Count()})
	}
	span.SetAttribute("duration", span.Duration().String())
	return err
}

// Search implements store.Store, tracing the vector search.
func (w *InstrumentedTracingStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	ctx, span := StartSpan(ctx, "store.search",
		WithKind(SpanKindServer),
		WithAttributes(map[string]interface{}{
			"query":      query,
			"query.type": "vector",
		}),
	)
	defer span.End()
	span.SetAttribute("top_k", opts.TopK)

	results, err := w.inner.Search(ctx, query, opts)
	span.SetAttribute("results", len(results))
	if err != nil {
		span.SetStatus(StatusError, err.Error())
	} else {
		span.SetStatus(StatusOK, "")
		w.recordRetrieve(ctx, len(results))
	}
	span.SetAttribute("duration", span.Duration().String())
	return results, err
}

// SearchHybrid implements store.Store, tracing the hybrid search.
func (w *InstrumentedTracingStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	ctx, span := StartSpan(ctx, "store.search",
		WithKind(SpanKindServer),
		WithAttributes(map[string]interface{}{
			"query":       query,
			"query.type":  "hybrid",
			"bm25.weight": opts.BM25Weight,
		}),
	)
	defer span.End()
	span.SetAttribute("top_k", opts.TopK)

	results, err := w.inner.SearchHybrid(ctx, query, opts)
	span.SetAttribute("results", len(results))
	if err != nil {
		span.SetStatus(StatusError, err.Error())
	} else {
		span.SetStatus(StatusOK, "")
		w.recordRetrieve(ctx, len(results))
	}
	span.SetAttribute("duration", span.Duration().String())
	return results, err
}

// recordRetrieve adds a child span for the result-retrieval step. The parent
// span is carried in ctx.
func (w *InstrumentedTracingStore) recordRetrieve(ctx context.Context, n int) {
	_, retrieve := StartSpan(ctx, "store.retrieve")
	retrieve.SetAttribute("results", n)
	retrieve.SetStatus(StatusOK, "")
	retrieve.End()
}

// GetChunk implements store.Store, tracing the point-retrieval step.
func (w *InstrumentedTracingStore) GetChunk(id string) (*core.Chunk, bool) {
	_, span := StartSpan(context.Background(), "store.retrieve")
	chunk, ok := w.inner.GetChunk(id)
	if ok {
		span.SetStatus(StatusOK, "")
	} else {
		span.SetStatus(StatusError, "not found")
	}
	span.SetAttribute("chunk.id", id)
	span.SetAttribute("found", ok)
	span.End()
	return chunk, ok
}

// DeleteChunk implements store.Store.
func (w *InstrumentedTracingStore) DeleteChunk(ctx context.Context, id string) error {
	_, span := StartSpan(ctx, "store.delete_chunk")
	err := w.inner.DeleteChunk(ctx, id)
	span.SetAttribute("chunk.id", id)
	if err != nil {
		span.SetStatus(StatusError, err.Error())
	} else {
		span.SetStatus(StatusOK, "")
	}
	span.End()
	return err
}

// DeleteDocument implements store.Store.
func (w *InstrumentedTracingStore) DeleteDocument(ctx context.Context, docID string) error {
	_, span := StartSpan(ctx, "store.delete_document")
	err := w.inner.DeleteDocument(ctx, docID)
	span.SetAttribute("document.id", docID)
	if err != nil {
		span.SetStatus(StatusError, err.Error())
	} else {
		span.SetStatus(StatusOK, "")
	}
	span.End()
	return err
}

// Count implements store.Store.
func (w *InstrumentedTracingStore) Count() int { return w.inner.Count() }

// Namespaces implements store.Store.
func (w *InstrumentedTracingStore) Namespaces() []string { return w.inner.Namespaces() }

// Close implements store.Store.
func (w *InstrumentedTracingStore) Close() error { return w.inner.Close() }
