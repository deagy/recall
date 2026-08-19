package tracing

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// Compile-time assertion that the wrapper is a drop-in store.Store.
var _ store.Store = (*InstrumentedTracingStore)(nil)

type traceFakeStore struct {
	searchRes []index.SearchResult
	count     int
}

func (f *traceFakeStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	return nil
}
func (f *traceFakeStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return f.searchRes, nil
}
func (f *traceFakeStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return f.searchRes, nil
}
func (f *traceFakeStore) GetChunk(id string) (*core.Chunk, bool)                 { return &core.Chunk{ID: id}, true }
func (f *traceFakeStore) DeleteChunk(ctx context.Context, id string) error       { return nil }
func (f *traceFakeStore) DeleteDocument(ctx context.Context, docID string) error { return nil }
func (f *traceFakeStore) Count() int                                             { return f.count }
func (f *traceFakeStore) Namespaces() []string                                   { return nil }
func (f *traceFakeStore) Close() error                                           { return nil }

// withTracer installs a fresh InMemoryProcessor-backed default tracer for the
// test and returns the processor. The previous default tracer is restored
// afterwards.
func withTracer(t *testing.T) *InMemoryProcessor {
	t.Helper()
	proc := NewInMemoryProcessor()
	old := DefaultTracer()
	t.Cleanup(func() { SetDefaultTracer(old) })
	SetDefaultTracer(NewTracer(proc))
	return proc
}

func findSpan(spans []*Span, name string) *Span {
	for _, s := range spans {
		if s.Name == name {
			return s
		}
	}
	return nil
}

func spanNames(spans []*Span) []string {
	out := make([]string, 0, len(spans))
	for _, s := range spans {
		out = append(out, s.Name)
	}
	return out
}

func TestInstrumentedTracingStore_SearchSpan(t *testing.T) {
	proc := withTracer(t)
	inner := &traceFakeStore{count: 5, searchRes: []index.SearchResult{{Score: 0.9}, {Score: 0.8}}}
	w := NewInstrumentedTracingStore(inner)

	if _, err := w.Search(context.Background(), "hello", index.SearchOptions{TopK: 2}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	spans := proc.Spans()
	searchSpan := findSpan(spans, "store.search")
	retrieveSpan := findSpan(spans, "store.retrieve")
	if searchSpan == nil {
		t.Fatalf("expected a store.search span, got %v", spanNames(spans))
	}
	if retrieveSpan == nil {
		t.Fatalf("expected a store.retrieve span, got %v", spanNames(spans))
	}
	if retrieveSpan.TraceID != searchSpan.TraceID {
		t.Fatal("expected retrieve to share the search trace")
	}
	if retrieveSpan.ParentID != searchSpan.SpanID {
		t.Fatal("expected retrieve to be a child of search")
	}
	if searchSpan.Attribute("query") != "hello" {
		t.Fatalf("expected query=hello, got %v", searchSpan.Attribute("query"))
	}
	if searchSpan.Attribute("query.type") != "vector" {
		t.Fatalf("expected query.type=vector, got %v", searchSpan.Attribute("query.type"))
	}
	if searchSpan.Attribute("top_k") != 2 {
		t.Fatalf("expected top_k=2, got %v", searchSpan.Attribute("top_k"))
	}
	if searchSpan.Attribute("results") != 2 {
		t.Fatalf("expected results=2, got %v", searchSpan.Attribute("results"))
	}
	if searchSpan.Status != StatusOK {
		t.Fatalf("expected OK status, got %v", searchSpan.Status)
	}
}

func TestInstrumentedTracingStore_UploadSpan(t *testing.T) {
	proc := withTracer(t)
	inner := &traceFakeStore{count: 3}
	w := NewInstrumentedTracingStore(inner)

	if err := w.Upload(context.Background(), &core.Document{ID: "doc1", Namespace: "ns"}, "content"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	uploadSpan := findSpan(proc.Spans(), "store.upload")
	if uploadSpan == nil {
		t.Fatal("expected a store.upload span")
	}
	if uploadSpan.Attribute("document.id") != "doc1" {
		t.Fatalf("expected document.id=doc1, got %v", uploadSpan.Attribute("document.id"))
	}
	if uploadSpan.Attribute("namespace") != "ns" {
		t.Fatalf("expected namespace=ns, got %v", uploadSpan.Attribute("namespace"))
	}
	if uploadSpan.Status != StatusOK {
		t.Fatalf("expected OK status, got %v", uploadSpan.Status)
	}
	if len(uploadSpan.Events()) == 0 {
		t.Fatal("expected an upload event")
	}
}

func TestInstrumentedTracingStore_GetChunkSpan(t *testing.T) {
	proc := withTracer(t)
	inner := &traceFakeStore{}
	w := NewInstrumentedTracingStore(inner)

	if _, ok := w.GetChunk("c1"); !ok {
		t.Fatal("expected the chunk to be found")
	}
	retrieveSpan := findSpan(proc.Spans(), "store.retrieve")
	if retrieveSpan == nil {
		t.Fatal("expected a store.retrieve span for GetChunk")
	}
	if retrieveSpan.Attribute("chunk.id") != "c1" {
		t.Fatalf("expected chunk.id=c1, got %v", retrieveSpan.Attribute("chunk.id"))
	}
	if retrieveSpan.Attribute("found") != true {
		t.Fatalf("expected found=true, got %v", retrieveSpan.Attribute("found"))
	}
}
