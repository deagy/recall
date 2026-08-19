package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// Compile-time assertion that the wrapper is a drop-in store.Store.
var _ store.Store = (*InstrumentedStore)(nil)

// fakeStore is a minimal store.Store for exercising the wrapper.
type fakeStore struct {
	uploadErr error
	searchErr error
	searchRes []index.SearchResult
	count     int
}

func (f *fakeStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	time.Sleep(time.Millisecond)
	return f.uploadErr
}

func (f *fakeStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	time.Sleep(time.Millisecond)
	return f.searchRes, f.searchErr
}

func (f *fakeStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return f.searchRes, f.searchErr
}

func (f *fakeStore) GetChunk(id string) (*core.Chunk, bool)                 { return nil, false }
func (f *fakeStore) DeleteChunk(ctx context.Context, id string) error       { return nil }
func (f *fakeStore) DeleteDocument(ctx context.Context, docID string) error { return nil }
func (f *fakeStore) Count() int                                             { return f.count }
func (f *fakeStore) Namespaces() []string                                   { return []string{"default"} }
func (f *fakeStore) Close() error                                           { return nil }

func TestInstrumentedStore_RecordsMetrics(t *testing.T) {
	r := NewRegistry()
	inner := &fakeStore{count: 10, searchRes: []index.SearchResult{{Score: 0.9}}}
	w := NewInstrumentedStore(inner, r)

	if _, err := w.Search(context.Background(), "hello", index.SearchOptions{}); err != nil {
		t.Fatalf("unexpected search error: %v", err)
	}
	if _, err := w.SearchHybrid(context.Background(), "hello", index.SearchOptions{}); err != nil {
		t.Fatalf("unexpected hybrid error: %v", err)
	}
	if err := w.Upload(context.Background(), &core.Document{ID: "doc1"}, "content"); err != nil {
		t.Fatalf("unexpected upload error: %v", err)
	}

	m := w.Metrics()
	if m == nil {
		t.Fatal("expected metrics to be set")
	}
	if m.ErrorRate() != 0 {
		t.Fatalf("expected 0 error rate, got %v", m.ErrorRate())
	}
	if m.Size() != 10 {
		t.Fatalf("expected store size 10, got %v", m.Size())
	}
	if m.SearchLatencyP50() <= 0 {
		t.Fatalf("expected positive search latency, got %v", m.SearchLatencyP50())
	}
}

func TestInstrumentedStore_RecordsErrors(t *testing.T) {
	r := NewRegistry()
	inner := &fakeStore{searchErr: errors.New("boom")}
	w := NewInstrumentedStore(inner, r)

	for i := 0; i < 2; i++ {
		if _, err := w.Search(context.Background(), "q", index.SearchOptions{}); err == nil {
			t.Fatal("expected search to return an error")
		}
	}
	if got := w.Metrics().ErrorRate(); got != 1 {
		t.Fatalf("expected error rate 1, got %v", got)
	}
}

func TestInstrumentedStore_LoggingWithCorrelation(t *testing.T) {
	var buf bytes.Buffer
	r := NewRegistry()
	l := NewLogger(&buf, LevelDebug, true)
	inner := &fakeStore{searchRes: []index.SearchResult{{Score: 0.5}}}
	w := NewInstrumentedStore(inner, r).WithLogger(l)

	id := NewCorrelationID()
	ctx := WithCorrelationID(context.Background(), id)
	if _, err := w.Search(ctx, "hello", index.SearchOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var rec map[string]interface{}
	if err := json.Unmarshal([]byte(buf.String()), &rec); err != nil {
		t.Fatalf("expected valid JSON log: %v\n%s", err, buf.String())
	}
	if rec["correlation_id"] != id {
		t.Fatalf("expected correlation_id %q, got %v", id, rec["correlation_id"])
	}
	if rec["msg"] != "search" {
		t.Fatalf("expected msg=search, got %v", rec["msg"])
	}
	if rec["query"] != "hello" {
		t.Fatalf("expected query=hello, got %v", rec["query"])
	}
}

func TestInstrumentedStore_Delegates(t *testing.T) {
	inner := &fakeStore{count: 7}
	w := NewInstrumentedStore(inner, nil) // no registry -> no metrics

	if w.Metrics() != nil {
		t.Fatal("expected nil metrics when constructed with a nil registry")
	}
	if w.Count() != 7 {
		t.Fatalf("expected count 7, got %d", w.Count())
	}
	if w.Inner() != store.Store(inner) {
		t.Fatal("expected Inner to return the wrapped store")
	}
	if got := w.Namespaces(); len(got) != 1 || got[0] != "default" {
		t.Fatalf("unexpected namespaces: %v", got)
	}
	if _, ok := w.GetChunk("x"); ok {
		t.Fatal("expected GetChunk to report not-found")
	}
	if err := w.DeleteChunk(context.Background(), "x"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := w.DeleteDocument(context.Background(), "d"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
