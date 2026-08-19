package analytics

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Compile-time assertion that the wrapper is a drop-in store.Store.
var _ store.Store = (*InstrumentedAnalyticsStore)(nil)

type analyticsFakeStore struct {
	searchRes []index.SearchResult
}

func (f *analyticsFakeStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	return nil
}
func (f *analyticsFakeStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return f.searchRes, nil
}
func (f *analyticsFakeStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return f.searchRes, nil
}
func (f *analyticsFakeStore) GetChunk(id string) (*core.Chunk, bool)                 { return nil, false }
func (f *analyticsFakeStore) DeleteChunk(ctx context.Context, id string) error       { return nil }
func (f *analyticsFakeStore) DeleteDocument(ctx context.Context, docID string) error { return nil }
func (f *analyticsFakeStore) Count() int                                             { return 0 }
func (f *analyticsFakeStore) Namespaces() []string                                   { return nil }
func (f *analyticsFakeStore) Close() error                                           { return nil }

type errStore struct{}

func (e *errStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	return errors.New("nope")
}
func (e *errStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return nil, errors.New("nope")
}
func (e *errStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return nil, errors.New("nope")
}
func (e *errStore) GetChunk(id string) (*core.Chunk, bool)                 { return nil, false }
func (e *errStore) DeleteChunk(ctx context.Context, id string) error       { return nil }
func (e *errStore) DeleteDocument(ctx context.Context, docID string) error { return nil }
func (e *errStore) Count() int                                             { return 0 }
func (e *errStore) Namespaces() []string                                   { return nil }
func (e *errStore) Close() error                                           { return nil }

func TestInstrumentedAnalyticsStore_RecordsSearch(t *testing.T) {
	log := NewQueryLog(100)
	inner := &analyticsFakeStore{searchRes: []index.SearchResult{{Score: 0.9}, {Score: 0.7}}}
	w := NewInstrumentedAnalyticsStore(inner, log)

	_, err := w.Search(context.Background(), "hello", index.SearchOptions{TopK: 2})
	require.NoError(t, err)

	recs := log.Records()
	require.Len(t, recs, 1)
	assert.Equal(t, "hello", recs[0].Query)
	assert.Equal(t, 2, recs[0].Results)
	assert.InDelta(t, 0.9, recs[0].TopScore, 1e-9)
	assert.Equal(t, "vector", recs[0].Metadata["query.type"])
	assert.Greater(t, recs[0].Latency, time.Duration(0))
}

func TestInstrumentedAnalyticsStore_RecordsHybrid(t *testing.T) {
	log := NewQueryLog(100)
	inner := &analyticsFakeStore{searchRes: []index.SearchResult{{Score: 0.6}}}
	w := NewInstrumentedAnalyticsStore(inner, log)

	_, err := w.SearchHybrid(context.Background(), "hybrid q", index.SearchOptions{TopK: 1, BM25Weight: 0.5})
	require.NoError(t, err)

	recs := log.Records()
	require.Len(t, recs, 1)
	assert.Equal(t, "hybrid", recs[0].Metadata["query.type"])
}

func TestInstrumentedAnalyticsStore_DropOffOnError(t *testing.T) {
	log := NewQueryLog(100)
	w := NewInstrumentedAnalyticsStore(&errStore{}, log)
	_, _ = w.Search(context.Background(), "boom", index.SearchOptions{})

	recs := log.Records()
	require.Len(t, recs, 1)
	assert.Equal(t, "boom", recs[0].Query)
	assert.NotEmpty(t, recs[0].Error)

	// The errored query is a drop-off.
	drops := log.DropOff(0.5, 0)
	require.Len(t, drops, 1)
	assert.Equal(t, "boom", drops[0].Query)
}

func TestInstrumentedAnalyticsStore_NilLog(t *testing.T) {
	w := NewInstrumentedAnalyticsStore(&analyticsFakeStore{}, nil)
	_, err := w.Search(context.Background(), "q", index.SearchOptions{})
	require.NoError(t, err)
	assert.Nil(t, w.Log())
}
