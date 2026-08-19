package analytics

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQueryLog_RecordAndCount(t *testing.T) {
	l := NewQueryLog(10)
	l.LogQuery("what is recall?", 10*time.Millisecond, 5, 0.9, nil)
	l.LogQuery("what is recall?", 20*time.Millisecond, 3, 0.8, nil)
	l.LogQuery("hnsw index", 5*time.Millisecond, 2, 0.7, nil)
	assert.Equal(t, 3, l.Count())

	recs := l.Records()
	require.Len(t, recs, 3)
	assert.Equal(t, "what is recall?", recs[0].Query)
	assert.NotEmpty(t, recs[0].ID)
	assert.False(t, recs[0].Time.IsZero())
}

func TestQueryLog_BoundedRingBuffer(t *testing.T) {
	l := NewQueryLog(3)
	for i := 0; i < 5; i++ {
		l.LogQuery("q", time.Millisecond, 1, 0.5, nil)
	}
	assert.Equal(t, 3, l.Count())
	recs := l.Records()
	require.Len(t, recs, 3)
	assert.NotEmpty(t, recs[0].ID)
	// IDs of retained records are unique.
	seen := make(map[string]bool)
	for _, r := range recs {
		assert.False(t, seen[r.ID], "expected unique record IDs")
		seen[r.ID] = true
	}
}

func TestQueryLog_Since(t *testing.T) {
	l := NewQueryLog(10)
	l.LogQuery("old", time.Millisecond, 1, 0.5, nil)
	mid := time.Now().UTC().Add(time.Second)
	l.Record(QueryRecord{Query: "new", Time: mid})
	got := l.Since(mid)
	require.Len(t, got, 1)
	assert.Equal(t, "new", got[0].Query)
}

func TestQueryLog_Reset(t *testing.T) {
	l := NewQueryLog(10)
	l.LogQuery("a", time.Millisecond, 1, 0.5, nil)
	l.Reset()
	assert.Equal(t, 0, l.Count())
	assert.Empty(t, l.Records())
}

func TestPopularQueries(t *testing.T) {
	l := NewQueryLog(100)
	for i := 0; i < 5; i++ {
		l.LogQuery("What Is Recall?", time.Millisecond, 3, 0.9, nil)
	}
	for i := 0; i < 2; i++ {
		l.LogQuery("hnsw", time.Millisecond, 1, 0.5, nil)
	}
	l.LogQuery("sqlite", time.Millisecond, 2, 0.6, nil)

	pop := l.PopularQueries(0)
	require.Len(t, pop, 3)
	assert.Equal(t, "what is recall?", pop[0].Query)
	assert.Equal(t, 5, pop[0].Count)
	assert.Equal(t, 2, pop[1].Count)
	assert.Equal(t, 1, pop[2].Count)

	top := l.PopularQueries(1)
	require.Len(t, top, 1)
	assert.Equal(t, "what is recall?", top[0].Query)
}

func TestDropOff(t *testing.T) {
	l := NewQueryLog(100)
	l.LogQuery("obscure term", time.Millisecond, 0, 0, nil)
	l.LogQuery("obscure term", time.Millisecond, 0, 0, nil)
	l.LogQuery("weak match", time.Millisecond, 1, 0.1, nil)
	l.LogQuery("good query", time.Millisecond, 4, 0.9, nil)
	l.LogQuery("boom", time.Millisecond, 0, 0, errors.New("boom"))

	drops := l.DropOff(0.5, 0)
	require.Len(t, drops, 3)
	assert.Equal(t, "obscure term", drops[0].Query)
	assert.Equal(t, 2, drops[0].Count)
}

type captureSink struct {
	write func(QueryRecord) error
}

func (c *captureSink) Write(r QueryRecord) error { return c.write(r) }
func (c *captureSink) Close() error              { return nil }

func TestQueryLog_SinkReceivesRecords(t *testing.T) {
	var got []QueryRecord
	s := &captureSink{write: func(r QueryRecord) error { got = append(got, r); return nil }}
	l := NewQueryLog(10).WithSink(s)
	l.LogQuery("q", time.Millisecond, 1, 0.5, nil)
	require.Len(t, got, 1)
	assert.Equal(t, "q", got[0].Query)
}

func TestQueryLog_SinkErrorCaptured(t *testing.T) {
	wantErr := errors.New("sink down")
	s := &captureSink{write: func(r QueryRecord) error { return wantErr }}
	l := NewQueryLog(10).WithSink(s)
	l.LogQuery("q", time.Millisecond, 1, 0.5, nil)
	require.Error(t, l.LastError())
	assert.Equal(t, "sink down", l.LastError().Error())
}
