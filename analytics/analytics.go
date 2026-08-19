// Package analytics provides query analytics: a bounded, thread-safe
// QueryLog that records each query's latency and results, plus helpers for
// trending (popular) queries and drop-off detection (queries that return no
// good results). Records can be exported to pluggable sinks (file, HTTP, or a
// custom message-queue sink).
//
// The package is stdlib-only and dependency-free.
package analytics

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// QueryRecord is a single logged query.
type QueryRecord struct {
	// ID uniquely identifies the record.
	ID string
	// Time is when the query was recorded.
	Time time.Time
	// Query is the raw query text.
	Query string
	// Latency is how long the query took.
	Latency time.Duration
	// Results is the number of results returned.
	Results int
	// TopScore is the best result score (0 when there were no results).
	TopScore float64
	// Error is the error message if the query failed (empty on success).
	Error string
	// Namespace optionally scopes the query.
	Namespace string
	// Metadata holds optional extra tags.
	Metadata map[string]string
}

// QueryLog is a bounded, thread-safe log of query records. When the log is
// full, the oldest records are overwritten (ring buffer).
type QueryLog struct {
	mu      sync.Mutex
	buf     []QueryRecord
	maxLen  int
	pos     int // next write position
	size    int // number of stored records
	sink    Sink
	lastErr error
}

// NewQueryLog returns a QueryLog retaining at most maxLen records. A non-positive
// maxLen defaults to 1024.
func NewQueryLog(maxLen int) *QueryLog {
	if maxLen <= 0 {
		maxLen = 1024
	}
	return &QueryLog{buf: make([]QueryRecord, maxLen), maxLen: maxLen}
}

// WithSink sets the sink that receives every recorded query. It returns the log
// for chaining.
func (l *QueryLog) WithSink(s Sink) *QueryLog {
	l.mu.Lock()
	l.sink = s
	l.mu.Unlock()
	return l
}

// Record stores a query record and forwards it to the configured sink. A zero
// Time or empty ID is filled in automatically. Sink failures are non-fatal; they
// are captured and retrievable via LastError.
func (l *QueryLog) Record(rec QueryRecord) {
	if rec.Time.IsZero() {
		rec.Time = time.Now().UTC()
	}
	if rec.ID == "" {
		rec.ID = newRecordID()
	}
	l.mu.Lock()
	l.buf[l.pos] = rec
	l.pos = (l.pos + 1) % l.maxLen
	if l.size < l.maxLen {
		l.size++
	}
	sink := l.sink
	l.mu.Unlock()

	if sink != nil {
		if err := sink.Write(rec); err != nil {
			l.mu.Lock()
			l.lastErr = err
			l.mu.Unlock()
		}
	}
}

// LogQuery is a convenience for recording a query with its outcome.
func (l *QueryLog) LogQuery(query string, latency time.Duration, results int, topScore float64, err error) {
	rec := QueryRecord{
		Query:    query,
		Latency:  latency,
		Results:  results,
		TopScore: topScore,
	}
	if err != nil {
		rec.Error = err.Error()
	}
	l.Record(rec)
}

// Records returns a copy of the stored records in oldest-to-newest order.
func (l *QueryLog) Records() []QueryRecord {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]QueryRecord, 0, l.size)
	start := (l.pos - l.size + l.maxLen) % l.maxLen
	for i := 0; i < l.size; i++ {
		out = append(out, l.buf[(start+i)%l.maxLen])
	}
	return out
}

// Since returns records recorded at or after t, in oldest-to-newest order.
func (l *QueryLog) Since(t time.Time) []QueryRecord {
	all := l.Records()
	out := make([]QueryRecord, 0, len(all))
	for _, r := range all {
		if !r.Time.Before(t) {
			out = append(out, r)
		}
	}
	return out
}

// Count returns the number of records currently retained.
func (l *QueryLog) Count() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.size
}

// Reset clears all records.
func (l *QueryLog) Reset() {
	l.mu.Lock()
	for i := range l.buf {
		l.buf[i] = QueryRecord{}
	}
	l.pos = 0
	l.size = 0
	l.mu.Unlock()
}

// LastError returns the most recent sink write error, if any.
func (l *QueryLog) LastError() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.lastErr
}

// QueryCount aggregates a query's occurrences and performance.
type QueryCount struct {
	// Query is the normalized query text.
	Query string
	// Count is how many times the query was recorded.
	Count int
	// AvgLatency is the mean latency across its occurrences.
	AvgLatency time.Duration
	// LastSeen is the most recent time the query was recorded.
	LastSeen time.Time
}

// PopularQueries returns the most frequently recorded queries, most frequent
// first. Queries are grouped after normalizing case and surrounding
// whitespace. If limit is positive, at most limit entries are returned.
func (l *QueryLog) PopularQueries(limit int) []QueryCount {
	type agg struct {
		count int
		total time.Duration
		last  time.Time
	}
	m := make(map[string]*agg)
	for _, r := range l.Records() {
		key := normalizeQuery(r.Query)
		a := m[key]
		if a == nil {
			a = &agg{}
			m[key] = a
		}
		a.count++
		a.total += r.Latency
		if r.Time.After(a.last) {
			a.last = r.Time
		}
	}
	out := make([]QueryCount, 0, len(m))
	for q, a := range m {
		out = append(out, QueryCount{
			Query:      q,
			Count:      a.count,
			AvgLatency: a.total / time.Duration(a.count),
			LastSeen:   a.last,
		})
	}
	sortByCountDesc(out)
	return limitTop(out, limit)
}

// DropOffQuery aggregates a query that repeatedly failed to return good
// results.
type DropOffQuery struct {
	// Query is the normalized query text.
	Query string
	// Count is how many times the query dropped off.
	Count int
	// AvgTopScore is the mean top score across its drop-offs.
	AvgTopScore float64
	// LastSeen is the most recent time the query dropped off.
	LastSeen time.Time
}

// DropOff returns queries that did not return good results, most frequent
// first. A query is a drop-off when it errored, returned no results, or had a
// top score below threshold. If limit is positive, at most limit entries are
// returned.
func (l *QueryLog) DropOff(threshold float64, limit int) []DropOffQuery {
	type agg struct {
		count    int
		totalTop float64
		last     time.Time
	}
	m := make(map[string]*agg)
	for _, r := range l.Records() {
		dropped := r.Error != "" || r.Results == 0 || r.TopScore < threshold
		if !dropped {
			continue
		}
		key := normalizeQuery(r.Query)
		a := m[key]
		if a == nil {
			a = &agg{}
			m[key] = a
		}
		a.count++
		a.totalTop += r.TopScore
		if r.Time.After(a.last) {
			a.last = r.Time
		}
	}
	out := make([]DropOffQuery, 0, len(m))
	for q, a := range m {
		out = append(out, DropOffQuery{
			Query:       q,
			Count:       a.count,
			AvgTopScore: a.totalTop / float64(a.count),
			LastSeen:    a.last,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Query < out[j].Query
	})
	return limitTop(out, limit)
}

// normalizeQuery canonicalizes a query for grouping: trims surrounding
// whitespace and lowercases.
func normalizeQuery(q string) string {
	return strings.ToLower(strings.TrimSpace(q))
}

// sortByCountDesc sorts QueryCount entries by count descending, then query.
func sortByCountDesc(out []QueryCount) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Query < out[j].Query
	})
}

// limitTop truncates a sorted slice to at most limit entries.
func limitTop[T any](out []T, limit int) []T {
	if limit > 0 && len(out) > limit {
		return out[:limit]
	}
	return out
}

var recordCounter int64

// newRecordID returns a unique record ID.
func newRecordID() string {
	n := atomic.AddInt64(&recordCounter, 1)
	return fmt.Sprintf("q-%d-%d", time.Now().UnixNano(), n)
}
