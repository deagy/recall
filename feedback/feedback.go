// Package feedback provides relevance feedback and query expansion for
// improving retrieval quality. It implements the classic Rocchio algorithm in
// both vector and lexical forms, and an iterative "expand-and-retrieve" loop.
//
// All types are thread-safe and dependency-free. Feedback can be accumulated
// in a Collector and serialized to chunk metadata for future training.
package feedback

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/deagy/recall/core"
)

// Label is a human relevance judgment for a retrieved chunk.
type Label int

const (
	// LabelUnlabeled means no judgment was recorded for the chunk.
	LabelUnlabeled Label = iota
	// LabelRelevant marks the chunk as relevant to the query.
	LabelRelevant
	// LabelNotRelevant marks the chunk as not relevant to the query.
	LabelNotRelevant
)

// String returns a human-readable label name.
func (l Label) String() string {
	switch l {
	case LabelRelevant:
		return "relevant"
	case LabelNotRelevant:
		return "not_relevant"
	default:
		return "unlabeled"
	}
}

// Feedback captures one round of relevance feedback for a query: which
// retrieved chunks a human judged relevant or not relevant.
type Feedback struct {
	// ID is a unique identifier for this feedback round.
	ID string

	// Query is the original query text this feedback applies to.
	Query string

	// Time is when the feedback was recorded.
	Time time.Time

	// Comment is an optional free-form annotation from the reviewer.
	Comment string

	// Labels maps chunk ID to relevance label.
	Labels map[string]Label
}

// NewFeedback creates a Feedback with a generated ID, timestamp, and labels.
// A nil labels map is replaced with an empty one.
func NewFeedback(query string, labels map[string]Label) *Feedback {
	if labels == nil {
		labels = map[string]Label{}
	}
	return &Feedback{
		ID:     newID("fb"),
		Query:  query,
		Time:   time.Now().UTC(),
		Labels: labels,
	}
}

// Relevant returns the chunk IDs labeled as relevant, sorted for determinism.
func (f *Feedback) Relevant() []string { return f.labeled(LabelRelevant) }

// Irrelevant returns the chunk IDs labeled as not relevant, sorted.
func (f *Feedback) Irrelevant() []string { return f.labeled(LabelNotRelevant) }

func (f *Feedback) labeled(l Label) []string {
	var out []string
	for id, lab := range f.Labels {
		if lab == l {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// HasJudgment reports whether at least one relevant or not-relevant label exists.
func (f *Feedback) HasJudgment() bool {
	for _, l := range f.Labels {
		if l == LabelRelevant || l == LabelNotRelevant {
			return true
		}
	}
	return false
}

// ToMetadata serializes the feedback into chunk-metadata-friendly values so it
// can be persisted on chunks or documents for future training or audit.
func (f *Feedback) ToMetadata() map[string]core.Value {
	md := make(map[string]core.Value)
	md["feedback.id"] = core.String{Value: f.ID}
	md["feedback.query"] = core.String{Value: f.Query}
	md["feedback.time"] = core.String{Value: f.Time.UTC().Format(time.RFC3339Nano)}
	md["feedback.relevant"] = core.String{Value: strings.Join(f.Relevant(), ",")}
	md["feedback.irrelevant"] = core.String{Value: strings.Join(f.Irrelevant(), ",")}
	if f.Comment != "" {
		md["feedback.comment"] = core.String{Value: f.Comment}
	}
	return md
}

// Collector is a thread-safe in-memory store of feedback rounds. It acts as the
// "training store" for relevance feedback: accumulate human judgments here,
// then aggregate them to drive Rocchio expansion or offline model training.
type Collector struct {
	mu    sync.RWMutex
	items []*Feedback
}

// NewCollector creates an empty Collector.
func NewCollector() *Collector { return &Collector{} }

// Add records a feedback round.
func (c *Collector) Add(f *Feedback) {
	if f == nil {
		return
	}
	c.mu.Lock()
	c.items = append(c.items, f)
	c.mu.Unlock()
}

// All returns a copy of every feedback round in insertion order.
func (c *Collector) All() []*Feedback {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]*Feedback, len(c.items))
	copy(out, c.items)
	return out
}

// ByQuery returns all feedback rounds for a query, in insertion order.
func (c *Collector) ByQuery(query string) []*Feedback {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []*Feedback
	for _, f := range c.items {
		if f.Query == query {
			out = append(out, f)
		}
	}
	return out
}

// Count returns the number of recorded feedback rounds.
func (c *Collector) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// RelevantFor returns the sorted union of relevant chunk IDs across all
// feedback rounds for a query.
func (c *Collector) RelevantFor(query string) []string {
	return c.unionFor(query, LabelRelevant)
}

// IrrelevantFor returns the sorted union of not-relevant chunk IDs across all
// feedback rounds for a query.
func (c *Collector) IrrelevantFor(query string) []string {
	return c.unionFor(query, LabelNotRelevant)
}

func (c *Collector) unionFor(query string, l Label) []string {
	seen := make(map[string]struct{})
	c.mu.RLock()
	for _, f := range c.items {
		if f.Query != query {
			continue
		}
		for id, lab := range f.Labels {
			if lab == l {
				seen[id] = struct{}{}
			}
		}
	}
	c.mu.RUnlock()
	out := make([]string, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// newID returns a short unique identifier with the given prefix.
func newID(prefix string) string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
	}
	return prefix + "-" + hex.EncodeToString(b)
}
