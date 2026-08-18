// Package index provides the interface and implementations for storing
// and retrieving text chunks with their embeddings.
package index

import (
	"context"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/fuse"
)

// SearchResult represents a single result from a similarity search.
type SearchResult struct {
	// Chunk is the matching chunk.
	Chunk *core.Chunk

	// Score is the relevance score (higher is more similar).
	Score float64

	// RerankScore is the fine-rank score assigned by a reranker (higher is
	// more relevant). It is zero until a reranker has processed this result.
	RerankScore float64

	// RerankRank is the 1-based position this result occupies after
	// reranking. It is zero when no reranker has run.
	RerankRank int

	// Reranker is the name of the reranker that produced RerankScore, for
	// score attribution. Empty when no reranker has run.
	Reranker string
}

// SearchOptions configures a search operation.
type SearchOptions struct {
	// TopK is the maximum number of results to return.
	TopK int

	// Filters are metadata filters to apply before searching.
	Filters []Filter

	// MinScore is the minimum relevance score for a result to be included.
	MinScore float64

	// Hybrid enables BM25 keyword search combined with vector similarity.
	Hybrid bool

	// BM25Weight is the weight for BM25 scores in hybrid search (0-1).
	// 0 = pure vector, 1 = pure BM25, 0.5 = equal weighting.
	BM25Weight float64

	// Fusion allows custom score fusion (overrides BM25Weight if set).
	Fusion fuse.Fusion

	// EfSearch controls the HNSW search width (only used when HNSW is active).
	// 0 means use the default (50).
	EfSearch int
}

// DefaultSearchOptions returns SearchOptions with sensible defaults.
func DefaultSearchOptions(topK int) SearchOptions {
	if topK <= 0 {
		topK = 10
	}
	return SearchOptions{
		TopK:     topK,
		MinScore: 0,
	}
}

// Filter is a metadata filter for search results.
type Filter interface {
	// Match returns true if the chunk's metadata matches this filter.
	Match(chunk *core.Chunk) bool
}

// Index defines the interface for storing and retrieving embedded chunks.
type Index interface {
	// Add inserts a chunk into the index. The chunk's embedding must be non-nil.
	Add(ctx context.Context, chunk *core.Chunk) error

	// AddBatch inserts multiple chunks into the index.
	AddBatch(ctx context.Context, chunks []*core.Chunk) error

	// Delete removes a chunk from the index by its ID.
	Delete(ctx context.Context, id string) error

	// Search finds the most similar chunks to the given query embedding.
	Search(ctx context.Context, query []float32, opts SearchOptions) ([]SearchResult, error)

	// Count returns the number of chunks in the index.
	Count() int

	// Dimension returns the embedding dimension of the index.
	Dimension() int

	// Namespace returns the namespace of this index.
	Namespace() string
}

// TermFilter matches chunks where a metadata key equals a specific string value.
type TermFilter struct {
	Key   string
	Value string
}

func (f *TermFilter) Match(chunk *core.Chunk) bool {
	return chunk.GetMetadataString(f.Key) == f.Value
}

// TermInFilter matches chunks where a metadata key's value is in a set.
type TermInFilter struct {
	Key    string
	Values []string
}

func (f *TermInFilter) Match(chunk *core.Chunk) bool {
	v := chunk.GetMetadataString(f.Key)
	for _, val := range f.Values {
		if v == val {
			return true
		}
	}
	return false
}

// RangeFilter matches chunks where a numeric metadata value is in a range.
type RangeFilter struct {
	Key     string
	Min     *float64
	Max     *float64
	MinIncl bool
	MaxIncl bool
}

func (f *RangeFilter) Match(chunk *core.Chunk) bool {
	v := chunk.GetMetadata(f.Key)
	if v == nil {
		return false
	}
	fVal, ok := core.ToFloat64(v)
	if !ok {
		return false
	}
	if f.Min != nil {
		if f.MinIncl && fVal < *f.Min {
			return false
		}
		if !f.MinIncl && fVal <= *f.Min {
			return false
		}
	}
	if f.Max != nil {
		if f.MaxIncl && fVal > *f.Max {
			return false
		}
		if !f.MaxIncl && fVal >= *f.Max {
			return false
		}
	}
	return true
}

// DateRangeFilter matches chunks where the CreatedAt is in a date range.
type DateRangeFilter struct {
	Key     string // metadata key containing a time string
	Min     *time.Time
	Max     *time.Time
	MinIncl bool
	MaxIncl bool
}

func (f *DateRangeFilter) Match(chunk *core.Chunk) bool {
	v := chunk.GetMetadata(f.Key)
	if v == nil {
		return false
	}
	// Try to parse as time
	s := v.String()
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t, err = time.Parse("2006-01-02", s)
		if err != nil {
			return false
		}
	}
	if f.Min != nil {
		if f.MinIncl && t.Before(*f.Min) {
			return false
		}
		if !f.MinIncl && !t.After(*f.Min) {
			return false
		}
	}
	if f.Max != nil {
		if f.MaxIncl && t.After(*f.Max) {
			return false
		}
		if !f.MaxIncl && !t.Before(*f.Max) {
			return false
		}
	}
	return true
}
