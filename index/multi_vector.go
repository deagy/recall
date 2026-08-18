package index

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// Aggregation selects how a chunk's multiple embeddings are collapsed
// into a single similarity score.
type Aggregation int

const (
	// MaxSimAggregation scores a chunk by its best (max) single-vector
	// similarity — the ColBERT-style MaxSim upper bound. Default.
	MaxSimAggregation Aggregation = iota

	// MeanAggregation scores a chunk by the mean of all its vector
	// similarities. Robust to one spurious vector.
	MeanAggregation

	// TopMeanAggregation scores a chunk by the mean of its TopN best
	// similarities (0 falls back to MaxSim semantics when TopN == 1).
	TopMeanAggregation
)

// multiVectorItem holds one chunk plus all of its embeddings.
type multiVectorItem struct {
	chunk   *core.Chunk
	vectors [][]float32
}

// MultiVectorIndex stores multiple embeddings per chunk (e.g. one per
// passage segment, or separate "query-side" and "passage-side" vectors)
// and scores each stored vector against the query, aggregating per the
// configured Aggregation. This supports retrieval models where a single
// vector under-represents a chunk.
type MultiVectorIndex struct {
	ns  string
	dim int

	// Aggregation selects the scoring mode. Zero value = MaxSim.
	Aggregation Aggregation

	// TopN is the number of best vectors used by TopMeanAggregation.
	TopN int

	mu    sync.RWMutex
	items map[string]*multiVectorItem
}

// NewMultiVectorIndex creates a MultiVectorIndex for the given
// namespace and embedding dimension.
func NewMultiVectorIndex(ns string, dim int) *MultiVectorIndex {
	return &MultiVectorIndex{
		ns:    ns,
		dim:   dim,
		items: make(map[string]*multiVectorItem),
	}
}

// Add indexes a chunk using its single Embedding as the one-vector
// multi-set.
func (m *MultiVectorIndex) Add(_ context.Context, chunk *core.Chunk) error {
	if chunk == nil {
		return core.ErrInvalidChunk
	}
	if len(chunk.Embedding) == 0 {
		return core.ErrInvalidEmbedding
	}
	return m.addMultiLocked(chunk, [][]float32{chunk.Embedding})
}

// AddMulti indexes a chunk with an explicit set of embeddings. Each
// vector must have the index dimension and the set must be non-empty.
func (m *MultiVectorIndex) AddMulti(_ context.Context, chunk *core.Chunk, vectors [][]float32) error {
	if chunk == nil {
		return core.ErrInvalidChunk
	}
	if len(vectors) == 0 {
		return fmt.Errorf("multi-vector: chunk %s has no vectors", chunk.ID)
	}
	for _, v := range vectors {
		if len(v) != m.dim {
			return core.ErrEmbeddingMismatch
		}
	}
	return m.addMultiLocked(chunk, vectors)
}

func (m *MultiVectorIndex) addMultiLocked(chunk *core.Chunk, vectors [][]float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Keep owned copies so later mutation of the caller's slices cannot
	// change the index.
	owned := make([][]float32, len(vectors))
	for i, v := range vectors {
		owned[i] = append([]float32(nil), v...)
	}
	m.items[chunk.ID] = &multiVectorItem{chunk: chunk, vectors: owned}
	return nil
}

// AddBatch indexes multiple single-embedding chunks.
func (m *MultiVectorIndex) AddBatch(ctx context.Context, chunks []*core.Chunk) error {
	for _, c := range chunks {
		if err := m.Add(ctx, c); err != nil {
			return err
		}
	}
	return nil
}

// Delete removes a chunk and all of its vectors.
func (m *MultiVectorIndex) Delete(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.items[id]; !ok {
		return core.ErrNotFound
	}
	delete(m.items, id)
	return nil
}

// Search scores every stored vector against the query and aggregates
// per chunk.
func (m *MultiVectorIndex) Search(_ context.Context, query []float32, opts SearchOptions) ([]SearchResult, error) {
	if len(query) != m.dim {
		return nil, core.ErrEmbeddingMismatch
	}
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	type scored struct {
		chunk *core.Chunk
		score float64
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []scored
	for _, item := range m.items {
		if !matchesAllFilters(item.chunk, opts.Filters) {
			continue
		}
		sims := make([]float64, len(item.vectors))
		for i, v := range item.vectors {
			sims[i] = embedder.CosineSimilarity(query, v)
		}
		score := aggregate(m.Aggregation, m.TopN, sims)
		if score < opts.MinScore {
			continue
		}
		results = append(results, scored{item.chunk, score})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].score != results[j].score {
			return results[i].score > results[j].score
		}
		return results[i].chunk.ID < results[j].chunk.ID
	})
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	out := make([]SearchResult, len(results))
	for i, r := range results {
		out[i] = SearchResult{Chunk: r.chunk, Score: r.score}
	}
	return out, nil
}

// GetChunk returns a chunk by ID.
func (m *MultiVectorIndex) GetChunk(id string) (*core.Chunk, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	item, ok := m.items[id]
	if !ok {
		return nil, false
	}
	return item.chunk, true
}

// VectorCount returns how many embeddings are stored for a chunk.
func (m *MultiVectorIndex) VectorCount(id string) (int, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	item, ok := m.items[id]
	if !ok {
		return 0, false
	}
	return len(item.vectors), true
}

// Count returns the number of chunks in the index.
func (m *MultiVectorIndex) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.items)
}

// Dimension returns the embedding dimension.
func (m *MultiVectorIndex) Dimension() int { return m.dim }

// Namespace returns the namespace of this index.
func (m *MultiVectorIndex) Namespace() string { return m.ns }

// aggregate collapses per-vector similarities per the aggregation mode.
func aggregate(mode Aggregation, topN int, sims []float64) float64 {
	switch mode {
	case MeanAggregation:
		var sum float64
		for _, s := range sims {
			sum += s
		}
		return sum / float64(len(sims))
	case TopMeanAggregation:
		sorted := append([]float64(nil), sims...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
		n := topN
		if n <= 0 {
			n = 1
		}
		if n > len(sorted) {
			n = len(sorted)
		}
		var sum float64
		for i := 0; i < n; i++ {
			sum += sorted[i]
		}
		return sum / float64(n)
	default: // MaxSimAggregation
		max := math.Inf(-1)
		for _, s := range sims {
			if s > max {
				max = s
			}
		}
		return max
	}
}
