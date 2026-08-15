package index

import (
	"context"
	"sort"
	"sync"

	"github.com/deagy/recall/bm25"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// MemoryIndex is an in-memory index that stores chunks and their embeddings.
type MemoryIndex struct {
	mu        sync.RWMutex
	namespace string
	dimension int
	chunks    map[string]*core.Chunk
	bm25      *bm25.BM25
}

// NewMemoryIndex creates a new in-memory index.
func NewMemoryIndex(namespace string, dimension int) *MemoryIndex {
	return &MemoryIndex{
		namespace: namespace,
		dimension: dimension,
		chunks:    make(map[string]*core.Chunk),
		bm25:      bm25.New(bm25.DefaultConfig()),
	}
}

// Add inserts a chunk into the index.
func (m *MemoryIndex) Add(_ context.Context, chunk *core.Chunk) error {
	if chunk.Embedding == nil {
		return core.ErrInvalidEmbedding
	}
	if len(chunk.Embedding) != m.dimension {
		return core.ErrEmbeddingMismatch
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.chunks[chunk.ID] = chunk
	m.bm25.AddDocument(chunk.ID, chunk.Content)
	return nil
}

// AddBatch inserts multiple chunks into the index.
func (m *MemoryIndex) AddBatch(_ context.Context, chunks []*core.Chunk) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, chunk := range chunks {
		if chunk.Embedding == nil {
			return core.ErrInvalidEmbedding
		}
		if len(chunk.Embedding) != m.dimension {
			return core.ErrEmbeddingMismatch
		}
		m.chunks[chunk.ID] = chunk
		m.bm25.AddDocument(chunk.ID, chunk.Content)
	}
	return nil
}

// Delete removes a chunk from the index.
func (m *MemoryIndex) Delete(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.chunks, id)
	m.bm25.RemoveDocument(id)
	return nil
}

// Search finds the most similar chunks to the given query embedding.
// Search finds the most similar chunks to the given query embedding.
func (m *MemoryIndex) Search(_ context.Context, query []float32, opts SearchOptions) ([]SearchResult, error) {
	if len(query) != m.dimension {
		return nil, core.ErrEmbeddingMismatch
	}
	if opts.TopK <= 0 {
		opts.TopK = 10
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// Compute similarity for all chunks
	type scored struct {
		chunk *core.Chunk
		score float64
	}
	var results []scored

	for _, chunk := range m.chunks {
		if !matchesAllFilters(chunk, opts.Filters) {
			continue
		}
		score := embedder.CosineSimilarity(query, chunk.Embedding)
		if score < opts.MinScore {
			continue
		}
		results = append(results, scored{chunk: chunk, score: score})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}

	searchResults := make([]SearchResult, len(results))
	for i, r := range results {
		searchResults[i] = SearchResult{Chunk: r.chunk, Score: r.score}
	}
	return searchResults, nil
}

// Count returns the number of chunks in the index.
func (m *MemoryIndex) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.chunks)
}

// Dimension returns the embedding dimension.
func (m *MemoryIndex) Dimension() int {
	return m.dimension
}

// Namespace returns the namespace.
func (m *MemoryIndex) Namespace() string {
	return m.namespace
}

// GetChunk returns a chunk by ID.
func (m *MemoryIndex) GetChunk(id string) (*core.Chunk, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	c, ok := m.chunks[id]
	return c, ok
}

// matchesAllFilters returns true if the chunk matches all filters.
func matchesAllFilters(chunk *core.Chunk, filters []Filter) bool {
	for _, f := range filters {
		if !f.Match(chunk) {
			return false
		}
	}
	return true
}
