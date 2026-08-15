// Package index provides the interface and implementations for storing
// and retrieving text chunks with their embeddings.
package index

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/deagy/recall/bm25"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// MemoryIndex is an in-memory index that stores chunks and their embeddings.
// For datasets > HNSWThreshold, it uses an HNSW graph for approximate nearest neighbor search.
type MemoryIndex struct {
	mu          sync.RWMutex
	namespace   string
	dimension   int
	chunks      map[string]*core.Chunk
	bm25        *bm25.BM25
	hnsw        *HNSW
	hnswEnabled bool
}

// HNSWThreshold is the number of chunks above which HNSW is used for search.
const HNSWThreshold = 1000

// HNSWConfig holds configuration for the HNSW index.
type HNSWConfig struct {
	M              int
	M0             int
	EfConstruction int
	EfSearch       int
}

// DefaultHNSWConfig returns standard HNSW parameters.
func DefaultHNSWConfig() HNSWConfig {
	return HNSWConfig{
		M:              16,
		M0:             32,
		EfConstruction: 200,
		EfSearch:       50,
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

	if !m.hnswEnabled && len(m.chunks) > HNSWThreshold {
		m.buildHNSW()
	}
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

	if !m.hnswEnabled && len(m.chunks) > HNSWThreshold {
		m.buildHNSW()
	}
	return nil
}

// buildHNSW constructs the HNSW graph from current chunks.
func (m *MemoryIndex) buildHNSW() {
	cfg := DefaultHNSWConfig()
	h := NewHNSW(m.dimension, cfg)
	for _, chunk := range m.chunks {
		h.Add(chunk.ID, chunk.Embedding)
	}
	m.hnsw = h
	m.hnswEnabled = true
}

// Delete removes a chunk from the index.
func (m *MemoryIndex) Delete(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.chunks, id)
	m.bm25.RemoveDocument(id)
	if m.hnswEnabled {
		m.buildHNSW()
	}
	return nil
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

	if m.hnswEnabled && len(m.chunks) > HNSWThreshold {
		return m.searchHNSW(query, opts)
	}
	return m.searchBruteForce(query, opts)
}

// searchBruteForce performs exact nearest neighbor search.
func (m *MemoryIndex) searchBruteForce(query []float32, opts SearchOptions) ([]SearchResult, error) {
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

// searchHNSW performs approximate nearest neighbor search using the HNSW graph.
func (m *MemoryIndex) searchHNSW(query []float32, opts SearchOptions) ([]SearchResult, error) {
	hnswResults := m.hnsw.Search(query, opts.EfSearch)

	type scored struct {
		chunk *core.Chunk
		score float64
	}
	var results []scored

	for _, id := range hnswResults {
		chunk, ok := m.chunks[id]
		if !ok {
			continue
		}
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

// --- HNSW Implementation ---

type hnswNode struct {
	id          string
	embedding   []float32
	connections [][]int
	layer       int
}

type HNSW struct {
	mu      sync.RWMutex
	dim     int
	cfg     HNSWConfig
	entries []int
	nodes   []*hnswNode
	nodeIdx map[string]int
	rng     *rand.Rand
}

func NewHNSW(dim int, cfg HNSWConfig) *HNSW {
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.M0 <= 0 {
		cfg.M0 = 2 * cfg.M
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	if cfg.EfSearch <= 0 {
		cfg.EfSearch = 50
	}
	return &HNSW{
		dim:     dim,
		cfg:     cfg,
		entries: make([]int, 0),
		nodes:   make([]*hnswNode, 0),
		nodeIdx: make(map[string]int),
		rng:     rand.New(rand.NewSource(42)),
	}
}

func (h *HNSW) layerHeight() int {
	mu := 1.0 / math.Log(float64(h.cfg.M))
	r := h.rng.Float64()
	layer := int(-math.Log(r) * mu)
	maxLayer := len(h.entries)
	if layer > maxLayer {
		layer = maxLayer
	}
	return layer
}

func cosineSim(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Add inserts a new node into the HNSW graph.
func (h *HNSW) Add(id string, embedding []float32) {
	h.mu.Lock()
	defer h.mu.Unlock()

	idx := len(h.nodes)
	node := &hnswNode{
		id:          id,
		embedding:   embedding,
		connections: make([][]int, 0),
	}
	node.layer = h.layerHeight()

	for len(h.entries) <= node.layer {
		h.entries = append(h.entries, 0)
	}

	if len(h.nodes) == 0 {
		node.connections = make([][]int, node.layer+1)
		for l := 0; l <= node.layer; l++ {
			node.connections[l] = make([]int, 0)
		}
		h.entries = make([]int, node.layer+1)
		for l := 0; l <= node.layer; l++ {
			h.entries[l] = idx
		}
		h.nodes = append(h.nodes, node)
		h.nodeIdx[id] = idx
		return
	}

	for l := node.layer; l >= 0; l-- {
		maxConn := h.cfg.M
		if l == 0 {
			maxConn = h.cfg.M0
		}

		var candidates []int
		if l < len(h.entries) && h.entries[l] >= 0 {
			candidates = append(candidates, h.entries[l])
		}

		type scored struct {
			idx   int
			score float64
		}
		var candidateScores []scored
		for _, cIdx := range candidates {
			score := cosineSim(embedding, h.nodes[cIdx].embedding)
			candidateScores = append(candidateScores, scored{cIdx, score})
		}
		sort.Slice(candidateScores, func(i, j int) bool {
			return candidateScores[i].score > candidateScores[j].score
		})

		ef := h.cfg.EfConstruction
		if ef > len(candidateScores) {
			ef = len(candidateScores)
		}
		candidateScores = candidateScores[:ef]

		var neighbors []int
		for _, cs := range candidateScores {
			if cs.idx != idx {
				neighbors = append(neighbors, cs.idx)
			}
			if len(neighbors) >= maxConn {
				break
			}
		}

		if l >= len(node.connections) {
			newConn := make([][]int, l+1)
			copy(newConn, node.connections)
			node.connections = newConn
		}
		for _, nIdx := range neighbors {
			node.connections[l] = append(node.connections[l], nIdx)
			if l >= len(h.nodes[nIdx].connections) {
				newConn := make([][]int, l+1)
				copy(newConn, h.nodes[nIdx].connections)
				h.nodes[nIdx].connections = newConn
			}
			h.nodes[nIdx].connections[l] = append(h.nodes[nIdx].connections[l], idx)
		}

		if l < len(h.entries) {
			entryScore := cosineSim(embedding, h.nodes[h.entries[l]].embedding)
			nodeScore := cosineSim(embedding, node.embedding)
			if nodeScore > entryScore {
				h.entries[l] = idx
			}
		}
	}

	node.connections = node.connections[:node.layer+1]
	h.nodes = append(h.nodes, node)
	h.nodeIdx[id] = idx
}

// Search finds the nearest neighbors to the query embedding.
func (h *HNSW) Search(query []float32, ef int) []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.nodes) == 0 {
		return nil
	}
	if ef <= 0 {
		ef = h.cfg.EfSearch
	}

	current := h.entries[len(h.entries)-1]

	for l := len(h.entries) - 2; l >= 0; l-- {
		bestIdx := h.entries[l]
		bestScore := cosineSim(query, h.nodes[bestIdx].embedding)
		changed := true
		for changed {
			changed = false
			for _, neighbor := range h.nodes[bestIdx].connections[l] {
				score := cosineSim(query, h.nodes[neighbor].embedding)
				if score > bestScore {
					bestScore = score
					bestIdx = neighbor
					changed = true
				}
			}
		}
		current = bestIdx
	}

	type candidate struct {
		idx   int
		score float64
	}

	visited := make(map[int]bool)
	heap := make([]candidate, 0)
	heap = append(heap, candidate{current, cosineSim(query, h.nodes[current].embedding)})
	visited[current] = true

	var results []candidate

	for len(heap) > 0 {
		minIdx := 0
		for i := 1; i < len(heap); i++ {
			if heap[i].score < heap[minIdx].score {
				minIdx = i
			}
		}

		curr := heap[minIdx]
		heap = append(heap[:minIdx], heap[minIdx+1:]...)

		if len(results) >= ef && curr.score < results[len(results)-1].score {
			break
		}

		results = append(results, curr)
		sort.Slice(results, func(i, j int) bool {
			return results[i].score > results[j].score
		})

		for _, neighbor := range h.nodes[curr.idx].connections[0] {
			if !visited[neighbor] {
				visited[neighbor] = true
				score := cosineSim(query, h.nodes[neighbor].embedding)
				heap = append(heap, candidate{neighbor, score})
			}
		}
	}

	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = h.nodes[r.idx].id
	}
	return ids
}
