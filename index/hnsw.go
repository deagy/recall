// Package index provides the interface and implementations for storing
// and retrieving text chunks with their embeddings.
package index

import (
	"container/heap"
	"context"
	"math"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/deagy/recall/bm25"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// MemoryIndex is an in-memory index that stores chunks and their embeddings.
// For datasets > HNSWThreshold, it uses an HNSW graph for approximate nearest neighbor search.
type MemoryIndex struct {
	mu                 sync.RWMutex
	namespace          string
	dimension          int
	chunks             map[string]*core.Chunk
	bm25               *bm25.BM25
	hnsw               *HNSW
	hnswEnabled        bool
	deleted            map[string]bool // tombstones for HNSW
	tombstoneThreshold float64         // ratio of deleted entries to trigger rebuild
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

	m.syncHNSW(chunk)
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
		m.syncHNSW(chunk)
	}

	return nil
}

// syncHNSW keeps the HNSW graph in sync with newly added chunks: it builds the
// graph once the threshold is crossed and inserts chunks individually
// afterwards, so chunks added after activation remain searchable.
// Callers must hold m.mu.
func (m *MemoryIndex) syncHNSW(chunk *core.Chunk) {
	if !m.hnswEnabled {
		if len(m.chunks) > HNSWThreshold {
			m.buildHNSW()
		}
		return
	}
	if !m.hnsw.Contains(chunk.ID) {
		m.hnsw.Add(chunk.ID, chunk.Embedding)
	}
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

	// Check if HNSW is enabled and tombstone ratio exceeds threshold
	if m.hnswEnabled {
		if m.deleted == nil {
			m.deleted = make(map[string]bool)
		}
		m.deleted[id] = true
		// Drop the chunk so it is invisible to searches, Count, and GetChunk
		// until the graph is rebuilt.
		delete(m.chunks, id)
		m.bm25.RemoveDocument(id)
		if float64(len(m.deleted)) > m.tombstoneThreshold*float64(len(m.chunks)) {
			m.rebuildHNSW()
		}
	} else {
		delete(m.chunks, id)
		m.bm25.RemoveDocument(id)
	}
	return nil
}

// rebuildHNSW rebuilds the HNSW graph excluding tombstoned entries.
func (m *MemoryIndex) rebuildHNSW() {
	cfg := DefaultHNSWConfig()
	h := NewHNSW(m.dimension, cfg)
	for id, chunk := range m.chunks {
		if !m.deleted[id] {
			h.Add(id, chunk.Embedding)
		}
	}
	m.hnsw = h
	m.deleted = make(map[string]bool)
}

// NewMemoryIndex creates a new in-memory index.
func NewMemoryIndex(namespace string, dimension int) *MemoryIndex {
	return &MemoryIndex{
		namespace:          namespace,
		dimension:          dimension,
		chunks:             make(map[string]*core.Chunk),
		bm25:               bm25.New(bm25.DefaultConfig()),
		tombstoneThreshold: 0.2, // rebuild when 20% of entries are deleted
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

// SearchBM25 returns keyword (BM25) matches for the query from this
// index's internal BM25 index, sorted by score descending. This is the
// index's single source of keyword state: documents are added on
// Add/AddBatch and pruned on Delete, so deleted chunks never match.
func (m *MemoryIndex) SearchBM25(query string) []bm25.SearchResult {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.bm25.Search(query)
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
	// PCG avoids the deprecated v1 NewSource constructor while keeping a
	// fixed seed so layer assignment stays deterministic across runs.
	return &HNSW{
		dim:     dim,
		cfg:     cfg,
		entries: make([]int, 0),
		nodes:   make([]*hnswNode, 0),
		nodeIdx: make(map[string]int),
		rng:     rand.New(rand.NewPCG(42, 0)),
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
		id:        id,
		embedding: embedding,
		layer:     h.layerHeight(),
	}
	node.connections = make([][]int, node.layer+1)
	for l := 0; l <= node.layer; l++ {
		node.connections[l] = make([]int, 0)
	}
	h.nodes = append(h.nodes, node)
	h.nodeIdx[id] = idx

	if len(h.nodes) == 1 {
		// The first node is the entry point of the graph.
		h.entries = make([]int, node.layer+1)
		for l := 0; l <= node.layer; l++ {
			h.entries[l] = idx
		}
		return
	}

	maxLayer := len(h.entries) - 1

	// Find the starting entry on the topmost layer the node occupies.
	entry := -1
	if node.layer <= maxLayer {
		entry = h.entries[maxLayer]
		for l := maxLayer; l > node.layer; l-- {
			entry = h.greedyClosest(entry, l, embedding)
		}
	}

	// Connect the node on each of its layers, top to bottom.
	for l := node.layer; l >= 0; l-- {
		if entry == -1 {
			if l >= len(h.entries) {
				// Brand-new level above the old top: this node is the only
				// occupant, so there are no peers to link to.
				continue
			}
			entry = h.entries[l]
		}

		maxConn := h.cfg.M
		if l == 0 {
			maxConn = h.cfg.M0
		}

		candidates := h.searchLayer(entry, l, embedding, h.cfg.EfConstruction)
		if len(candidates) == 0 {
			continue
		}

		n := len(candidates)
		if n > maxConn {
			n = maxConn
		}
		for i := 0; i < n; i++ {
			node.connections[l] = append(node.connections[l], candidates[i].idx)
			h.linkBack(candidates[i].idx, idx, l, maxConn)
		}

		// The closest candidate seeds the search on the next layer down.
		entry = candidates[0].idx
	}

	// Publish the node as entry point for any levels it opened.
	for len(h.entries) <= node.layer {
		h.entries = append(h.entries, idx)
	}
}

// Contains reports whether a node with the given ID is present in the graph.
func (h *HNSW) Contains(id string) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	_, ok := h.nodeIdx[id]
	return ok
}

// linkBack adds idx to the neighbor's connections on layer, keeping the list
// at most maxConn entries. When the list is full one slot is reserved for the
// new link (so fresh nodes stay reachable) and the remaining maxConn-1 slots
// go to the closest old neighbors. Callers must hold h.mu.
func (h *HNSW) linkBack(neighbor, idx, layer, maxConn int) {
	n := h.nodes[neighbor]
	if layer >= len(n.connections) {
		return
	}
	conn := n.connections[layer]
	for _, c := range conn {
		if c == idx {
			return
		}
	}
	if len(conn) < maxConn {
		n.connections[layer] = append(conn, idx)
		return
	}

	scored := make([]hnswCand, 0, len(conn))
	for _, c := range conn {
		if c == neighbor {
			continue
		}
		scored = append(scored, hnswCand{c, cosineSim(n.embedding, h.nodes[c].embedding)})
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].score > scored[j].score })
	trimmed := make([]int, 0, maxConn)
	for i := 0; i < len(scored) && i < maxConn-1; i++ {
		trimmed = append(trimmed, scored[i].idx)
	}
	trimmed = append(trimmed, idx)
	n.connections[layer] = trimmed
}

// greedyClosest follows layer l links from start, returning the index of the
// node closest to query. Callers must hold h.mu.
func (h *HNSW) greedyClosest(start, layer int, query []float32) int {
	best := start
	if layer >= len(h.nodes[best].connections) {
		return best
	}
	bestScore := cosineSim(query, h.nodes[best].embedding)
	improved := true
	for improved {
		improved = false
		for _, nIdx := range h.nodes[best].connections[layer] {
			score := cosineSim(query, h.nodes[nIdx].embedding)
			if score > bestScore {
				bestScore = score
				best = nIdx
				improved = true
			}
		}
	}
	return best
}

// hnswCand is a node index paired with its cosine similarity score.
type hnswCand struct {
	idx   int
	score float64
}

// candMaxHeap is a max-heap of hnswCand so the best candidate pops first.
type candMaxHeap []hnswCand

func (q candMaxHeap) Len() int           { return len(q) }
func (q candMaxHeap) Less(i, j int) bool { return q[i].score > q[j].score }
func (q candMaxHeap) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }

func (q *candMaxHeap) Push(x any) { *q = append(*q, x.(hnswCand)) }

func (q *candMaxHeap) Pop() any {
	old := *q
	n := len(old)
	x := old[n-1]
	*q = old[:n-1]
	return x
}

// searchLayer beam-searches layer l starting from entry, returning the up to
// ef closest nodes sorted by descending similarity. Callers must hold h.mu.
func (h *HNSW) searchLayer(entry, layer int, query []float32, ef int) []hnswCand {
	if entry < 0 || layer < 0 {
		return nil
	}

	visited := make(map[int]bool)
	var candidates candMaxHeap
	results := make([]hnswCand, 0, ef)

	entryScore := cosineSim(query, h.nodes[entry].embedding)
	visited[entry] = true
	candidates = append(candidates, hnswCand{entry, entryScore})
	results = append(results, hnswCand{entry, entryScore})
	worst := entryScore
	heap.Init(&candidates)

	for len(candidates) > 0 {
		curr := heap.Pop(&candidates).(hnswCand)
		if len(results) >= ef && curr.score < worst {
			break
		}
		if layer >= len(h.nodes[curr.idx].connections) {
			continue
		}
		for _, nIdx := range h.nodes[curr.idx].connections[layer] {
			if visited[nIdx] {
				continue
			}
			visited[nIdx] = true
			score := cosineSim(query, h.nodes[nIdx].embedding)
			if len(results) < ef {
				heap.Push(&candidates, hnswCand{nIdx, score})
				results = append(results, hnswCand{nIdx, score})
				if score < worst {
					worst = score
				}
			} else if score > worst {
				// Replace the current worst with the better candidate.
				heap.Push(&candidates, hnswCand{nIdx, score})
				worstIdx := 0
				for i := range results {
					if results[i].score < results[worstIdx].score {
						worstIdx = i
					}
				}
				results[worstIdx] = hnswCand{nIdx, score}
				worst = results[0].score
				for i := 1; i < len(results); i++ {
					if results[i].score < worst {
						worst = results[i].score
					}
				}
			}
		}
	}

	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	return results
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

	entry := h.entries[len(h.entries)-1]
	for l := len(h.entries) - 1; l > 0; l-- {
		entry = h.greedyClosest(entry, l, query)
	}

	results := h.searchLayer(entry, 0, query, ef)
	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = h.nodes[r.idx].id
	}
	return ids
}
