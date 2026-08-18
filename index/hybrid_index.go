package index

import (
	"context"
	"sort"
	"sync"

	"github.com/deagy/recall/bm25"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/fuse"
)

// HybridIndex combines a dense vector index and a sparse BM25 keyword
// index over the same chunks into a single index that serves fused
// hybrid search. Vector-only or keyword-only hits are both retained:
// a chunk that ranks only on keywords (e.g. exact technical terms with
// weak semantic overlap) still surfaces.
type HybridIndex struct {
	ns  string
	dim int

	vec    *MemoryIndex
	kw     *bm25.BM25
	fusion fuse.Fusion

	mu     sync.RWMutex
	chunks map[string]*core.Chunk
}

// NewHybridIndex creates a HybridIndex. A nil fusion means Search will
// weight the two score maps by SearchOptions.BM25Weight (0.5/0.5 when
// unset).
func NewHybridIndex(ns string, dim int, fusion fuse.Fusion) *HybridIndex {
	return &HybridIndex{
		ns:     ns,
		dim:    dim,
		vec:    NewMemoryIndex(ns, dim),
		kw:     bm25.New(bm25.DefaultConfig()),
		fusion: fusion,
		chunks: make(map[string]*core.Chunk),
	}
}

// Add inserts a chunk into both sub-indexes.
func (h *HybridIndex) Add(ctx context.Context, chunk *core.Chunk) error {
	if chunk == nil {
		return core.ErrInvalidChunk
	}
	if len(chunk.Embedding) == 0 {
		return core.ErrInvalidEmbedding
	}
	if err := h.vec.Add(ctx, chunk); err != nil {
		return err
	}
	h.kw.AddDocument(chunk.ID, chunk.Content)
	h.mu.Lock()
	h.chunks[chunk.ID] = chunk
	h.mu.Unlock()
	return nil
}

// Delete removes a chunk from both sub-indexes.
func (h *HybridIndex) Delete(ctx context.Context, id string) error {
	if err := h.vec.Delete(ctx, id); err != nil {
		return err
	}
	h.kw.RemoveDocument(id)
	h.mu.Lock()
	delete(h.chunks, id)
	h.mu.Unlock()
	return nil
}

// Search performs fused hybrid search: dense similarity against
// queryEmb plus BM25 ranking against the raw query text.
func (h *HybridIndex) Search(ctx context.Context, query string, queryEmb []float32, opts SearchOptions) ([]SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	vecResults, err := h.vec.Search(ctx, queryEmb, opts)
	if err != nil {
		return nil, err
	}

	vecScores := make(map[string]float64, len(vecResults))
	chunkByID := make(map[string]*core.Chunk, len(vecResults))
	for _, r := range vecResults {
		vecScores[r.Chunk.ID] = r.Score
		chunkByID[r.Chunk.ID] = r.Chunk
	}

	kwScores := make(map[string]float64)
	for _, r := range h.kw.Search(query) {
		kwScores[r.DocID] = r.Score
	}

	allIDs := make(map[string]struct{}, len(vecScores)+len(kwScores))
	for id := range vecScores {
		allIDs[id] = struct{}{}
	}
	for id := range kwScores {
		allIDs[id] = struct{}{}
	}

	// Fusion precedence: SearchOptions.Fusion > constructor fusion >
	// weighted sum over SearchOptions.BM25Weight (default 0.5/0.5).
	var fusedMap map[string]float64
	fusion := opts.Fusion
	if fusion == nil {
		fusion = h.fusion
	}
	if fusion != nil {
		fusedMap = fusion.Fuse(vecScores, kwScores)
	}

	type scored struct {
		chunk *core.Chunk
		score float64
	}
	var results []scored
	h.mu.RLock()
	for id := range allIDs {
		var score float64
		if fusedMap != nil {
			score = fusedMap[id]
		} else {
			weight := opts.BM25Weight
			if weight == 0 {
				weight = 0.5
			}
			score = (1-weight)*vecScores[id] + weight*kwScores[id]
		}
		if score <= 0 || score < opts.MinScore {
			continue
		}
		chunk := chunkByID[id]
		if chunk == nil {
			// Keyword-only match: resolve from the shared chunk map and
			// apply the filter set (vector results are pre-filtered).
			chunk = h.chunks[id]
			if chunk == nil || !matchesAllFilters(chunk, opts.Filters) {
				continue
			}
		}
		results = append(results, scored{chunk, score})
	}
	h.mu.RUnlock()

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
func (h *HybridIndex) GetChunk(id string) (*core.Chunk, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	c, ok := h.chunks[id]
	return c, ok
}

// SearchBM25 exposes the keyword sub-index directly.
func (h *HybridIndex) SearchBM25(query string) []bm25.SearchResult {
	return h.kw.Search(query)
}

// Count returns the number of indexed chunks.
func (h *HybridIndex) Count() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.chunks)
}

// Dimension returns the embedding dimension.
func (h *HybridIndex) Dimension() int { return h.dim }

// Namespace returns the namespace of this index.
func (h *HybridIndex) Namespace() string { return h.ns }
