package store

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// MemoryStore is an in-memory store that uses an in-memory index.
type MemoryStore struct {
	mu        sync.RWMutex
	config    Config
	embedder  embedder.Embedder
	chunker   chunker.Chunker
	indexes   map[string]*index.MemoryIndex
	docChunks map[string]map[string]bool // docID -> set of chunkIDs
}

// NewMemoryStore creates a new in-memory store.
func NewMemoryStore(cfg Config) (*MemoryStore, error) {
	if cfg.Namespace == "" {
		cfg.Namespace = "default"
	}
	if cfg.Embedder == nil {
		cfg.Embedder = embedder.NewMockEmbedder(384)
	}
	if cfg.ChunkerFactory == nil {
		cfg.ChunkerFactory = chunker.NewFixed
	}

	return &MemoryStore{
		config:    cfg,
		embedder:  cfg.Embedder,
		chunker:   cfg.ChunkerFactory(chunker.DefaultConfig()),
		indexes:   make(map[string]*index.MemoryIndex),
		docChunks: make(map[string]map[string]bool),
	}, nil
}

// Upload processes a document: chunks it, embeds the chunks, and indexes them.
func (s *MemoryStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	if doc == nil {
		return core.ErrInvalidDocument
	}
	if content == "" {
		return core.ErrInvalidChunk
	}

	// Chunk the document
	chunks, err := s.chunker.Chunk(doc, content)
	if err != nil {
		return fmt.Errorf("chunking: %w", err)
	}
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks produced")
	}

	// Embed all chunks
	contents := make([]string, len(chunks))
	for i, c := range chunks {
		contents[i] = c.Content
	}

	embeddings, err := s.embedder.EmbedBatch(ctx, contents)
	if err != nil {
		return fmt.Errorf("embedding: %w", err)
	}

	// Assign embeddings to chunks
	for i, chunk := range chunks {
		chunk.Embedding = embeddings[i]
	}

	// Get or create index for the document's namespace (per-document
	// override falls back to the store's configured namespace).
	ns := doc.Namespace
	if ns == "" {
		ns = s.config.Namespace
	}
	stampChunkNamespace(chunks, ns)
	s.mu.Lock()
	idx, ok := s.indexes[ns]
	if !ok {
		idx = index.NewMemoryIndex(ns, s.embedder.Dimension())
		s.indexes[ns] = idx
	}
	s.mu.Unlock()

	// Add chunks to the index (also indexes BM25 internally; the index is
	// the single keyword source and prunes it on Delete).
	if err := idx.AddBatch(ctx, chunks); err != nil {
		return fmt.Errorf("indexing: %w", err)
	}

	// Track document -> chunk mappings
	s.mu.Lock()
	if _, ok := s.docChunks[doc.ID]; !ok {
		s.docChunks[doc.ID] = make(map[string]bool)
	}
	for _, c := range chunks {
		s.docChunks[doc.ID][c.ID] = true
	}
	s.mu.Unlock()

	doc.ChunkCount = len(chunks)
	return nil
}

// Search finds the most relevant chunks for a query string.
func (s *MemoryStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	// Embed the query
	queryEmbed, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	// Search across all namespaces
	var allResults []index.SearchResult
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, idx := range s.indexes {
		results, err := idx.Search(ctx, queryEmbed, opts)
		if err != nil {
			return nil, err
		}
		allResults = append(allResults, results...)
	}

	// Sort all results by score
	sortResults(allResults)

	// Limit to top K
	if len(allResults) > opts.TopK {
		allResults = allResults[:opts.TopK]
	}

	return allResults, nil
}

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
func (s *MemoryStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	// Embed the query for vector search
	queryEmbed, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	// Perform vector search across all namespaces
	var vecResults []index.SearchResult
	s.mu.RLock()
	for _, idx := range s.indexes {
		results, err := idx.Search(ctx, queryEmbed, opts)
		if err != nil {
			s.mu.RUnlock()
			return nil, err
		}
		vecResults = append(vecResults, results...)
	}
	s.mu.RUnlock()

	// Perform BM25 keyword search across all namespaces. Each index's
	// internal BM25 is the single keyword source: it is pruned on every
	// Delete, so deleted chunks never score here.
	bm25ResultsMap := make(map[string]float64) // chunkID -> BM25 score
	s.mu.RLock()
	for _, idx := range s.indexes {
		for _, r := range idx.SearchBM25(query) {
			bm25ResultsMap[r.DocID] = r.Score
		}
	}
	s.mu.RUnlock()

	// Fuse scores
	fused := fuseMap(vecResults, bm25ResultsMap, func(id string) *core.Chunk {
		s.mu.RLock()
		defer s.mu.RUnlock()
		for _, idx := range s.indexes {
			if chunk, ok := idx.GetChunk(id); ok {
				return chunk
			}
		}
		return nil
	}, opts)

	// Sort and limit
	sort.Slice(fused, func(i, j int) bool {
		return fused[i].Score > fused[j].Score
	})
	if len(fused) > opts.TopK {
		fused = fused[:opts.TopK]
	}

	return fused, nil
}

// fuseMap combines vector search results with BM25 scores using the
// configured fusion method. Chunks that match only the keyword (BM25) side
// are resolved through lookup, so strong keyword hits with weak vector
// similarity are not silently dropped; lookup must return nil for chunks
// that are no longer in the index (e.g. deleted).
func fuseMap(vecResults []index.SearchResult, bm25Scores map[string]float64, lookup func(id string) *core.Chunk, opts index.SearchOptions) []index.SearchResult {
	vecScoreMap := make(map[string]float64, len(vecResults))
	chunkByID := make(map[string]*core.Chunk, len(vecResults))
	for _, r := range vecResults {
		vecScoreMap[r.Chunk.ID] = r.Score
		chunkByID[r.Chunk.ID] = r.Chunk
	}

	allIDs := make(map[string]bool, len(vecScoreMap)+len(bm25Scores))
	for id := range vecScoreMap {
		allIDs[id] = true
	}
	for id := range bm25Scores {
		allIDs[id] = true
	}

	// A custom fusion is computed once for the full score maps.
	var fusedMap map[string]float64
	if opts.Fusion != nil {
		fusedMap = opts.Fusion.Fuse(vecScoreMap, bm25Scores)
	}

	var results []index.SearchResult
	for id := range allIDs {
		var fusedScore float64
		if opts.Fusion != nil {
			fusedScore = fusedMap[id]
		} else {
			// Weighted sum per SearchOptions.BM25Weight: 0 = pure vector,
			// 1 = pure BM25.
			fusedScore = (1-opts.BM25Weight)*vecScoreMap[id] + opts.BM25Weight*bm25Scores[id]
		}
		if fusedScore <= 0 {
			continue
		}

		chunk := chunkByID[id]
		if chunk == nil {
			// Keyword-only match: resolve the full chunk from the index.
			chunk = lookup(id)
		}
		if chunk == nil {
			// No longer present in the index (e.g. deleted): skip.
			continue
		}
		// Vector results are already filtered at the index level; keyword-only
		// matches are not, so the final chunk must satisfy the filters too.
		if !chunkMatchesFilters(chunk, opts.Filters) {
			continue
		}
		results = append(results, index.SearchResult{Chunk: chunk, Score: fusedScore})
	}
	return results
}

// chunkMatchesFilters reports whether the chunk satisfies every metadata
// filter (nil filters are ignored).
func chunkMatchesFilters(chunk *core.Chunk, filters []index.Filter) bool {
	for _, f := range filters {
		if f != nil && !f.Match(chunk) {
			return false
		}
	}
	return true
}

// GetChunk returns a chunk by its ID.
func (s *MemoryStore) GetChunk(id string) (*core.Chunk, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, idx := range s.indexes {
		if chunk, ok := idx.GetChunk(id); ok {
			return chunk, true
		}
	}
	return nil, false
}

// DeleteChunk removes a chunk from the store.
func (s *MemoryStore) DeleteChunk(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, idx := range s.indexes {
		if err := idx.Delete(ctx, id); err == nil {
			return nil
		}
	}
	return core.ErrNotFound
}

// DeleteDocument removes all chunks belonging to a document.
func (s *MemoryStore) DeleteDocument(ctx context.Context, docID string) error {
	s.mu.Lock()
	chunkIDs, ok := s.docChunks[docID]
	if !ok {
		s.mu.Unlock()
		return core.ErrNotFound
	}
	// Remove from indexes
	indexes := make([]*index.MemoryIndex, 0, len(s.indexes))
	for _, idx := range s.indexes {
		indexes = append(indexes, idx)
	}
	s.mu.Unlock()

	firstErr := error(nil)
	for id := range chunkIDs {
		for _, idx := range indexes {
			if err := idx.Delete(ctx, id); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}

	s.mu.Lock()
	delete(s.docChunks, docID)
	s.mu.Unlock()
	return firstErr
}

// Count returns the total number of chunks across all namespaces.
func (s *MemoryStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	total := 0
	for _, idx := range s.indexes {
		total += idx.Count()
	}
	return total
}

// Namespaces returns the list of namespaces in the store.
func (s *MemoryStore) Namespaces() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ns := make([]string, 0, len(s.indexes))
	for name := range s.indexes {
		ns = append(ns, name)
	}
	return ns
}

// Namespace returns the store's default namespace (used for documents that
// do not override it).
func (s *MemoryStore) Namespace() string {
	return s.config.Namespace
}

// Close cleans up resources.
func (s *MemoryStore) Close() error {
	return nil
}

// sortResults sorts search results by score descending.
func sortResults(results []index.SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}
