package store

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/deagy/recall/bm25"
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
	bm25s     map[string]*bm25.BM25
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
		bm25s:     make(map[string]*bm25.BM25),
		docChunks: make(map[string]map[string]bool),
	}, nil
}

// Upload processes a document: chunks it, embeds the chunks, and indexes them.
func (s *MemoryStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	if doc == nil {
		return core.ErrInvalidChunk
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

	// Get or create index for the document's namespace
	ns := s.config.Namespace
	s.mu.Lock()
	idx, ok := s.indexes[ns]
	if !ok {
		idx = index.NewMemoryIndex(ns, s.embedder.Dimension())
		s.indexes[ns] = idx
	}
	bm25Idx, ok := s.bm25s[ns]
	if !ok {
		bm25Idx = bm25.New(bm25.DefaultConfig())
		s.bm25s[ns] = bm25Idx
	}
	s.mu.Unlock()

	// Add chunks to the index (also indexes BM25 internally)
	if err := idx.AddBatch(ctx, chunks); err != nil {
		return fmt.Errorf("indexing: %w", err)
	}

	// Add chunks to BM25 index
	for _, c := range chunks {
		bm25Idx.AddDocument(c.ID, c.Content)
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

	// Perform BM25 search across all namespaces
	bm25ResultsMap := make(map[string]float64) // chunkID -> BM25 score
	s.mu.RLock()
	for _, bm25Idx := range s.bm25s {
		results := bm25Idx.Search(query)
		for _, r := range results {
			bm25ResultsMap[r.DocID] = r.Score
		}
	}
	s.mu.RUnlock()

	// Fuse scores
	fused := fuseMap(vecResults, bm25ResultsMap, opts)

	// Sort and limit
	sort.Slice(fused, func(i, j int) bool {
		return fused[i].Score > fused[j].Score
	})
	if len(fused) > opts.TopK {
		fused = fused[:opts.TopK]
	}

	return fused, nil
}

// fuseMap combines vector search results with BM25 scores using the configured fusion method.
func fuseMap(vecResults []index.SearchResult, bm25Scores map[string]float64, opts index.SearchOptions) []index.SearchResult {
	// Build a map of chunkID -> vector score
	vecScoreMap := make(map[string]float64)
	for _, r := range vecResults {
		vecScoreMap[r.Chunk.ID] = r.Score
	}

	// Determine fusion method
	var alpha float64
	if opts.BM25Weight > 0 {
		alpha = 1.0 - opts.BM25Weight // alpha is weight for vector (1-BM25Weight)
	} else {
		alpha = 1.0 // Default: pure vector
	}

	// Collect all chunk IDs
	allIDs := make(map[string]bool)
	for id := range vecScoreMap {
		allIDs[id] = true
	}
	for id := range bm25Scores {
		allIDs[id] = true
	}

	// Fuse scores
	type fusedResult struct {
		chunk *core.Chunk
		score float64
	}
	var results []fusedResult

	for id := range allIDs {
		vecScore := vecScoreMap[id]
		bm25Score := bm25Scores[id]

		var fusedScore float64
		if opts.Fusion != nil {
			// Use custom fusion
			fusionInput := []map[string]float64{vecScoreMap, bm25Scores}
			fusedMap := opts.Fusion.Fuse(fusionInput...)
			fusedScore = fusedMap[id]
		} else {
			// Weighted sum: alpha * vecScore + (1-alpha) * bm25Score
			fusedScore = alpha*vecScore + (1-alpha)*bm25Score
		}

		if fusedScore > 0 {
			// Find the chunk
			var chunk *core.Chunk
			for _, r := range vecResults {
				if r.Chunk.ID == id {
					chunk = r.Chunk
					break
				}
			}
			if chunk == nil {
				// Chunk only in BM25 results, skip (we don't have the full chunk)
				continue
			}
			results = append(results, fusedResult{chunk: chunk, score: fusedScore})
		}
	}

	// Convert to SearchResult
	searchResults := make([]index.SearchResult, len(results))
	for i, r := range results {
		searchResults[i] = index.SearchResult{Chunk: r.chunk, Score: r.score}
	}
	return searchResults
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
func (s *MemoryStore) DeleteChunk(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, idx := range s.indexes {
		if err := idx.Delete(context.Background(), id); err == nil {
			return nil
		}
	}
	return core.ErrNotFound
}

// DeleteDocument removes all chunks belonging to a document.
func (s *MemoryStore) DeleteDocument(docID string) error {
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

	for id := range chunkIDs {
		for _, idx := range indexes {
			_ = idx.Delete(context.Background(), id)
		}
	}

	s.mu.Lock()
	delete(s.docChunks, docID)
	s.mu.Unlock()
	return nil
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
