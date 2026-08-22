package distributed

import (
	"context"
	"fmt"
	"math"
	"strings"
	"unicode"

	"github.com/deagy/recall/bm25"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

// ShardIndex provides vector similarity and hybrid (vector + BM25) search
// over a snapshot of a shard's chunk map.
//
// The snapshot is an independent copy taken under the shard's read lock at
// construction time (see NewShardIndex), so ShardIndex methods are safe to
// call concurrently with writes to the live shard and never require holding
// the shard's lock. Results reflect the shard's state at snapshot time, not
// its current state.
type ShardIndex struct {
	data map[string]*core.Chunk
	ns   string
}

// NewShardIndex creates a new ShardIndex from a snapshot of the shard's
// current chunk map. A nil shard yields an empty index.
func NewShardIndex(shard *Shard) *ShardIndex {
	if shard == nil {
		return &ShardIndex{}
	}
	shard.mu.RLock()
	snapshot := make(map[string]*core.Chunk, len(shard.Data))
	for id, chunk := range shard.Data {
		snapshot[id] = chunk
	}
	ns := shard.ID
	shard.mu.RUnlock()
	return &ShardIndex{data: snapshot, ns: ns}
}

// matchesFilters reports whether chunk satisfies every metadata filter.
func matchesFilters(chunk *core.Chunk, filters []index.Filter) bool {
	for _, filter := range filters {
		if !filter.Match(chunk) {
			return false
		}
	}
	return true
}

// Search performs vector similarity search over the shard snapshot.
func (si *ShardIndex) Search(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(si.data) == 0 {
		return []index.SearchResult{}, nil
	}

	var results []index.SearchResult

	for _, chunk := range si.data {
		if chunk.Embedding == nil {
			continue
		}

		score := cosineSimilarity(query, chunk.Embedding)

		if !matchesFilters(chunk, opts.Filters) {
			continue
		}

		if score < opts.MinScore {
			continue
		}

		results = append(results, index.SearchResult{
			Chunk: chunk,
			Score: score,
		})
	}

	// Sort by score (descending) with a deterministic tie-break.
	sortSearchResults(results)

	// Limit results
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}

	return results, nil
}

// SearchHybrid performs hybrid search over the shard snapshot, combining
// vector similarity (queryEmb against each chunk's stored embedding) with
// BM25 keyword scores (query against each chunk's content).
//
// The combination mirrors index.HybridIndex: a custom opts.Fusion when set,
// otherwise a weighted sum over opts.BM25Weight (0.5/0.5 when unset).
// Results with a non-positive fused score are dropped, MinScore applies to
// the fused score, and at most TopK results are returned. Because the vector
// and keyword scores are independent, chunks with no vector match can still
// surface on a strong keyword hit (and vice versa).
//
// The BM25 index is built per call over the snapshot — O(n) per query, which
// is acceptable for in-process shards; very large shards at high query rates
// should use an incrementally maintained keyword index (see ROADMAP).
func (si *ShardIndex) SearchHybrid(ctx context.Context, query string, queryEmb []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(si.data) == 0 {
		return []index.SearchResult{}, nil
	}

	bm25Indexer := bm25.New(bm25.DefaultConfig())
	for _, chunk := range si.data {
		bm25Indexer.AddDocument(chunk.ID, chunk.Content)
	}
	kwScores := make(map[string]float64, len(si.data))
	for _, r := range bm25Indexer.Search(query) {
		kwScores[r.DocID] = r.Score
	}

	vecScores := make(map[string]float64, len(si.data))
	for id, chunk := range si.data {
		if chunk.Embedding != nil {
			vecScores[id] = cosineSimilarity(queryEmb, chunk.Embedding)
		}
	}

	var fused map[string]float64
	if opts.Fusion != nil {
		fused = opts.Fusion.Fuse(vecScores, kwScores)
	}

	var results []index.SearchResult
	for id, chunk := range si.data {
		var score float64
		if fused != nil {
			score = fused[id]
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
		if !matchesFilters(chunk, opts.Filters) {
			continue
		}
		results = append(results, index.SearchResult{
			Chunk: chunk,
			Score: score,
		})
	}

	// Sort by score (descending) with a deterministic tie-break.
	sortSearchResults(results)

	// Limit results
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}

	return results, nil
}

// generateQueryEmbedding creates a lexical query embedding based on hashed
// word frequencies. It is a fallback used only when a DistributedStore has
// no embedder configured; its vectors are not semantically meaningful and
// generally do not align with embeddings produced by a real embedding model.
func generateQueryEmbedding(query string) []float32 {
	dimension := 128
	embedding := make([]float32, dimension)

	// Simple word hashing to distribute tokens across dimensions
	words := strings.FieldsFunc(query, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})

	for _, word := range words {
		word = strings.ToLower(word)
		// Hash the word to distribute across dimensions
		hash := simpleHash(word)
		dim := int(hash) % dimension
		embedding[dim] += 1.0
	}

	// Normalize
	norm := 0.0
	for _, v := range embedding {
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range embedding {
			embedding[i] /= float32(norm)
		}
	}

	return embedding
}

// simpleHash is a simple hash function for distributing tokens across dimensions.
func simpleHash(s string) uint64 {
	h := uint64(0)
	for _, c := range s {
		h = h*31 + uint64(c)
	}
	return h
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

// Compile-time check to ensure ShardIndex implements the Index interface.
var _ index.Index = (*ShardIndex)(nil)

// Add is not implemented for ShardIndex (read-only).
func (si *ShardIndex) Add(ctx context.Context, chunk *core.Chunk) error {
	return fmt.Errorf("ShardIndex is read-only")
}

// AddBatch is not implemented for ShardIndex (read-only).
func (si *ShardIndex) AddBatch(ctx context.Context, chunks []*core.Chunk) error {
	return fmt.Errorf("ShardIndex is read-only")
}

// Delete is not implemented for ShardIndex (read-only).
func (si *ShardIndex) Delete(ctx context.Context, id string) error {
	return fmt.Errorf("ShardIndex is read-only")
}

// Count returns the number of chunks in the snapshot.
func (si *ShardIndex) Count() int {
	return len(si.data)
}

// Dimension returns the embedding dimension (0 if no chunks have embeddings).
func (si *ShardIndex) Dimension() int {
	for _, chunk := range si.data {
		if chunk.Embedding != nil {
			return len(chunk.Embedding)
		}
	}
	return 0
}

// Namespace returns the shard ID the snapshot was taken from.
func (si *ShardIndex) Namespace() string {
	return si.ns
}
