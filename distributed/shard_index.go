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

// ShardIndex provides vector similarity search within a single shard.
type ShardIndex struct {
	shard *Shard
}

// NewShardIndex creates a new ShardIndex from a shard.
func NewShardIndex(shard *Shard) *ShardIndex {
	return &ShardIndex{shard: shard}
}

// Search performs vector similarity search within the shard.
func (si *ShardIndex) Search(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	if si.shard == nil || len(si.shard.Data) == 0 {
		return []index.SearchResult{}, nil
	}

	var results []index.SearchResult

	for _, chunk := range si.shard.Data {
		if chunk.Embedding == nil {
			continue
		}

		score := cosineSimilarity(query, chunk.Embedding)

		// Apply filters
		if len(opts.Filters) > 0 {
			matchesAll := true
			for _, filter := range opts.Filters {
				if !filter.Match(chunk) {
					matchesAll = false
					break
				}
			}
			if !matchesAll {
				continue
			}
		}

		if score < opts.MinScore {
			continue
		}

		results = append(results, index.SearchResult{
			Chunk: chunk,
			Score: score,
		})
	}

	// Sort by score (descending)
	sortResultsByScore(results)

	// Limit results
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}

	return results, nil
}

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
func (si *ShardIndex) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if si.shard == nil || len(si.shard.Data) == 0 {
		return []index.SearchResult{}, nil
	}

	// Generate query embedding (simplified - would use actual embedder in production)
	queryEmbedding := generateQueryEmbedding(query)

	// Create BM25 indexer for the shard
	bm25Indexer := bm25.New(bm25.DefaultConfig())
	for _, chunk := range si.shard.Data {
		bm25Indexer.AddDocument(chunk.ID, chunk.Content)
	}

	// Query BM25
	bm25Results := bm25Indexer.Search(query)
	bm25Scores := make(map[string]float64)
	for _, r := range bm25Results {
		bm25Scores[r.DocID] = r.Score
	}

	var results []index.SearchResult

	for _, chunk := range si.shard.Data {
		if chunk.Embedding == nil {
			continue
		}

		// Vector similarity score
		vectorScore := cosineSimilarity(queryEmbedding, chunk.Embedding)

		// BM25 score
		bm25Score := 0.0
		if score, exists := bm25Scores[chunk.ID]; exists {
			bm25Score = score
		}

		// Combine scores (default 50/50)
		bm25Weight := opts.BM25Weight
		if bm25Weight == 0 {
			bm25Weight = 0.5
		}

		combinedScore := (1-bm25Weight)*vectorScore + bm25Weight*bm25Score

		// Apply filters
		if len(opts.Filters) > 0 {
			matchesAll := true
			for _, filter := range opts.Filters {
				if !filter.Match(chunk) {
					matchesAll = false
					break
				}
			}
			if !matchesAll {
				continue
			}
		}

		if combinedScore < opts.MinScore {
			continue
		}

		results = append(results, index.SearchResult{
			Chunk: chunk,
			Score: combinedScore,
		})
	}

	// Sort by score (descending)
	sortResultsByScore(results)

	// Limit results
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}

	return results, nil
}

// generateQueryEmbedding creates a simple query embedding based on word frequencies.
// In production, this would use an actual embedding model.
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

// sortResultsByScore sorts results by score in descending order.
func sortResultsByScore(results []index.SearchResult) {
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
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

// Count returns the number of chunks in the shard.
func (si *ShardIndex) Count() int {
	if si.shard == nil {
		return 0
	}
	return len(si.shard.Data)
}

// Dimension returns the embedding dimension (0 if no chunks have embeddings).
func (si *ShardIndex) Dimension() int {
	if si.shard == nil {
		return 0
	}
	for _, chunk := range si.shard.Data {
		if chunk.Embedding != nil {
			return len(chunk.Embedding)
		}
	}
	return 0
}

// Namespace returns the shard ID as the namespace.
func (si *ShardIndex) Namespace() string {
	if si.shard == nil {
		return ""
	}
	return si.shard.ID
}
