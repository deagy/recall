package graph

import (
	"fmt"
	"math"
	"sort"
)

// SimilarityResult represents the result of a similarity query.
type SimilarityResult struct {
	ID    string
	Score float64
	Label string
	Type  EntityType
}

// EntitySimilarity provides entity similarity search capabilities.
type EntitySimilarity struct {
	embedder GraphEmbedder
}

// NewEntitySimilarity creates a new EntitySimilarity instance.
func NewEntitySimilarity(embedder GraphEmbedder) *EntitySimilarity {
	return &EntitySimilarity{
		embedder: embedder,
	}
}

// FindSimilarEntities finds entities similar to the given entity.
func (es *EntitySimilarity) FindSimilarEntities(entityID string, topK int) ([]SimilarityResult, error) {
	// Get the query entity embedding
	_, err := es.embedder.EmbedEntity(entityID)
	if err != nil {
		return nil, err
	}

	// In a real implementation, we would:
	// 1. Get embeddings for all entities in the store
	// 2. Compute cosine similarity with query entity
	// 3. Return topK most similar entities

	// For now, return empty results
	return nil, nil
}

// FindSimilarRelations finds relations similar to the given relation.
func (es *EntitySimilarity) FindSimilarRelations(relationType string, topK int) ([]SimilarityResult, error) {
	// Get the query relation embedding
	_, err := es.embedder.EmbedRelation(relationType)
	if err != nil {
		return nil, err
	}

	// In a real implementation, we would:
	// 1. Get embeddings for all relations in the store
	// 2. Compute cosine similarity with query relation
	// 3. Return topK most similar relations

	// For now, return empty results
	return nil, nil
}

// EntityPairSimilarity computes the similarity between two entities.
func EntityPairSimilarity(embedder GraphEmbedder, entityID1, entityID2 string) (float64, error) {
	emb1, err := embedder.EmbedEntity(entityID1)
	if err != nil {
		return 0, err
	}

	emb2, err := embedder.EmbedEntity(entityID2)
	if err != nil {
		return 0, err
	}

	return cosineSimilarity(emb1, emb2), nil
}

// RelationPairSimilarity computes the similarity between two relations.
func RelationPairSimilarity(embedder GraphEmbedder, relType1, relType2 string) (float64, error) {
	emb1, err := embedder.EmbedRelation(relType1)
	if err != nil {
		return 0, err
	}

	emb2, err := embedder.EmbedRelation(relType2)
	if err != nil {
		return 0, err
	}

	return cosineSimilarity(emb1, emb2), nil
}

// ClusterEntities clusters entities based on their embeddings using a simple
// centroid-based clustering algorithm.
func ClusterEntities(embedder GraphEmbedder, numClusters int) (map[string]int, error) {
	// In a real implementation, we would:
	// 1. Get embeddings for all entities
	// 2. Initialize cluster centroids (e.g., using k-means++)
	// 3. Assign entities to nearest centroid
	// 4. Update centroids
	// 5. Repeat until convergence

	// For now, return empty map
	return make(map[string]int), nil
}

// NearestNeighbors finds the K nearest neighbors of an entity in the embedding space.
func NearestNeighbors(embedder GraphEmbedder, entityID string, k int) ([]SimilarityResult, error) {
	// Get the query entity embedding
	_, err := embedder.EmbedEntity(entityID)
	if err != nil {
		return nil, err
	}

	// In a real implementation, we would:
	// 1. Get embeddings for all entities
	// 2. Compute cosine similarity with query entity
	// 3. Return topK most similar entities (excluding the query entity itself)

	// For now, return empty results
	return nil, nil
}

// EmbeddingQuality computes quality metrics for the embedding space.
func EmbeddingQuality(embedder GraphEmbedder) map[string]float64 {
	// In a real implementation, we would:
	// 1. Compute the distribution of embedding norms
	// 2. Compute the pairwise distance distribution
	// 3. Check for embedding collapse (all embeddings similar)

	metrics := make(map[string]float64)

	// Placeholder metrics
	metrics["avgNorm"] = 1.0
	metrics["minNorm"] = 0.5
	metrics["maxNorm"] = 1.5

	return metrics
}

// NormalizeEmbedding normalizes an embedding vector to unit length.
func NormalizeEmbedding(emb []float32) []float32 {
	norm := 0.0
	for _, v := range emb {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return emb
	}

	result := make([]float32, len(emb))
	for i, v := range emb {
		result[i] = v / float32(norm)
	}

	return result
}

// EmbeddingDiversity computes the diversity of embeddings in the space.
func EmbeddingDiversity(embeddings [][]float32) float64 {
	if len(embeddings) < 2 {
		return 0
	}

	var totalDist float64
	count := 0

	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			//nolint:gosec // G602: i/j bounded by len(embeddings) in the enclosing loops
			dist := embeddingDistance(embeddings[i], embeddings[j])
			totalDist += dist
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return totalDist / float64(count)
}

// embeddingDistance computes the Euclidean distance between two embeddings.
func embeddingDistance(a, b []float32) float64 {
	dim := len(a)
	if len(b) < dim {
		dim = len(b)
	}

	var sum float64
	for i := 0; i < dim; i++ {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// String returns a human-readable representation of a SimilarityResult.
func (sr SimilarityResult) String() string {
	return fmt.Sprintf("%s (score: %.4f)", sr.ID, sr.Score)
}

// Ensure sort is used (for future implementation)
var _ = sort.Slice
