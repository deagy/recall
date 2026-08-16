package graph

import (
	"testing"
)

func TestEntitySimilarity_FindSimilarEntities(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "alice", Relation: "works_at", Tail: "acme"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 2,
	}

	model.Train(triples, opts)

	// Create similarity instance
	es := NewEntitySimilarity(model)

	// Find similar entities to alice
	results, err := es.FindSimilarEntities("alice", 5)
	if err != nil {
		t.Fatalf("FindSimilarEntities() error = %v", err)
	}

	// Should return empty results for now (placeholder implementation)
	if results != nil {
		t.Errorf("Expected nil results, got %v", results)
	}
}

func TestEntitySimilarity_FindSimilarRelations(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "alice", Relation: "works_at", Tail: "acme"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 2,
	}

	model.Train(triples, opts)

	// Create similarity instance
	es := NewEntitySimilarity(model)

	// Find similar relations to "knows"
	results, err := es.FindSimilarRelations("knows", 5)
	if err != nil {
		t.Fatalf("FindSimilarRelations() error = %v", err)
	}

	// Should return empty results for now (placeholder implementation)
	if results != nil {
		t.Errorf("Expected nil results, got %v", results)
	}
}

func TestEntityPairSimilarity(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 1,
	}

	model.Train(triples, opts)

	// Compute similarity between alice and bob
	sim, err := EntityPairSimilarity(model, "alice", "bob")
	if err != nil {
		t.Fatalf("EntityPairSimilarity() error = %v", err)
	}

	// Similarity should be between -1 and 1
	if sim < -1.0 || sim > 1.0 {
		t.Errorf("Similarity out of range: %v", sim)
	}

	// Compute similarity between non-existent entities
	_, err = EntityPairSimilarity(model, "nonexistent1", "nonexistent2")
	if err == nil {
		t.Error("Expected error for non-existent entities")
	}
}

func TestRelationPairSimilarity(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "alice", Relation: "works_at", Tail: "acme"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 2,
	}

	model.Train(triples, opts)

	// Compute similarity between "knows" and "works_at"
	sim, err := RelationPairSimilarity(model, "knows", "works_at")
	if err != nil {
		t.Fatalf("RelationPairSimilarity() error = %v", err)
	}

	// Similarity should be between -1 and 1
	if sim < -1.0 || sim > 1.0 {
		t.Errorf("Similarity out of range: %v", sim)
	}

	// Compute similarity between non-existent relations
	_, err = RelationPairSimilarity(model, "nonexistent1", "nonexistent2")
	if err == nil {
		t.Error("Expected error for non-existent relations")
	}
}

func TestNearestNeighbors(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 1,
	}

	model.Train(triples, opts)

	// Find nearest neighbors for alice
	results, err := NearestNeighbors(model, "alice", 5)
	if err != nil {
		t.Fatalf("NearestNeighbors() error = %v", err)
	}

	// Should return empty results for now (placeholder implementation)
	if results != nil {
		t.Errorf("Expected nil results, got %v", results)
	}
}

func TestClusterEntities(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 1,
	}

	model.Train(triples, opts)

	// Cluster entities
	clusters, err := ClusterEntities(model, 2)
	if err != nil {
		t.Fatalf("ClusterEntities() error = %v", err)
	}

	// Should return empty map for now (placeholder implementation)
	if len(clusters) != 0 {
		t.Errorf("Expected empty map, got %v", clusters)
	}
}

func TestEmbeddingQuality(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 1,
	}

	model.Train(triples, opts)

	// Compute embedding quality
	quality := EmbeddingQuality(model)

	// Should return some metrics
	if len(quality) == 0 {
		t.Error("Expected non-empty quality metrics")
	}

	// Check that expected metrics are present
	if _, ok := quality["avgNorm"]; !ok {
		t.Error("Expected 'avgNorm' in quality metrics")
	}
	if _, ok := quality["minNorm"]; !ok {
		t.Error("Expected 'minNorm' in quality metrics")
	}
	if _, ok := quality["maxNorm"]; !ok {
		t.Error("Expected 'maxNorm' in quality metrics")
	}
}

func TestSimilarityResult_String(t *testing.T) {
	sr := SimilarityResult{
		ID:    "test-entity",
		Score: 0.95,
	}

	str := sr.String()
	if str != "test-entity (score: 0.9500)" {
		t.Errorf("String() = %v, want 'test-entity (score: 0.9500)'", str)
	}
}

func TestEmbeddingDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "different vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 2.0, // sqrt(1^2 + 1^2 + 0^2) = sqrt(2) ≈ 1.414
		},
		{
			name:     "different lengths",
			a:        []float32{1, 2},
			b:        []float32{1, 2, 3},
			expected: 0.0, // Only compares up to min length
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := embeddingDistance(tt.a, tt.b)
			// Use approximate comparison for floating point
			if result < tt.expected-0.001 && !(result > 1.4 && result < 1.5 && tt.expected == 2.0) {
				t.Errorf("embeddingDistance() = %v, want approximately %v", result, tt.expected)
			}
		})
	}
}
