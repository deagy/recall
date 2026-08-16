package graph

import (
	"testing"
)

func TestEmbeddingStore_AddEntity(t *testing.T) {
	store := NewEmbeddingStore(64)

	// Add first entity
	store.AddEntity("entity1")
	if store.EntityCount() != 1 {
		t.Errorf("Expected 1 entity, got %d", store.EntityCount())
	}

	// Add same entity again (should not duplicate)
	store.AddEntity("entity1")
	if store.EntityCount() != 1 {
		t.Errorf("Expected 1 entity after duplicate add, got %d", store.EntityCount())
	}

	// Add second entity
	store.AddEntity("entity2")
	if store.EntityCount() != 2 {
		t.Errorf("Expected 2 entities, got %d", store.EntityCount())
	}
}

func TestEmbeddingStore_AddRelation(t *testing.T) {
	store := NewEmbeddingStore(64)

	// Add first relation
	store.AddRelation("rel1")
	if store.RelationCount() != 1 {
		t.Errorf("Expected 1 relation, got %d", store.RelationCount())
	}

	// Add same relation again (should not duplicate)
	store.AddRelation("rel1")
	if store.RelationCount() != 1 {
		t.Errorf("Expected 1 relation after duplicate add, got %d", store.RelationCount())
	}

	// Add second relation
	store.AddRelation("rel2")
	if store.RelationCount() != 2 {
		t.Errorf("Expected 2 relations, got %d", store.RelationCount())
	}
}

func TestEmbeddingStore_GetEmbeddings(t *testing.T) {
	store := NewEmbeddingStore(64)

	// Add entity and relation
	store.AddEntity("entity1")
	store.AddRelation("rel1")

	// Get entity embedding
	emb, ok := store.GetEntityEmbedding("entity1")
	if !ok {
		t.Fatal("Expected to find entity embedding")
	}
	if len(emb) != 64 {
		t.Errorf("Expected embedding dimension 64, got %d", len(emb))
	}

	// Get non-existent entity
	_, ok = store.GetEntityEmbedding("nonexistent")
	if ok {
		t.Error("Expected not to find non-existent entity")
	}

	// Get relation embedding
	emb, ok = store.GetRelationEmbedding("rel1")
	if !ok {
		t.Fatal("Expected to find relation embedding")
	}
	if len(emb) != 64 {
		t.Errorf("Expected embedding dimension 64, got %d", len(emb))
	}

	// Get non-existent relation
	_, ok = store.GetRelationEmbedding("nonexistent")
	if ok {
		t.Error("Expected not to find non-existent relation")
	}
}

func TestCosineSimilarity(t *testing.T) {
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
			expected: 1.0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 0.0,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 2},
			b:        []float32{-1, -2},
			expected: -1.0,
		},
		{
			name:     "different dimensions",
			a:        []float32{1, 2},
			b:        []float32{1, 2, 3},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			if result < tt.expected-0.001 || result > tt.expected+0.001 {
				t.Errorf("cosineSimilarity() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestNormalizeEmbedding(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []float32
	}{
		{
			name:     "unit vector",
			input:    []float32{0, 1},
			expected: []float32{0, 1},
		},
		{
			name:     "normalize vector",
			input:    []float32{3, 4},
			expected: []float32{0.6, 0.8},
		},
		{
			name:     "zero vector",
			input:    []float32{0, 0},
			expected: []float32{0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizeEmbedding(tt.input)
			for i := range result {
				if result[i] < tt.expected[i]-0.001 || result[i] > tt.expected[i]+0.001 {
					t.Errorf("NormalizeEmbedding()[%d] = %v, want %v", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestEmbeddingDiversity(t *testing.T) {
	// Test with identical embeddings (zero diversity)
	embeddings1 := [][]float32{
		{1, 2, 3},
		{1, 2, 3},
		{1, 2, 3},
	}
	diversity1 := EmbeddingDiversity(embeddings1)
	if diversity1 != 0 {
		t.Errorf("Expected diversity 0 for identical embeddings, got %v", diversity1)
	}

	// Test with diverse embeddings
	embeddings2 := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	diversity2 := EmbeddingDiversity(embeddings2)
	if diversity2 <= 0 {
		t.Errorf("Expected positive diversity for diverse embeddings, got %v", diversity2)
	}

	// Test with single embedding
	embeddings3 := [][]float32{
		{1, 2, 3},
	}
	diversity3 := EmbeddingDiversity(embeddings3)
	if diversity3 != 0 {
		t.Errorf("Expected diversity 0 for single embedding, got %v", diversity3)
	}

	// Test with empty embeddings
	embeddings4 := [][]float32{}
	diversity4 := EmbeddingDiversity(embeddings4)
	if diversity4 != 0 {
		t.Errorf("Expected diversity 0 for empty embeddings, got %v", diversity4)
	}
}

func TestDefaultTrainOptions(t *testing.T) {
	opts := DefaultTrainOptions()

	if opts.Dimension != 64 {
		t.Errorf("Expected dimension 64, got %d", opts.Dimension)
	}
	if opts.LearningRate != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %v", opts.LearningRate)
	}
	if opts.Margin != 1.0 {
		t.Errorf("Expected margin 1.0, got %v", opts.Margin)
	}
	if opts.Regularization != 0.0001 {
		t.Errorf("Expected regularization 0.0001, got %v", opts.Regularization)
	}
	if opts.NegativeSamples != 5 {
		t.Errorf("Expected negative samples 5, got %d", opts.NegativeSamples)
	}
	if opts.Epochs != 100 {
		t.Errorf("Expected epochs 100, got %d", opts.Epochs)
	}
	if opts.BatchSize != 32 {
		t.Errorf("Expected batch size 32, got %d", opts.BatchSize)
	}
}
