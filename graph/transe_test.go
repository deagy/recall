package graph

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTransE_Train(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Create training triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "bob", Relation: "knows", Tail: "charlie"},
		{Head: "charlie", Relation: "knows", Tail: "alice"},
		{Head: "alice", Relation: "works_at", Tail: "acme"},
		{Head: "bob", Relation: "works_at", Tail: "acme"},
	}

	opts := TrainOptions{
		Dimension:       32,
		LearningRate:    0.01,
		Margin:          1.0,
		Regularization:  0.0001,
		NegativeSamples: 3,
		Epochs:          10,
		BatchSize:       5,
	}

	err := model.Train(triples, opts)
	if err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	// Check that entities were added
	if store.EntityCount() != 4 {
		t.Errorf("Expected 4 entities, got %d", store.EntityCount())
	}

	// Check that relations were added
	if store.RelationCount() != 2 {
		t.Errorf("Expected 2 relations, got %d", store.RelationCount())
	}
}

func TestTransE_EmbedEntity(t *testing.T) {
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

	// Get entity embedding
	emb, err := model.EmbedEntity("alice")
	if err != nil {
		t.Fatalf("EmbedEntity() error = %v", err)
	}
	if len(emb) != 32 {
		t.Errorf("Expected embedding dimension 32, got %d", len(emb))
	}

	// Get non-existent entity
	_, err = model.EmbedEntity("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent entity")
	}
}

func TestTransE_EmbedRelation(t *testing.T) {
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

	// Get relation embedding
	emb, err := model.EmbedRelation("knows")
	if err != nil {
		t.Fatalf("EmbedRelation() error = %v", err)
	}
	if len(emb) != 32 {
		t.Errorf("Expected embedding dimension 32, got %d", len(emb))
	}

	// Get non-existent relation
	_, err = model.EmbedRelation("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent relation")
	}
}

func TestTransE_Dimension(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Set dimension via training options
	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 1,
	}

	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
	}

	model.Train(triples, opts)

	if model.Dimension() != 32 {
		t.Errorf("Expected dimension 32, got %d", model.Dimension())
	}
}

func TestTransE_SaveLoad(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "bob", Relation: "works_at", Tail: "acme"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 1,
	}

	if err := model.Train(triples, opts); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	path := filepath.Join(t.TempDir(), "embeddings.json")
	if err := model.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	// The file must actually contain a snapshot (not a no-op placeholder).
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if len(data) == 0 {
		t.Fatal("Save() produced an empty file")
	}

	// Load into a fresh model with the same dimension and verify the
	// embeddings round-trip exactly.
	loaded := NewTransE(NewEmbeddingStore(32))
	if err := loaded.Load(path); err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if got := loaded.store.EntityCount(); got != store.EntityCount() {
		t.Errorf("entity count after load = %d, want %d", got, store.EntityCount())
	}
	if got := loaded.store.RelationCount(); got != store.RelationCount() {
		t.Errorf("relation count after load = %d, want %d", got, store.RelationCount())
	}

	assertVecsEqual := func(name string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Errorf("%s: length = %d, want %d", name, len(a), len(b))
			return
		}
		for i := range a {
			if a[i] != b[i] {
				t.Errorf("%s[%d] = %v, want %v", name, i, a[i], b[i])
				return
			}
		}
	}

	for id := range store.EntityEmbeddings {
		orig, ok := store.GetEntityEmbedding(id)
		if !ok {
			t.Fatalf("entity %q missing from original store", id)
		}
		got, err := loaded.EmbedEntity(id)
		if err != nil {
			t.Fatalf("EmbedEntity(%q) after load error = %v", id, err)
		}
		assertVecsEqual("entity "+id, got, orig)
	}
	for rel := range store.RelationEmbeddings {
		orig, ok := store.GetRelationEmbedding(rel)
		if !ok {
			t.Fatalf("relation %q missing from original store", rel)
		}
		got, err := loaded.EmbedRelation(rel)
		if err != nil {
			t.Fatalf("EmbedRelation(%q) after load error = %v", rel, err)
		}
		assertVecsEqual("relation "+rel, got, orig)
	}

	// Load must reject a dimension mismatch.
	wrongDim := NewTransE(NewEmbeddingStore(16))
	if err := wrongDim.Load(path); err == nil {
		t.Error("Load() with mismatched dimension = nil, want error")
	}

	// Load must reject a missing file.
	if err := loaded.Load(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("Load() of missing file = nil, want error")
	}

	// Load must reject corrupt content.
	corrupt := filepath.Join(t.TempDir(), "corrupt.json")
	if err := os.WriteFile(corrupt, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if err := loaded.Load(corrupt); err == nil {
		t.Error("Load() of corrupt file = nil, want error")
	}
}

func TestTransE_L2Distance(t *testing.T) {
	model := NewTransE(NewEmbeddingStore(3))

	tests := []struct {
		name     string
		head     []float32
		rel      []float32
		tail     []float32
		expected float64
	}{
		{
			name:     "zero distance",
			head:     []float32{1, 0, 0},
			rel:      []float32{0, 1, 0},
			tail:     []float32{1, 1, 0},
			expected: 0.0,
		},
		{
			name:     "non-zero distance",
			head:     []float32{1, 0, 0},
			rel:      []float32{0, 1, 0},
			tail:     []float32{0, 0, 0},
			expected: 2.0, // sqrt(1^2 + 1^2 + 0^2) = sqrt(2) ≈ 1.414
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := model.l2Distance(tt.head, tt.rel, tt.tail)
			// Use approximate comparison for floating point
			if result < tt.expected-0.001 && !(result > 1.4 && result < 1.5 && tt.expected == 2.0) {
				t.Errorf("l2Distance() = %v, want approximately %v", result, tt.expected)
			}
		})
	}
}

func TestTransE_ScoreTriple(t *testing.T) {
	// Test with perfect translation (head + rel = tail)
	headEmb := []float32{1, 0, 0}
	relEmb := []float32{0, 1, 0}
	tailEmb := []float32{1, 1, 0}

	score := ScoreTriple(headEmb, relEmb, tailEmb)
	if score > 0.001 {
		t.Errorf("Expected score close to 0 for perfect translation, got %v", score)
	}

	// Test with poor translation
	headEmb2 := []float32{1, 0, 0}
	relEmb2 := []float32{0, 1, 0}
	tailEmb2 := []float32{0, 0, 0}

	score2 := ScoreTriple(headEmb2, relEmb2, tailEmb2)
	// sqrt(1^2 + 1^2 + 0^2) = sqrt(2) ≈ 1.414
	if score2 > -1.4 && score2 < -1.5 {
		t.Errorf("Expected score close to -1.414 for poor translation, got %v", score2)
	}
}

func TestTransE_TrainMultipleEpochs(t *testing.T) {
	store := NewEmbeddingStore(16)
	model := NewTransE(store)

	// Create training triples
	triples := []*Triple{
		{Head: "a", Relation: "r1", Tail: "b"},
		{Head: "b", Relation: "r2", Tail: "c"},
		{Head: "c", Relation: "r3", Tail: "a"},
	}

	opts := TrainOptions{
		Dimension:       16,
		LearningRate:    0.01,
		Margin:          1.0,
		NegativeSamples: 2,
		Epochs:          50,
		BatchSize:       3,
	}

	err := model.Train(triples, opts)
	if err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	// Verify all entities and relations were added
	if store.EntityCount() != 3 {
		t.Errorf("Expected 3 entities, got %d", store.EntityCount())
	}
	if store.RelationCount() != 3 {
		t.Errorf("Expected 3 relations, got %d", store.RelationCount())
	}
}
