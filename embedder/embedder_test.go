package embedder

import (
	"context"
	"math"
	"testing"
)

func TestNewMockEmbedder_DefaultDimension(t *testing.T) {
	e := NewMockEmbedder(0)
	if e.Dimension() != 384 {
		t.Fatalf("expected default dimension 384, got %d", e.Dimension())
	}
}

func TestNewMockEmbedder_ValidDimension(t *testing.T) {
	e := NewMockEmbedder(1536)
	if e.Dimension() != 1536 {
		t.Fatalf("expected dimension 1536, got %d", e.Dimension())
	}
}

func TestMockEmbedder_Embed_ReturnsCorrectDimension(t *testing.T) {
	e := NewMockEmbedder(128)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != 128 {
		t.Fatalf("expected 128 dimensions, got %d", len(vec))
	}
}

func TestMockEmbedder_Embed_Deterministic(t *testing.T) {
	e := NewMockEmbedder(64)
	ctx := context.Background()

	vec1, _ := e.Embed(ctx, "test text")
	vec2, _ := e.Embed(ctx, "test text")

	for i := range vec1 {
		if vec1[i] != vec2[i] {
			t.Fatal("embeddings should be deterministic for same input")
		}
	}
}

func TestMockEmbedder_Embed_DifferentInputsDifferentOutputs(t *testing.T) {
	e := NewMockEmbedder(64)
	ctx := context.Background()

	vec1, _ := e.Embed(ctx, "hello")
	vec2, _ := e.Embed(ctx, "world")

	sim := CosineSimilarity(vec1, vec2)
	// Different texts should not have perfect similarity
	if sim > 0.99 {
		t.Fatalf("different texts should not have near-perfect similarity, got %f", sim)
	}
}

func TestMockEmbedder_Embed_Normalized(t *testing.T) {
	e := NewMockEmbedder(64)
	ctx := context.Background()

	vec, _ := e.Embed(ctx, "test")

	// Check unit length
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	if math.Abs(norm-1.0) > 0.01 {
		t.Fatalf("expected unit vector, got norm %f", norm)
	}
}

func TestMockEmbedder_EmbedBatch(t *testing.T) {
	e := NewMockEmbedder(32)
	ctx := context.Background()

	texts := []string{"hello", "world", "test"}
	results, err := e.EmbedBatch(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	for i, vec := range results {
		if len(vec) != 32 {
			t.Fatalf("result %d: expected 32 dimensions, got %d", i, len(vec))
		}
	}
}

func TestMockEmbedder_EmbedBatch_Empty(t *testing.T) {
	e := NewMockEmbedder(32)
	ctx := context.Background()

	results, err := e.EmbedBatch(ctx, []string{})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	if math.Abs(sim-1.0) > 0.001 {
		t.Fatalf("expected similarity 1.0, got %f", sim)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{-1, 0, 0}
	sim := CosineSimilarity(a, b)
	if math.Abs(sim-(-1.0)) > 0.001 {
		t.Fatalf("expected similarity -1.0, got %f", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	sim := CosineSimilarity(a, b)
	if math.Abs(sim) > 0.001 {
		t.Fatalf("expected similarity ~0, got %f", sim)
	}
}

func TestCosineSimilarity_DifferentDimensions(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	if sim != 0 {
		t.Fatalf("expected similarity 0 for different dimensions, got %f", sim)
	}
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	sim := CosineSimilarity(a, b)
	if sim != 0 {
		t.Fatalf("expected similarity 0 for zero vector, got %f", sim)
	}
}

func TestSqrt(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0, 0},
		{1, 1},
		{4, 2},
		{9, 3},
		{0.25, 0.5},
	}
	for _, tt := range tests {
		result := sqrt(tt.input)
		if math.Abs(result-tt.expected) > 0.001 {
			t.Fatalf("sqrt(%f) = %f, expected %f", tt.input, result, tt.expected)
		}
	}
}

func TestSqrt_Negative(t *testing.T) {
	result := sqrt(-1)
	if result != 0 {
		t.Fatalf("sqrt(-1) = %f, expected 0", result)
	}
}