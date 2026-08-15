package embedder

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- MockEmbedder tests using testify ---

func TestNewMockEmbedder_DefaultDimension(t *testing.T) {
	e := NewMockEmbedder(0)
	assert.Equal(t, 384, e.Dimension(), "default dimension should be 384")
}

func TestNewMockEmbedder_ValidDimension(t *testing.T) {
	e := NewMockEmbedder(1536)
	assert.Equal(t, 1536, e.Dimension(), "dimension should match requested value")
}

func TestMockEmbedder_Embed_ReturnsCorrectDimension(t *testing.T) {
	e := NewMockEmbedder(128)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "hello world")
	require.NoError(t, err)
	assert.Len(t, vec, 128, "embedding should have 128 dimensions")
}

func TestMockEmbedder_Embed_Deterministic(t *testing.T) {
	e := NewMockEmbedder(64)
	ctx := context.Background()

	vec1, err := e.Embed(ctx, "test text")
	require.NoError(t, err)
	vec2, err := e.Embed(ctx, "test text")
	require.NoError(t, err)

	assert.Equal(t, vec1, vec2, "embeddings should be deterministic for same input")
}

func TestMockEmbedder_Embed_DifferentInputsDifferentOutputs(t *testing.T) {
	e := NewMockEmbedder(64)
	ctx := context.Background()

	vec1, err := e.Embed(ctx, "hello")
	require.NoError(t, err)
	vec2, err := e.Embed(ctx, "world")
	require.NoError(t, err)

	sim := CosineSimilarity(vec1, vec2)
	// Different texts should not have perfect similarity
	assert.Less(t, sim, 0.99, "different texts should not have near-perfect similarity")
}

func TestMockEmbedder_Embed_Normalized(t *testing.T) {
	e := NewMockEmbedder(64)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "test")
	require.NoError(t, err)

	// Check unit length
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	assert.InDelta(t, 1.0, norm, 0.01, "embedding should be a unit vector")
}

func TestMockEmbedder_EmbedBatch(t *testing.T) {
	e := NewMockEmbedder(32)
	ctx := context.Background()

	texts := []string{"hello", "world", "test"}
	results, err := e.EmbedBatch(ctx, texts)
	require.NoError(t, err)
	assert.Len(t, results, 3, "should return 3 results")
	for i, vec := range results {
		assert.Len(t, vec, 32, "result %d should have 32 dimensions", i)
	}
}

func TestMockEmbedder_EmbedBatch_Empty(t *testing.T) {
	e := NewMockEmbedder(32)
	ctx := context.Background()

	results, err := e.EmbedBatch(ctx, []string{})
	require.NoError(t, err)
	assert.Empty(t, results, "empty input should produce empty output")
}

// --- CosineSimilarity tests ---

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, 1.0, sim, 0.001, "identical vectors should have similarity 1.0")
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{-1, 0, 0}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, -1.0, sim, 0.001, "opposite vectors should have similarity -1.0")
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, 0.0, sim, 0.001, "orthogonal vectors should have similarity ~0")
}

func TestCosineSimilarity_DifferentDimensions(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	assert.Equal(t, float64(0), sim, "different dimensions should return similarity 0")
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	sim := CosineSimilarity(a, b)
	assert.Equal(t, float64(0), sim, "zero vector should return similarity 0")
}

// --- sqrt tests ---

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
		t.Run("", func(t *testing.T) {
			result := math.Sqrt(tt.input)
			assert.InDelta(t, tt.expected, result, 0.001, "sqrt(%f) should be %f", tt.input, tt.expected)
		})
	}
}

func TestSqrt_Negative(t *testing.T) {
	result := math.Sqrt(-1)
	assert.True(t, math.IsNaN(result), "sqrt(-1) should return NaN")
}

// --- MockEmbedder-based tests (using existing MockEmbedder struct) ---

func TestMockEmbedder_Dimension(t *testing.T) {
	e := NewMockEmbedder(512)
	assert.Equal(t, 512, e.Dimension(), "dimension should match")
}

func TestMockEmbedder_Embed(t *testing.T) {
	e := NewMockEmbedder(3)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "test")
	require.NoError(t, err)
	assert.Len(t, vec, 3, "embedding should have 3 dimensions")
}

func TestMockEmbedder_EmbedError(t *testing.T) {
	// MockEmbedder never returns errors; verify it works for valid input
	e := NewMockEmbedder(3)
	ctx := context.Background()

	_, err := e.Embed(ctx, "test")
	assert.NoError(t, err, "MockEmbedder should not return errors")
}

func TestMockEmbedder_EmbedBatch_WithMock(t *testing.T) {
	// Using the existing MockEmbedder to verify batch embedding works correctly
	e := NewMockEmbedder(32)
	ctx := context.Background()

	texts := []string{"a", "b"}
	results, err := e.EmbedBatch(ctx, texts)
	require.NoError(t, err)
	assert.Len(t, results, 2, "should return 2 results")
	for i, vec := range results {
		assert.Len(t, vec, 32, "result %d should have 32 dimensions", i)
	}
}
