// Package embedder provides the interface for generating text embeddings.
// Users inject their preferred embedding implementation (OpenAI, local model, mock, etc.).
package embedder

import "context"

// Embedder defines the interface for converting text into vector embeddings.
type Embedder interface {
	// Embed converts a single text string into a float32 embedding vector.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch converts multiple text strings into embedding vectors.
	// Implementations may optimize batch processing for better performance.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimension returns the dimension of the embedding vectors produced by this embedder.
	Dimension() int
}

// MockEmbedder is a simple embedder that produces deterministic pseudo-random vectors.
// Useful for testing and development without external dependencies.
type MockEmbedder struct {
	dim int
}

// NewMockEmbedder creates a MockEmbedder with the given dimension.
func NewMockEmbedder(dim int) *MockEmbedder {
	if dim <= 0 {
		dim = 384
	}
	return &MockEmbedder{dim: dim}
}

// Embed generates a deterministic embedding based on the text content.
func (m *MockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vec := make([]float32, m.dim)
	// Simple deterministic hash-based embedding
	hash := uint32(0)
	for _, c := range text {
		hash = hash*31 + uint32(c)
	}
	for i := range vec {
		hash = hash*31 + uint32(i)
		vec[i] = float32(int32(hash>>16))/32768.0
	}
	// Normalize
	normalize(vec)
	return vec, nil
}

// EmbedBatch embeds multiple texts.
func (m *MockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		vec, err := m.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = vec
	}
	return result, nil
}

// Dimension returns the embedding dimension.
func (m *MockEmbedder) Dimension() int {
	return m.dim
}

// normalize normalizes a vector to unit length (L2 norm = 1).
func normalize(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	norm := float32(1.0 / sqrt(float64(sum)))
	for i := range v {
		v[i] *= norm
	}
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (sqrt(normA) * sqrt(normB))
}

// sqrt is a simple square root function for float64.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 20; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
