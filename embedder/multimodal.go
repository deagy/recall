package embedder

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
)

// MultiModalEmbedder embeds heterogeneous content (text and images)
// into a single shared vector space, enabling cross-modal retrieval:
// text queries can match stored images and vice versa. Real providers
// include CLIP-family models; the interface keeps them pluggable.
type MultiModalEmbedder interface {
	// EmbedText converts text into a vector in the shared space.
	EmbedText(ctx context.Context, text string) ([]float32, error)

	// EmbedImage converts raw image bytes (with MIME type) into a
	// vector in the shared space.
	EmbedImage(ctx context.Context, data []byte, mimeType string) ([]float32, error)

	// Dimension returns the shared embedding dimension.
	Dimension() int
}

// Modality labels the kind of content stored in a multi-modal index.
type Modality string

const (
	// ModalityText is plain text content.
	ModalityText Modality = "text"
	// ModalityImage is raw image content.
	ModalityImage Modality = "image"
)

// MockMultiModal is a deterministic multi-modal embedder for tests and
// offline use. Vectors are derived from FNV-1a hashes so that:
//   - identical inputs produce identical vectors (cosine 1),
//   - distinct inputs are near-orthogonal in expectation,
//   - text and image hashes share the same space, so a query "about"
//     content that reuses the same seed phrase matches deterministically
//     (see SeedForText for building such pairs).
type MockMultiModal struct {
	dim int
}

// NewMockMultiModal creates a mock multi-modal embedder of the given
// dimension (min 8).
func NewMockMultiModal(dim int) *MockMultiModal {
	if dim < 8 {
		dim = 8
	}
	return &MockMultiModal{dim: dim}
}

// Dimension returns the configured dimension.
func (m *MockMultiModal) Dimension() int { return m.dim }

// EmbedText embeds text deterministically.
func (m *MockMultiModal) EmbedText(_ context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("multimodal: empty text")
	}
	return m.vectorFor("text:" + text), nil
}

// EmbedImage embeds image bytes deterministically.
func (m *MockMultiModal) EmbedImage(_ context.Context, data []byte, mimeType string) ([]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("multimodal: empty image data")
	}
	if mimeType == "" {
		return nil, fmt.Errorf("multimodal: mimeType is required")
	}
	return m.vectorFor(fmt.Sprintf("image:%s:%x", mimeType, hexHash(data))), nil
}

// SeedForText returns the canonical seed string for a text embedding.
// Tests use it to assert exact vector equality.
func (m *MockMultiModal) SeedForText(text string) string { return "text:" + text }

// vectorFor builds a pseudo-random unit vector from a seed string.
func (m *MockMultiModal) vectorFor(seed string) []float32 {
	v := make([]float32, m.dim)
	h := fnvHash(seed)
	var norm float64
	for i := 0; i < m.dim; i++ {
		// Advance the hash independently per dimension.
		x := fnvHash(fmt.Sprintf("%x:%d", h, i))
		// Map to [-1, 1].
		v[i] = float32(int64(x)&0xFFFF)/65535*2 - 1
		norm += float64(v[i]) * float64(v[i])
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func fnvHash(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

// hexHash returns the FNV-1a hash of raw bytes.
func hexHash(b []byte) uint64 {
	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

// Compile-time interface check.
var _ MultiModalEmbedder = (*MockMultiModal)(nil)
