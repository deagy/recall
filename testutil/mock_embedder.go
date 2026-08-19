package testutil

import (
	"context"

	"github.com/deagy/recall/embedder"
)

// MockEmbedder is a deterministic embedder for tests: identical text always
// yields an identical (L2-normalized) vector, with no external dependencies.
// It is an alias for embedder.MockEmbedder, re-exported here so test files can
// import a single package for all test doubles.
type MockEmbedder = embedder.MockEmbedder

// NewMockEmbedder creates a deterministic mock embedder with the given
// dimension (0 or negative defaults to 384).
func NewMockEmbedder(dim int) *MockEmbedder {
	return embedder.NewMockEmbedder(dim)
}

// DeterministicEmbed returns the mock embedder's vector for text, asserting no
// error (the mock never fails). Convenient for building expected embeddings in
// tests.
func DeterministicEmbed(e *MockEmbedder, text string) []float32 {
	vec, err := e.Embed(context.Background(), text)
	if err != nil {
		panic("testutil: MockEmbedder never errors: " + err.Error())
	}
	return vec
}
