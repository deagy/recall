// Package chunker provides text chunking strategies for splitting documents
// into manageable pieces for embedding and retrieval.
package chunker

import (
	"github.com/deagy/recall/core"
)

// Config holds configuration for a chunker.
type Config struct {
	// MaxTokens is the maximum number of tokens per chunk.
	// Used by fixed-size chunkers.
	MaxTokens int

	// MinChunkSize is the minimum number of characters for a chunk. A chunk
	// below it is discarded, never merged: FixedChunker drops one built from a
	// single part -- a piece that had nothing to combine with -- and
	// RecursiveChunker drops any piece under the threshold outright.
	// AdaptiveChunker instead reads it as a lower size bound.
	//
	// The distinction matters under a boundary-preserving strategy such as
	// DocumentAwareChunker, which chunks each section independently: a section
	// shorter than this yields no chunk at all, and nothing records that it was
	// there. Set it to 0 to keep every chunk.
	MinChunkSize int

	// OverlapTokens is the number of tokens to overlap between adjacent chunks.
	// Helps maintain context across chunk boundaries.
	OverlapTokens int

	// Separator is the character or string used to split text.
	// Used by separator-based chunkers. Default is "\n\n".
	Separator string
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxTokens:     512,
		MinChunkSize:  50,
		OverlapTokens: 50,
		Separator:     "\n\n",
	}
}

// Chunker defines the interface for splitting documents into chunks.
type Chunker interface {
	// Chunk splits a document's text content into chunks.
	Chunk(doc *core.Document, content string) ([]*core.Chunk, error)
}

// Factory creates a Chunker with the given configuration.
type Factory func(cfg Config) Chunker
