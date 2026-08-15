// Package pipeline provides RAG (Retrieval-Augmented Generation) pipeline
// functionality including context window management, prompt templating,
// and context assembly for LLM queries.
package pipeline

import (
	"fmt"
	"strings"

	"github.com/deagy/recall/core"
)

// ContextWindow manages a collection of chunks within a token limit.
type ContextWindow struct {
	MaxTokens     int
	CurrentTokens int
	Chunks        []core.Chunk
}

// NewContextWindow creates a new context window with the given max token limit.
func NewContextWindow(maxTokens int) *ContextWindow {
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	return &ContextWindow{
		MaxTokens: maxTokens,
		Chunks:    make([]core.Chunk, 0),
	}
}

// AddChunk adds a chunk to the context window if it fits within the token limit.
// Returns true if the chunk was added, false if it would exceed the limit.
func (cw *ContextWindow) AddChunk(chunk core.Chunk) bool {
	tokens := cw.estimateTokens(chunk.Content)
	if cw.CurrentTokens+tokens > cw.MaxTokens {
		return false
	}
	cw.Chunks = append(cw.Chunks, chunk)
	cw.CurrentTokens += tokens
	return true
}

// estimateTokens approximates token count as characters / 4 (rough estimate).
func (cw *ContextWindow) estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	return len(s) / 4
}

// Tokens returns the current token count.
func (cw *ContextWindow) Tokens() int {
	return cw.CurrentTokens
}

// IsEmpty returns true if no chunks have been added.
func (cw *ContextWindow) IsEmpty() bool {
	return len(cw.Chunks) == 0
}

// String returns a formatted string of all chunks in the context.
func (cw *ContextWindow) String() string {
	var b strings.Builder
	for i, chunk := range cw.Chunks {
		if i > 0 {
			b.WriteString("\n\n---\n\n")
		}
		fmt.Fprintf(&b, "[Chunk %d] (ID: %s)\n%s", i+1, chunk.ID, chunk.Content)
	}
	return b.String()
}

// Len returns the number of chunks in the context.
func (cw *ContextWindow) Len() int {
	return len(cw.Chunks)
}
