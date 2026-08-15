package core

import (
	"time"
)

// Chunk represents a unit of text content with associated metadata.
// Chunks are the primary data unit for retrieval in RAG pipelines.
type Chunk struct {
	// ID is a unique identifier for this chunk.
	ID string

	// Content is the text content of this chunk.
	Content string

	// DocumentRef identifies the source document this chunk belongs to.
	DocumentRef string

	// ChunkIndex is the zero-based index of this chunk within its document.
	ChunkIndex int

	// Embedding is the vector embedding of this chunk's content.
	// May be nil if the chunk has not yet been embedded.
	Embedding []float32

	// Metadata contains arbitrary key-value metadata about this chunk.
	// Common keys include: source, author, date, tags, title.
	Metadata map[string]Value

	// CreatedAt is when this chunk was added to the store.
	CreatedAt time.Time

	// UpdatedAt is when this chunk was last updated.
	UpdatedAt time.Time
}

// EmbeddingDimension returns the dimension of this chunk's embedding.
func (c *Chunk) EmbeddingDimension() int {
	if c.Embedding == nil {
		return 0
	}
	return len(c.Embedding)
}

// HasEmbedding returns true if this chunk has a non-nil embedding.
func (c *Chunk) HasEmbedding() bool {
	return len(c.Embedding) > 0
}

// GetMetadata returns the value for a metadata key, or nil if not present.
func (c *Chunk) GetMetadata(key string) Value {
	if c.Metadata == nil {
		return nil
	}
	return c.Metadata[key]
}

// GetMetadataString returns the string value for a metadata key.
func (c *Chunk) GetMetadataString(key string) string {
	v := c.GetMetadata(key)
	if v == nil {
		return ""
	}
	return v.String()
}
