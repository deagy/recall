// Package store provides the interface and implementations for the knowledge store.
// The store is the main entry point for RAG operations: uploading documents,
// searching for relevant chunks, and managing namespaces.
package store

import (
	"context"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// Config holds configuration for a store.
type Config struct {
	// Namespace is the default namespace for this store.
	Namespace string

	// Embedder is the embedding implementation to use.
	Embedder embedder.Embedder

	// ChunkerFactory creates chunkers for document splitting.
	ChunkerFactory chunker.Factory
}

// Store defines the interface for the knowledge store.
type Store interface {
	// Upload processes a document: chunks it, embeds the chunks, and indexes them.
	Upload(ctx context.Context, doc *core.Document, content string) error

	// Search finds the most relevant chunks for a query string (vector similarity only).
	Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error)

	// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
	SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error)

	// GetChunk returns a chunk by its ID.
	GetChunk(id string) (*core.Chunk, bool)

	// DeleteChunk removes a chunk from the store.
	DeleteChunk(id string) error

	// DeleteDocument removes all chunks belonging to a document.
	DeleteDocument(docID string) error

	// Count returns the total number of chunks across all namespaces.
	Count() int

	// Namespaces returns the list of namespaces in the store.
	Namespaces() []string

	// Close cleans up any resources held by the store.
	Close() error
}
