// Package testutil provides reusable, deterministic test helpers for the
// Recall SDK: a pre-populated fixture store, a deterministic embedder,
// golden-file helpers, and a benchmark harness.
//
// It is intended to be imported from _test.go files only. It adds no
// production code to the SDK.
package testutil

import (
	"context"
	"fmt"
	"strconv"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/store"
)

// DefaultFixtureDim is the embedding dimension used by fixture stores.
const DefaultFixtureDim = 64

// FixtureDoc describes a document to load into a fixture store.
type FixtureDoc struct {
	// ID is the document ID. When empty it defaults to "doc-<index>".
	ID string

	// Title is the document title.
	Title string

	// Content is the document text.
	Content string

	// Namespace optionally overrides the store's default namespace.
	Namespace string

	// Tags are arbitrary labels.
	Tags []string

	// Metadata is arbitrary metadata.
	Metadata map[string]core.Value
}

// FixtureStore is a deterministic, pre-populated in-memory store for tests.
// The store is built with a mock embedder (seeded only by text content), so
// identical fixtures always produce identical embeddings and search results.
type FixtureStore struct {
	// Store is the underlying store (satisfies store.Store).
	Store store.Store

	// Embedder is the deterministic embedder used by the store.
	Embedder *embedder.MockEmbedder
}

// NewFixtureStore creates a MemoryStore backed by a deterministic mock
// embedder and loads the given fixture documents. Documents are chunked with a
// very large token budget so each document normally produces a single chunk
// with a predictable ID (see ChunkID). The store namespace is "test" unless a
// document overrides it.
func NewFixtureStore(docs ...FixtureDoc) (*FixtureStore, error) {
	embed := embedder.NewMockEmbedder(DefaultFixtureDim)
	cfg := store.Config{
		Namespace: "test",
		Embedder:  embed,
		ChunkerFactory: func(cfg chunker.Config) chunker.Chunker {
			cfg.MaxTokens = 1_000_000 // keep fixture docs in a single chunk
			cfg.MinChunkSize = 1
			cfg.OverlapTokens = 0
			return chunker.NewFixed(cfg)
		},
	}
	s, err := store.NewMemoryStore(cfg)
	if err != nil {
		return nil, err
	}
	for i, fd := range docs {
		id := fd.ID
		if id == "" {
			id = "doc-" + strconv.Itoa(i)
		}
		doc := &core.Document{
			ID:        id,
			Title:     fd.Title,
			Namespace: fd.Namespace,
			Tags:      fd.Tags,
			Metadata:  fd.Metadata,
		}
		if err := s.Upload(context.Background(), doc, fd.Content); err != nil {
			return nil, fmt.Errorf("fixture %d (%s): %w", i, id, err)
		}
	}
	return &FixtureStore{Store: s, Embedder: embed}, nil
}

// ChunkID returns the predictable chunk ID for chunk index i of a document.
// All built-in chunkers name chunks "<docID>::chunk-<i>".
func ChunkID(docID string, i int) string {
	return docID + "::chunk-" + strconv.Itoa(i)
}

// FirstChunkID returns the ID of the first chunk of a fixture document.
func (f *FixtureStore) FirstChunkID(docID string) string {
	return ChunkID(docID, 0)
}

// Close closes the underlying store.
func (f *FixtureStore) Close() error {
	return f.Store.Close()
}
