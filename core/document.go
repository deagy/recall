package core

import (
	"time"
)

// Document represents a source document that can be chunked and stored.
type Document struct {
	// ID is a unique identifier for this document.
	ID string

	// Title is a human-readable title for the document.
	Title string

	// Author is the author or creator of the document.
	Author string

	// Source is the source location (file path, URL, etc.) of the document.
	Source string

	// Namespace optionally overrides the store's default namespace for this
	// document's chunks. When empty, the store decides (its configured
	// namespace). Supported by the Memory and SQLite stores; search spans
	// all namespaces present in a store.
	Namespace string

	// Tags are arbitrary labels for categorizing the document.
	Tags []string

	// Metadata contains additional arbitrary metadata.
	Metadata map[string]Value

	// CreatedAt is when the document was first added to the store.
	CreatedAt time.Time

	// UpdatedAt is when the document was last updated.
	UpdatedAt time.Time

	// ChunkCount is the number of chunks this document has been split into.
	ChunkCount int
}

// NewDocument creates a new Document with the given ID and content metadata.
func NewDocument(id, title, source string) *Document {
	now := time.Now()
	return &Document{
		ID:        id,
		Title:     title,
		Source:    source,
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  make(map[string]Value),
	}
}

// AddTag adds a tag to the document if it's not already present.
func (d *Document) AddTag(tag string) {
	for _, t := range d.Tags {
		if t == tag {
			return
		}
	}
	d.Tags = append(d.Tags, tag)
}
