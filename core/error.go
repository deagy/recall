package core

import "errors"

var (
	// ErrNotFound is returned when a requested entity does not exist.
	ErrNotFound = errors.New("recall: not found")

	// ErrDuplicate is returned when attempting to add a duplicate entity.
	ErrDuplicate = errors.New("recall: duplicate")

	// ErrInvalidChunk is returned when a chunk has invalid content.
	ErrInvalidChunk = errors.New("recall: invalid chunk")

	// ErrInvalidDocument is returned when a document is missing or invalid.
	ErrInvalidDocument = errors.New("recall: invalid document")

	// ErrInvalidEmbedding is returned when an embedding has invalid dimensions.
	ErrInvalidEmbedding = errors.New("recall: invalid embedding")

	// ErrEmbeddingMismatch is returned when embedding dimensions don't match the index.
	ErrEmbeddingMismatch = errors.New("recall: embedding dimension mismatch")

	// ErrNamespaceNotFound is returned when a namespace does not exist.
	ErrNamespaceNotFound = errors.New("recall: namespace not found")

	// ErrNamespaceExists is returned when a namespace already exists.
	ErrNamespaceExists = errors.New("recall: namespace exists")

	// ErrEmptyQuery is returned when a query has no search criteria.
	ErrEmptyQuery = errors.New("recall: empty query")

	// ErrContextCancelled is returned when the context is cancelled during a long operation.
	ErrContextCancelled = errors.New("recall: context cancelled")
)
