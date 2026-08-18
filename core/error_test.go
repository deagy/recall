package core

import (
	"errors"
	"testing"
)

func TestErrorVariables(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected string
	}{
		{"ErrNotFound", ErrNotFound, "recall: not found"},
		{"ErrDuplicate", ErrDuplicate, "recall: duplicate"},
		{"ErrInvalidChunk", ErrInvalidChunk, "recall: invalid chunk"},
		{"ErrInvalidDocument", ErrInvalidDocument, "recall: invalid document"},
		{"ErrInvalidEmbedding", ErrInvalidEmbedding, "recall: invalid embedding"},
		{"ErrEmbeddingMismatch", ErrEmbeddingMismatch, "recall: embedding dimension mismatch"},
		{"ErrNamespaceNotFound", ErrNamespaceNotFound, "recall: namespace not found"},
		{"ErrNamespaceExists", ErrNamespaceExists, "recall: namespace exists"},
		{"ErrEmptyQuery", ErrEmptyQuery, "recall: empty query"},
		{"ErrContextCancelled", ErrContextCancelled, "recall: context cancelled"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.err == nil {
				t.Fatal("expected non-nil error")
			}
			if tt.err.Error() != tt.expected {
				t.Errorf("expected error message %q, got %q", tt.expected, tt.err.Error())
			}
		})
	}
}

func TestErrorIs(t *testing.T) {
	if !errors.Is(ErrNotFound, ErrNotFound) {
		t.Error("ErrNotFound should match itself")
	}
	if errors.Is(ErrNotFound, ErrDuplicate) {
		t.Error("ErrNotFound should not match ErrDuplicate")
	}
}
