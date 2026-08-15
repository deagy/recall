package core

import (
	"testing"
	"time"
)

func TestChunkEmbeddingDimension(t *testing.T) {
	c := &Chunk{}
	if c.EmbeddingDimension() != 0 {
		t.Errorf("expected 0, got %d", c.EmbeddingDimension())
	}

	c.Embedding = make([]float32, 1536)
	if c.EmbeddingDimension() != 1536 {
		t.Errorf("expected 1536, got %d", c.EmbeddingDimension())
	}
}

func TestChunkHasEmbedding(t *testing.T) {
	c := &Chunk{}
	if c.HasEmbedding() {
		t.Error("expected false")
	}

	c.Embedding = []float32{1, 2, 3}
	if !c.HasEmbedding() {
		t.Error("expected true")
	}
}

func TestChunkGetMetadata(t *testing.T) {
	c := &Chunk{
		Metadata: map[string]Value{
			"source": String{Value: "test.txt"},
			"date":   Number{Value: 12345},
		},
	}

	v := c.GetMetadata("source")
	if v == nil {
		t.Fatal("expected non-nil")
	}
	if v.String() != "test.txt" {
		t.Errorf("expected 'test.txt', got %q", v.String())
	}

	v = c.GetMetadata("missing")
	if v != nil {
		t.Errorf("expected nil, got %v", v)
	}

	str := c.GetMetadataString("source")
	if str != "test.txt" {
		t.Errorf("expected 'test.txt', got %q", str)
	}

	str = c.GetMetadataString("missing")
	if str != "" {
		t.Errorf("expected '', got %q", str)
	}
}

func TestChunkMetadataNil(t *testing.T) {
	c := &Chunk{}
	if c.GetMetadata("key") != nil {
		t.Error("expected nil for nil metadata map")
	}
}

func TestChunkTimestamps(t *testing.T) {
	now := time.Now()
	c := &Chunk{
		CreatedAt: now,
		UpdatedAt: now.Add(time.Second),
	}
	if c.CreatedAt.After(c.UpdatedAt) {
		t.Error("CreatedAt should be <= UpdatedAt")
	}
}
