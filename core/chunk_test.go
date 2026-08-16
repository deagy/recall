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

func TestChunkGetMetadataString_NilMetadata(t *testing.T) {
	c := &Chunk{}
	str := c.GetMetadataString("key")
	if str != "" {
		t.Errorf("expected empty string for nil metadata, got %q", str)
	}
}

func TestChunkGetMetadataString_NilValue(t *testing.T) {
	c := &Chunk{
		Metadata: map[string]Value{
			"key": nil,
		},
	}
	str := c.GetMetadataString("key")
	if str != "" {
		t.Errorf("expected empty string for nil value, got %q", str)
	}
}

func TestChunkEmbeddingDimension_EmptySlice(t *testing.T) {
	c := &Chunk{
		Embedding: []float32{},
	}
	if c.EmbeddingDimension() != 0 {
		t.Errorf("expected 0, got %d", c.EmbeddingDimension())
	}
	if c.HasEmbedding() {
		t.Error("expected false for empty slice")
	}
}

func TestChunkEmbeddingDimension_SingleElement(t *testing.T) {
	c := &Chunk{
		Embedding: []float32{1.0},
	}
	if c.EmbeddingDimension() != 1 {
		t.Errorf("expected 1, got %d", c.EmbeddingDimension())
	}
	if !c.HasEmbedding() {
		t.Error("expected true for single element")
	}
}

func TestChunkMetadata_CopyBehavior(t *testing.T) {
	c := &Chunk{
		Metadata: map[string]Value{
			"key": String{Value: "original"},
		},
	}

	// Get metadata reference
	v := c.GetMetadata("key")
	if v.String() != "original" {
		t.Errorf("expected 'original', got %q", v.String())
	}

	// Verify we can modify through the map
	c.Metadata["key"] = String{Value: "modified"}
	v2 := c.GetMetadata("key")
	if v2.String() != "modified" {
		t.Errorf("expected 'modified', got %q", v2.String())
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

func TestChunkTimestamps_SameTime(t *testing.T) {
	now := time.Now()
	c := &Chunk{
		CreatedAt: now,
		UpdatedAt: now,
	}
	if c.CreatedAt.After(c.UpdatedAt) {
		t.Error("CreatedAt should be <= UpdatedAt")
	}
}

func TestChunk_Embedding_EmptySlice(t *testing.T) {
	c := &Chunk{
		Embedding: []float32{},
	}
	if c.EmbeddingDimension() != 0 {
		t.Errorf("expected 0, got %d", c.EmbeddingDimension())
	}
	if c.HasEmbedding() {
		t.Error("expected false for empty slice")
	}
}

func TestChunk_Embedding_SingleElement(t *testing.T) {
	c := &Chunk{
		Embedding: []float32{1.0},
	}
	if c.EmbeddingDimension() != 1 {
		t.Errorf("expected 1, got %d", c.EmbeddingDimension())
	}
	if !c.HasEmbedding() {
		t.Error("expected true for single element")
	}
}

func TestChunk_Embedding_MultipleElements(t *testing.T) {
	c := &Chunk{
		Embedding: []float32{1.0, 2.0, 3.0, 4.0, 5.0},
	}
	if c.EmbeddingDimension() != 5 {
		t.Errorf("expected 5, got %d", c.EmbeddingDimension())
	}
	if !c.HasEmbedding() {
		t.Error("expected true for multiple elements")
	}
}

func TestChunk_Metadata_MultipleKeys(t *testing.T) {
	c := &Chunk{
		Metadata: map[string]Value{
			"key1": String{Value: "value1"},
			"key2": Number{Value: 42},
			"key3": Boolean{Value: true},
		},
	}

	if c.GetMetadataString("key1") != "value1" {
		t.Error("expected 'value1'")
	}
	if c.GetMetadataString("key2") != "42" {
		t.Error("expected '42'")
	}
	if c.GetMetadataString("key3") != "true" {
		t.Error("expected 'true'")
	}
}

func TestChunk_Metadata_Update(t *testing.T) {
	c := &Chunk{
		Metadata: map[string]Value{
			"key": String{Value: "original"},
		},
	}

	c.Metadata["key"] = String{Value: "updated"}
	if c.GetMetadataString("key") != "updated" {
		t.Error("expected 'updated'")
	}
}

func TestChunk_Metadata_Delete(t *testing.T) {
	c := &Chunk{
		Metadata: map[string]Value{
			"key": String{Value: "value"},
		},
	}

	delete(c.Metadata, "key")
	if c.GetMetadata("key") != nil {
		t.Error("expected nil after delete")
	}
}

func TestDocument_AddTag_ExistingTag(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")
	d.Tags = []string{"go"}
	d.AddTag("go")

	if len(d.Tags) != 1 {
		t.Errorf("expected 1 tag, got %d", len(d.Tags))
	}
}

func TestDocument_AddTag_MultipleUniqueTags(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.AddTag("a")
	d.AddTag("b")
	d.AddTag("c")
	d.AddTag("d")
	d.AddTag("e")

	if len(d.Tags) != 5 {
		t.Errorf("expected 5 tags, got %d", len(d.Tags))
	}
}

func TestDocument_Metadata_MultipleEntries(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.Metadata["key1"] = String{Value: "value1"}
	d.Metadata["key2"] = Number{Value: 42}
	d.Metadata["key3"] = Boolean{Value: true}

	if d.Metadata["key1"].String() != "value1" {
		t.Error("expected 'value1'")
	}
	if d.Metadata["key2"].String() != "42" {
		t.Error("expected '42'")
	}
	if d.Metadata["key3"].String() != "true" {
		t.Error("expected 'true'")
	}
}

func TestDocument_Metadata_Update(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.Metadata["key"] = String{Value: "original"}
	d.Metadata["key"] = String{Value: "updated"}

	if d.Metadata["key"].String() != "updated" {
		t.Error("expected 'updated'")
	}
}

func TestDocument_Metadata_Delete(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.Metadata["key"] = String{Value: "value"}
	delete(d.Metadata, "key")

	if d.Metadata["key"] != nil {
		t.Error("expected nil after delete")
	}
}

func TestDocument_ChunkCount_Update(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	d.ChunkCount = 10
	if d.ChunkCount != 10 {
		t.Errorf("expected 10, got %d", d.ChunkCount)
	}

	d.ChunkCount = 0
	if d.ChunkCount != 0 {
		t.Errorf("expected 0, got %d", d.ChunkCount)
	}
}

func TestDocument_CreatedAt_UpdatedAt(t *testing.T) {
	d := NewDocument("doc-1", "Test", "source")

	if d.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}
	if d.UpdatedAt.IsZero() {
		t.Error("expected non-zero UpdatedAt")
	}
	if d.CreatedAt.After(d.UpdatedAt) {
		t.Error("CreatedAt should be <= UpdatedAt")
	}
}
