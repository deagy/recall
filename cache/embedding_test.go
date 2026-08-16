package cache

import (
	"testing"
	"time"
)

func TestEmbeddingCache_SetAndGet(t *testing.T) {
	ec := NewEmbeddingCache(100)

	embedding := []float32{0.1, 0.2, 0.3, 0.4}
	ec.Set("test text", embedding, 5*time.Minute)

	cached, ok := ec.Get("test text")
	if !ok {
		t.Fatal("Expected to find cached embedding")
	}
	if len(cached) != len(embedding) {
		t.Errorf("Expected embedding length %d, got %d", len(embedding), len(cached))
	}
}

func TestEmbeddingCache_GetNonExistent(t *testing.T) {
	ec := NewEmbeddingCache(100)

	_, ok := ec.Get("nonexistent text")
	if ok {
		t.Error("Expected not to find nonexistent embedding")
	}
}

func TestEmbeddingCache_Delete(t *testing.T) {
	ec := NewEmbeddingCache(100)

	embedding := []float32{0.1, 0.2, 0.3}
	ec.Set("test text", embedding, 5*time.Minute)
	ec.Delete("test text")

	_, ok := ec.Get("test text")
	if ok {
		t.Error("Expected not to find deleted embedding")
	}
}

func TestEmbeddingCache_Clear(t *testing.T) {
	ec := NewEmbeddingCache(100)

	embedding := []float32{0.1, 0.2, 0.3}
	ec.Set("test text", embedding, 5*time.Minute)
	ec.Clear()

	_, ok := ec.Get("test text")
	if ok {
		t.Error("Expected not to find embedding after clear")
	}
}

func TestEmbeddingCache_Stats(t *testing.T) {
	ec := NewEmbeddingCache(100)

	embedding := []float32{0.1, 0.2, 0.3}
	ec.Set("test text", embedding, 5*time.Minute)

	stats := ec.Stats()
	if stats.Size != 1 {
		t.Errorf("Expected size 1, got %d", stats.Size)
	}
}

func TestGenerateEmbeddingKey(t *testing.T) {
	// Test with same text (should produce same key)
	key1 := GenerateEmbeddingKey("test text")
	key2 := GenerateEmbeddingKey("test text")

	if key1 != key2 {
		t.Errorf("Expected same key for same text, got %q and %q", key1, key2)
	}

	// Test with different text (should produce different key)
	key3 := GenerateEmbeddingKey("different text")
	if key1 == key3 {
		t.Error("Expected different key for different text")
	}
}
