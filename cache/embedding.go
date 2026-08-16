package cache

import (
	"crypto/sha256"
	"fmt"
	"time"
)

// EmbeddingCache provides caching for embedding vectors to avoid redundant computation.
type EmbeddingCache struct {
	inner *LRUCache
}

// NewEmbeddingCache creates a new EmbeddingCache with default configuration.
func NewEmbeddingCache(maxSize int) *EmbeddingCache {
	config := DefaultCacheConfig()
	config.MaxSize = maxSize
	return &EmbeddingCache{
		inner: NewLRUCache(config),
	}
}

// GenerateKey generates a cache key for an embedding input.
func GenerateEmbeddingKey(text string) string {
	hash := sha256.Sum256([]byte(text))
	return fmt.Sprintf("emb:%x", hash[:8])
}

// Get retrieves a cached embedding vector.
func (ec *EmbeddingCache) Get(text string) ([]float32, bool) {
	key := GenerateEmbeddingKey(text)
	val, ok := ec.inner.Get(key)
	if !ok {
		return nil, false
	}
	return val.([]float32), true
}

// Set stores an embedding vector in the cache.
func (ec *EmbeddingCache) Set(text string, embedding []float32, ttl time.Duration) {
	key := GenerateEmbeddingKey(text)
	ec.inner.Set(key, embedding, ttl)
}

// Delete removes a cached embedding.
func (ec *EmbeddingCache) Delete(text string) {
	key := GenerateEmbeddingKey(text)
	ec.inner.Delete(key)
}

// Clear removes all cached embeddings.
func (ec *EmbeddingCache) Clear() {
	ec.inner.Clear()
}

// Stats returns cache statistics.
func (ec *EmbeddingCache) Stats() CacheStats {
	return ec.inner.Stats()
}

// InvalidateByPrefix invalidates all cached embeddings matching a prefix.
func (ec *EmbeddingCache) InvalidateByPrefix(prefix string) {
	// Simplified: clear all embeddings
	ec.Clear()
}
