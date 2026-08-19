package cache

import (
	"crypto/sha256"
	"fmt"
	"time"

	"github.com/deagy/recall/graph"
)

// GraphTraversalResult represents a cached graph traversal result.
type GraphTraversalResult struct {
	Query     string
	Results   []*graph.Entity
	Timestamp time.Time
	Latency   time.Duration
}

// GraphCache provides caching for graph traversal results.
type GraphCache struct {
	inner *LRUCache
}

// NewGraphCache creates a new GraphCache with default configuration.
func NewGraphCache(maxSize int) *GraphCache {
	config := DefaultCacheConfig()
	config.MaxSize = maxSize
	return &GraphCache{
		inner: NewLRUCache(config),
	}
}

// GenerateGraphTraversalKey generates a cache key for a graph traversal query.
func GenerateGraphTraversalKey(entityID string, traversalType string, depth int) string {
	key := fmt.Sprintf("graph:%s:%s:%d", entityID, traversalType, depth)
	hash := sha256.Sum256([]byte(key))
	return fmt.Sprintf("graph:%x", hash[:8])
}

// Get retrieves a cached graph traversal result.
func (gc *GraphCache) Get(entityID string, traversalType string, depth int) (*GraphTraversalResult, bool) {
	key := GenerateGraphTraversalKey(entityID, traversalType, depth)
	val, ok := gc.inner.Get(key)
	if !ok {
		return nil, false
	}
	return val.(*GraphTraversalResult), true
}

// Set stores a graph traversal result in the cache.
func (gc *GraphCache) Set(entityID string, traversalType string, depth int, result *GraphTraversalResult, ttl time.Duration) {
	key := GenerateGraphTraversalKey(entityID, traversalType, depth)
	gc.inner.Set(key, result, ttl)
}

// Delete removes a cached graph traversal result.
func (gc *GraphCache) Delete(entityID string, traversalType string, depth int) {
	key := GenerateGraphTraversalKey(entityID, traversalType, depth)
	gc.inner.Delete(key)
}

// Clear removes all cached graph traversal results.
func (gc *GraphCache) Clear() {
	gc.inner.Clear()
}

// Stats returns cache statistics.
func (gc *GraphCache) Stats() CacheStats {
	return gc.inner.Stats()
}

// InvalidateByEntity invalidates all cached results for a specific entity.
func (gc *GraphCache) InvalidateByEntity(entityID string) {
	// Simplified: clear all graph cache entries
	// In a real implementation, you might maintain an entity-to-key index
	gc.Clear()
}
