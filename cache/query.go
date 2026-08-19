package cache

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
	"time"
)

// QueryResult represents a cached query result.
type QueryResult struct {
	Query     string
	Results   []interface{}
	Timestamp time.Time
	Latency   time.Duration
}

// QueryCache provides caching for query results.
type QueryCache struct {
	inner *LRUCache
}

// NewQueryCache creates a new QueryCache with default configuration.
func NewQueryCache(maxSize int) *QueryCache {
	config := DefaultCacheConfig()
	config.MaxSize = maxSize
	return &QueryCache{
		inner: NewLRUCache(config),
	}
}

// GenerateQueryKey generates a cache key for a query and its filters.
func GenerateQueryKey(query string, filters map[string]interface{}) string {
	// Create a canonical representation of the query and filters
	var parts []string
	parts = append(parts, query)

	// Sort filters by key for deterministic ordering
	keys := make([]string, 0, len(filters))
	for k := range filters {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, filters[k]))
	}

	canonical := strings.Join(parts, "|")

	// Hash the canonical string
	hash := sha256.Sum256([]byte(canonical))
	return fmt.Sprintf("query:%x", hash[:8])
}

// Get retrieves a cached query result.
func (qc *QueryCache) Get(query string, filters map[string]interface{}) (*QueryResult, bool) {
	key := GenerateQueryKey(query, filters)
	val, ok := qc.inner.Get(key)
	if !ok {
		return nil, false
	}
	return val.(*QueryResult), true
}

// Set stores a query result in the cache.
func (qc *QueryCache) Set(query string, filters map[string]interface{}, result *QueryResult, ttl time.Duration) {
	key := GenerateQueryKey(query, filters)
	qc.inner.Set(key, result, ttl)
}

// Delete removes a cached query result.
func (qc *QueryCache) Delete(query string, filters map[string]interface{}) {
	key := GenerateQueryKey(query, filters)
	qc.inner.Delete(key)
}

// Clear removes all cached query results.
func (qc *QueryCache) Clear() {
	qc.inner.Clear()
}

// Stats returns cache statistics.
func (qc *QueryCache) Stats() CacheStats {
	return qc.inner.Stats()
}

// InvalidateByPrefix invalidates all cached results matching a query prefix.
func (qc *QueryCache) InvalidateByPrefix(prefix string) {
	// Note: This is a simplified implementation. In a real system, you might
	// want to maintain an index of query prefixes for efficient invalidation.
	// For now, we'll just clear the entire cache.
	qc.Clear()
}
