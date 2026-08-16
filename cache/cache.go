// Package cache provides intelligent caching for query results, embeddings,
// and graph traversals to reduce latency and cost for repeated queries.
package cache

import (
	"sync"
	"time"
)

// Cache defines the interface for all cache implementations.
type Cache interface {
	// Get retrieves a value from the cache by key.
	Get(key string) (interface{}, bool)

	// Set stores a value in the cache with the given key and TTL.
	Set(key string, value interface{}, ttl time.Duration)

	// Delete removes a value from the cache by key.
	Delete(key string)

	// Clear removes all values from the cache.
	Clear()

	// Stats returns cache statistics.
	Stats() CacheStats
}

// CacheStats contains statistics about cache performance.
type CacheStats struct {
	Hits   int
	Misses int
	Size   int
}

// CacheEntry represents a single entry in the cache.
type CacheEntry struct {
	Value     interface{}
	ExpiresAt time.Time
	CreatedAt time.Time
	LastAccessed time.Time
}

// IsExpired returns true if the cache entry has expired.
func (e *CacheEntry) IsExpired() bool {
	if e.ExpiresAt.IsZero() {
		return false
	}
	return time.Now().After(e.ExpiresAt)
}

// DefaultCacheConfig returns default configuration for caches.
func DefaultCacheConfig() CacheConfig {
	return CacheConfig{
		MaxSize:    10000,
		DefaultTTL: 5 * time.Minute,
		EvictionPolicy: EvictionLRU,
	}
}

// CacheConfig holds configuration for cache instances.
type CacheConfig struct {
	MaxSize        int
	DefaultTTL     time.Duration
	EvictionPolicy EvictionPolicy
}

// EvictionPolicy defines the cache eviction strategy.
type EvictionPolicy int

const (
	// EvictionLRU uses Least Recently Used eviction.
	EvictionLRU EvictionPolicy = iota
	// EvictionLFU uses Least Frequently Used eviction.
	EvictionLFU
	// EvictionFIFO uses First In First Out eviction.
	EvictionFIFO
)

// CacheManager manages multiple cache instances.
type CacheManager struct {
	mu       sync.RWMutex
	caches   map[string]Cache
	defaults CacheConfig
}

// NewCacheManager creates a new CacheManager with default configuration.
func NewCacheManager(config CacheConfig) *CacheManager {
	return &CacheManager{
		caches:   make(map[string]Cache),
		defaults: config,
	}
}

// GetCache returns a cache by name, creating it if it doesn't exist.
func (m *CacheManager) GetCache(name string) Cache {
	m.mu.Lock()
	defer m.mu.Unlock()

	if c, ok := m.caches[name]; ok {
		return c
	}

	// Create a new LRU cache with default config
	c := NewLRUCache(m.defaults)
	m.caches[name] = c
	return c
}

// DeleteCache removes a cache by name.
func (m *CacheManager) DeleteCache(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if c, ok := m.caches[name]; ok {
		c.Clear()
		delete(m.caches, name)
	}
}

// ClearAll clears all caches.
func (m *CacheManager) ClearAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, c := range m.caches {
		c.Clear()
	}
}

// Stats returns aggregated statistics for all caches.
func (m *CacheManager) Stats() map[string]CacheStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := make(map[string]CacheStats)
	for name, c := range m.caches {
		stats[name] = c.Stats()
	}
	return stats
}
