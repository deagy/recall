package cache

import (
	"time"
)

// MultiLevelCache provides multi-level caching with L1 (fast, small) and L2 (slower, larger) tiers.
type MultiLevelCache struct {
	L1 Cache // In-memory (fast, small)
	L2 Cache // Can be disk-based or larger in-memory (slower, larger)
}

// NewMultiLevelCache creates a new MultiLevelCache with L1 and L2 caches.
func NewMultiLevelCache(l1, l2 Cache) *MultiLevelCache {
	return &MultiLevelCache{
		L1: l1,
		L2: l2,
	}
}

// Get retrieves a value from the cache, checking L1 first, then L2.
func (mlc *MultiLevelCache) Get(key string) (interface{}, bool) {
	// Check L1 first (fast)
	if val, ok := mlc.L1.Get(key); ok {
		return val, true
	}

	// Check L2 (slower)
	if val, ok := mlc.L2.Get(key); ok {
		// Promote to L1 for faster future access
		mlc.L1.Set(key, val, 0)
		return val, true
	}

	return nil, false
}

// Set stores a value in both L1 and L2 caches.
func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) {
	// Set in L1 with shorter TTL
	mlc.L1.Set(key, value, ttl/2)

	// Set in L2 with full TTL
	mlc.L2.Set(key, value, ttl)
}

// Delete removes a value from both L1 and L2 caches.
func (mlc *MultiLevelCache) Delete(key string) {
	mlc.L1.Delete(key)
	mlc.L2.Delete(key)
}

// Clear removes all values from both L1 and L2 caches.
func (mlc *MultiLevelCache) Clear() {
	mlc.L1.Clear()
	mlc.L2.Clear()
}

// Stats returns aggregated statistics from both L1 and L2 caches.
func (mlc *MultiLevelCache) Stats() CacheStats {
	l1Stats := mlc.L1.Stats()
	l2Stats := mlc.L2.Stats()

	return CacheStats{
		Hits:   l1Stats.Hits + l2Stats.Hits,
		Misses: l1Stats.Misses + l2Stats.Misses,
		Size:   l1Stats.Size + l2Stats.Size,
	}
}
