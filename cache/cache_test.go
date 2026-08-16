package cache

import (
	"testing"
	"time"
)

func TestCacheEntry_IsExpired(t *testing.T) {
	tests := []struct {
		name     string
		entry    CacheEntry
		expected bool
	}{
		{
			name: "not expired",
			entry: CacheEntry{
				ExpiresAt: time.Now().Add(1 * time.Hour),
			},
			expected: false,
		},
		{
			name: "expired",
			entry: CacheEntry{
				ExpiresAt: time.Now().Add(-1 * time.Hour),
			},
			expected: true,
		},
		{
			name: "no expiry",
			entry: CacheEntry{
				ExpiresAt: time.Time{},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.entry.IsExpired()
			if result != tt.expected {
				t.Errorf("IsExpired() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestDefaultCacheConfig(t *testing.T) {
	config := DefaultCacheConfig()

	if config.MaxSize != 10000 {
		t.Errorf("Expected MaxSize 10000, got %d", config.MaxSize)
	}
	if config.DefaultTTL != 5*time.Minute {
		t.Errorf("Expected DefaultTTL 5m, got %v", config.DefaultTTL)
	}
	if config.EvictionPolicy != EvictionLRU {
		t.Errorf("Expected EvictionPolicy LRU, got %v", config.EvictionPolicy)
	}
}

func TestCacheManager_GetCache(t *testing.T) {
	config := DefaultCacheConfig()
	manager := NewCacheManager(config)

	// Get cache (should create new)
	cache1 := manager.GetCache("test-cache")
	if cache1 == nil {
		t.Fatal("Expected non-nil cache")
	}

	// Get same cache again (should return same instance)
	cache2 := manager.GetCache("test-cache")
	if cache1 != cache2 {
		t.Error("Expected same cache instance")
	}

	// Get different cache
	cache3 := manager.GetCache("other-cache")
	if cache1 == cache3 {
		t.Error("Expected different cache instance")
	}
}

func TestCacheManager_DeleteCache(t *testing.T) {
	config := DefaultCacheConfig()
	manager := NewCacheManager(config)

	// Create cache
	manager.GetCache("test-cache")

	// Delete cache
	manager.DeleteCache("test-cache")

	// Try to get deleted cache (should create new)
	cache := manager.GetCache("test-cache")
	if cache == nil {
		t.Fatal("Expected non-nil cache")
	}
}

func TestCacheManager_ClearAll(t *testing.T) {
	config := DefaultCacheConfig()
	manager := NewCacheManager(config)

	// Create caches and add data
	cache1 := manager.GetCache("cache1")
	cache1.Set("key1", "value1", 0)

	cache2 := manager.GetCache("cache2")
	cache2.Set("key2", "value2", 0)

	// Clear all
	manager.ClearAll()

	// Verify data is cleared
	_, ok := cache1.Get("key1")
	if ok {
		t.Error("Expected key1 to be cleared")
	}

	_, ok = cache2.Get("key2")
	if ok {
		t.Error("Expected key2 to be cleared")
	}
}

func TestCacheManager_Stats(t *testing.T) {
	config := DefaultCacheConfig()
	manager := NewCacheManager(config)

	// Create cache and add data
	cache := manager.GetCache("test-cache")
	cache.Set("key1", "value1", 0)
	cache.Set("key2", "value2", 0)

	// Get stats
	stats := manager.Stats()
	if len(stats) != 1 {
		t.Errorf("Expected 1 cache in stats, got %d", len(stats))
	}

	cacheStats := stats["test-cache"]
	if cacheStats.Size != 2 {
		t.Errorf("Expected cache size 2, got %d", cacheStats.Size)
	}
}
