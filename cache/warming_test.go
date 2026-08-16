package cache

import (
	"testing"
	"time"
)

func TestCacheWarmer_AddAndWarm(t *testing.T) {
	cache := NewLRUCache(DefaultCacheConfig())
	warmer := NewCacheWarmer(cache)

	// Add warm requests
	warmer.AddRequest(WarmRequest{
		Query: "popular query 1",
		Result: []string{"result1", "result2"},
		TTL:    5 * time.Minute,
	})

	warmer.AddRequest(WarmRequest{
		Query: "popular query 2",
		Result: []string{"result3", "result4"},
		TTL:    5 * time.Minute,
	})

	// Warm the cache
	warmer.Warm()

	// Verify results are in cache
	_, ok := cache.Get("popular query 1")
	if !ok {
		t.Error("Expected 'popular query 1' to be in cache")
	}

	_, ok = cache.Get("popular query 2")
	if !ok {
		t.Error("Expected 'popular query 2' to be in cache")
	}
}

func TestCacheWarmer_ClearPending(t *testing.T) {
	cache := NewLRUCache(DefaultCacheConfig())
	warmer := NewCacheWarmer(cache)

	// Add warm requests
	warmer.AddRequest(WarmRequest{
		Query:  "query 1",
		Result: "result1",
		TTL:    5 * time.Minute,
	})

	// Clear pending
	warmer.ClearPending()

	// Warm (should do nothing)
	warmer.Warm()

	// Verify nothing is in cache
	_, ok := cache.Get("query 1")
	if ok {
		t.Error("Expected 'query 1' to not be in cache")
	}
}

func TestCacheWarmer_Stats(t *testing.T) {
	cache := NewLRUCache(DefaultCacheConfig())
	warmer := NewCacheWarmer(cache)

	// Add warm requests
	warmer.AddRequest(WarmRequest{
		Query:  "query 1",
		Result: "result1",
		TTL:    5 * time.Minute,
	})

	warmer.AddRequest(WarmRequest{
		Query:  "query 2",
		Result: "result2",
		TTL:    5 * time.Minute,
	})

	// Warm
	warmer.Warm()

	// Check stats
	stats := warmer.Stats()
	if stats.TotalRequests != 2 {
		t.Errorf("Expected 2 total requests, got %d", stats.TotalRequests)
	}
	if stats.TotalWarmed != 2 {
		t.Errorf("Expected 2 total warmed, got %d", stats.TotalWarmed)
	}
}

func TestWarmStats_String(t *testing.T) {
	ws := WarmStats{
		TotalRequests: 10,
		TotalWarmed:   8,
		StartTime:     time.Now().Add(-1 * time.Minute),
	}

	str := ws.String()
	if str == "" {
		t.Error("Expected non-empty string")
	}
}
