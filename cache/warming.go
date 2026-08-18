package cache

import (
	"fmt"
	"sync"
	"time"
)

// WarmRequest represents a request to warm the cache.
type WarmRequest struct {
	Query   string
	Filters map[string]interface{}
	Result  interface{}
	TTL     time.Duration
}

// CacheWarmer provides cache warming strategies to pre-populate the cache.
type CacheWarmer struct {
	mu       sync.Mutex
	cache    Cache
	requests []WarmRequest
	stats    WarmStats
}

// WarmStats tracks cache warming statistics.
type WarmStats struct {
	TotalRequests int
	TotalWarmed   int
	StartTime     time.Time
}

// NewCacheWarmer creates a new CacheWarmer.
func NewCacheWarmer(cache Cache) *CacheWarmer {
	return &CacheWarmer{
		cache:    cache,
		requests: make([]WarmRequest, 0),
		stats: WarmStats{
			StartTime: time.Now(),
		},
	}
}

// AddRequest adds a request to warm the cache.
func (cw *CacheWarmer) AddRequest(req WarmRequest) {
	cw.mu.Lock()
	defer cw.mu.Unlock()

	cw.requests = append(cw.requests, req)
	cw.stats.TotalRequests++
}

// Warm executes all pending warm requests.
func (cw *CacheWarmer) Warm() {
	cw.mu.Lock()
	requests := make([]WarmRequest, len(cw.requests))
	copy(requests, cw.requests)
	cw.mu.Unlock()

	for _, req := range requests {
		cw.cache.Set(req.Query, req.Result, req.TTL)
		cw.mu.Lock()
		cw.stats.TotalWarmed++
		cw.mu.Unlock()
	}
}

// ClearPending clears all pending warm requests.
func (cw *CacheWarmer) ClearPending() {
	cw.mu.Lock()
	defer cw.mu.Unlock()
	cw.requests = make([]WarmRequest, 0)
}

// Stats returns cache warming statistics.
func (cw *CacheWarmer) Stats() WarmStats {
	cw.mu.Lock()
	defer cw.mu.Unlock()
	return cw.stats
}

// String returns a human-readable summary of the cache warming statistics.
func (ws WarmStats) String() string {
	return fmt.Sprintf("WarmStats{total_requests: %d, total_warmed: %d, elapsed: %s}",
		ws.TotalRequests, ws.TotalWarmed, time.Since(ws.StartTime))
}
