package cache

import (
	"container/list"
	"sync"
	"time"
)

// LRUCache implements a Least Recently Used (LRU) cache with TTL support.
type LRUCache struct {
	mu       sync.Mutex
	config   CacheConfig
	items    map[string]*list.Element
	order    *list.List // Most recently used at front
	capacity int
}

type lruEntry struct {
	key   string
	entry *CacheEntry
}

// NewLRUCache creates a new LRU cache with the given configuration.
func NewLRUCache(config CacheConfig) *LRUCache {
	if config.MaxSize <= 0 {
		config.MaxSize = 10000
	}
	if config.DefaultTTL <= 0 {
		config.DefaultTTL = 5 * time.Minute
	}

	return &LRUCache{
		config:   config,
		items:    make(map[string]*list.Element),
		order:    list.New(),
		capacity: config.MaxSize,
	}
}

// Get retrieves a value from the cache by key.
func (c *LRUCache) Get(key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.items[key]
	if !ok {
		return nil, false
	}

	entry := elem.Value.(*lruEntry)

	// Check if expired
	if entry.entry.IsExpired() {
		c.removeElement(elem)
		return nil, false
	}

	// Move to front (most recently used)
	c.order.MoveToFront(elem)
	entry.entry.LastAccessed = time.Now()

	return entry.entry.Value, true
}

// Set stores a value in the cache with the given key and TTL.
func (c *LRUCache) Set(key string, value interface{}, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// If key exists, update it
	if elem, ok := c.items[key]; ok {
		entry := elem.Value.(*lruEntry)
		entry.entry.Value = value
		if ttl > 0 {
			entry.entry.ExpiresAt = time.Now().Add(ttl)
		} else {
			entry.entry.ExpiresAt = time.Now().Add(c.config.DefaultTTL)
		}
		entry.entry.LastAccessed = time.Now()
		c.order.MoveToFront(elem)
		return
	}

	// Evict if at capacity
	for c.order.Len() >= c.capacity {
		c.evict()
	}

	// Create new entry
	expiresAt := time.Now().Add(c.config.DefaultTTL)
	if ttl > 0 {
		expiresAt = time.Now().Add(ttl)
	}

	entry := &CacheEntry{
		Value:        value,
		ExpiresAt:    expiresAt,
		CreatedAt:    time.Now(),
		LastAccessed: time.Now(),
	}

	lruEntry := &lruEntry{
		key:   key,
		entry: entry,
	}

	elem := c.order.PushFront(lruEntry)
	c.items[key] = elem
}

// Delete removes a value from the cache by key.
func (c *LRUCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[key]; ok {
		c.removeElement(elem)
	}
}

// Clear removes all values from the cache.
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items = make(map[string]*list.Element)
	c.order = list.New()
}

// Stats returns cache statistics.
func (c *LRUCache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	return CacheStats{
		Size: c.order.Len(),
	}
}

// evict removes the least recently used entry from the cache.
func (c *LRUCache) evict() {
	elem := c.order.Back()
	if elem != nil {
		c.removeElement(elem)
	}
}

// removeElement removes an element from both the map and the list.
func (c *LRUCache) removeElement(elem *list.Element) {
	c.order.Remove(elem)
	entry := elem.Value.(*lruEntry)
	delete(c.items, entry.key)
}

// Len returns the number of items in the cache.
func (c *LRUCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}
