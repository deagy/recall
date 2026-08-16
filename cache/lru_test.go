package cache

import (
	"testing"
	"time"
)

func TestLRUCache_SetAndGet(t *testing.T) {
	config := DefaultCacheConfig()
	config.MaxSize = 100
	lru := NewLRUCache(config)

	// Set and get a value
	lru.Set("key1", "value1", 0)
	val, ok := lru.Get("key1")
	if !ok {
		t.Fatal("Expected to find key1")
	}
	if val != "value1" {
		t.Errorf("Expected value1, got %v", val)
	}
}

func TestLRUCache_GetNonExistent(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	_, ok := lru.Get("nonexistent")
	if ok {
		t.Error("Expected not to find nonexistent key")
	}
}

func TestLRUCache_Delete(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	lru.Set("key1", "value1", 0)
	lru.Delete("key1")

	_, ok := lru.Get("key1")
	if ok {
		t.Error("Expected not to find deleted key")
	}
}

func TestLRUCache_Clear(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	lru.Set("key1", "value1", 0)
	lru.Set("key2", "value2", 0)
	lru.Clear()

	_, ok := lru.Get("key1")
	if ok {
		t.Error("Expected not to find key1 after clear")
	}

	_, ok = lru.Get("key2")
	if ok {
		t.Error("Expected not to find key2 after clear")
	}
}

func TestLRUCache_Stats(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	lru.Set("key1", "value1", 0)
	lru.Set("key2", "value2", 0)

	stats := lru.Stats()
	if stats.Size != 2 {
		t.Errorf("Expected size 2, got %d", stats.Size)
	}
}

func TestLRUCache_Eviction(t *testing.T) {
	config := DefaultCacheConfig()
	config.MaxSize = 3
	lru := NewLRUCache(config)

	// Fill cache to capacity
	lru.Set("key1", "value1", 0)
	lru.Set("key2", "value2", 0)
	lru.Set("key3", "value3", 0)

	// Access key1 to make it recently used
	lru.Get("key1")

	// Add new key, should evict key2 (least recently used)
	lru.Set("key4", "value4", 0)

	// key1 should still exist (recently used)
	_, ok := lru.Get("key1")
	if !ok {
		t.Error("Expected key1 to still exist")
	}

	// key2 should be evicted (least recently used)
	_, ok = lru.Get("key2")
	if ok {
		t.Error("Expected key2 to be evicted")
	}

	// key3 should still exist
	_, ok = lru.Get("key3")
	if !ok {
		t.Error("Expected key3 to still exist")
	}

	// key4 should exist
	_, ok = lru.Get("key4")
	if !ok {
		t.Error("Expected key4 to exist")
	}
}

func TestLRUCache_ExpiredEntry(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	// Set with 1ms TTL
	lru.Set("key1", "value1", 1*time.Millisecond)

	// Wait for expiration
	time.Sleep(10 * time.Millisecond)

	// Should be expired
	_, ok := lru.Get("key1")
	if ok {
		t.Error("Expected expired key to not be found")
	}
}

func TestLRUCache_UpdateExisting(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	lru.Set("key1", "value1", 0)
	lru.Set("key1", "value2", 0)

	val, ok := lru.Get("key1")
	if !ok {
		t.Fatal("Expected to find key1")
	}
	if val != "value2" {
		t.Errorf("Expected value2, got %v", val)
	}
}

func TestLRUCache_Len(t *testing.T) {
	config := DefaultCacheConfig()
	lru := NewLRUCache(config)

	if lru.Len() != 0 {
		t.Errorf("Expected empty cache, got size %d", lru.Len())
	}

	lru.Set("key1", "value1", 0)
	if lru.Len() != 1 {
		t.Errorf("Expected size 1, got %d", lru.Len())
	}

	lru.Set("key2", "value2", 0)
	if lru.Len() != 2 {
		t.Errorf("Expected size 2, got %d", lru.Len())
	}
}

func TestLRUCache_DefaultConfig(t *testing.T) {
	// Test with zero config (should use defaults)
	lru := NewLRUCache(CacheConfig{})

	lru.Set("key1", "value1", 0)
	val, ok := lru.Get("key1")
	if !ok {
		t.Fatal("Expected to find key1")
	}
	if val != "value1" {
		t.Errorf("Expected value1, got %v", val)
	}
}
