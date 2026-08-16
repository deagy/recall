package cache

import (
	"testing"
	"time"
)

func TestMultiLevelCache_GetFromL1(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	// Set in L1 only
	mlc.L1.Set("key1", "value1", 0)

	// Should find in L1
	val, ok := mlc.Get("key1")
	if !ok {
		t.Fatal("Expected to find key1")
	}
	if val != "value1" {
		t.Errorf("Expected value1, got %v", val)
	}
}

func TestMultiLevelCache_GetFromL2(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	// Set in L2 only
	mlc.L2.Set("key1", "value1", 0)

	// Should find in L2 and promote to L1
	val, ok := mlc.Get("key1")
	if !ok {
		t.Fatal("Expected to find key1")
	}
	if val != "value1" {
		t.Errorf("Expected value1, got %v", val)
	}

	// Should also be in L1 now (promoted)
	_, ok = mlc.L1.Get("key1")
	if !ok {
		t.Error("Expected key1 to be promoted to L1")
	}
}

func TestMultiLevelCache_GetNonExistent(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	_, ok := mlc.Get("nonexistent")
	if ok {
		t.Error("Expected not to find nonexistent key")
	}
}

func TestMultiLevelCache_Set(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	mlc.Set("key1", "value1", 10*time.Minute)

	// Should be in both L1 and L2
	_, ok := mlc.L1.Get("key1")
	if !ok {
		t.Error("Expected key1 in L1")
	}

	_, ok = mlc.L2.Get("key1")
	if !ok {
		t.Error("Expected key1 in L2")
	}
}

func TestMultiLevelCache_Delete(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	mlc.Set("key1", "value1", 0)
	mlc.Delete("key1")

	_, ok := mlc.Get("key1")
	if ok {
		t.Error("Expected not to find deleted key")
	}
}

func TestMultiLevelCache_Clear(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	mlc.Set("key1", "value1", 0)
	mlc.Clear()

	_, ok := mlc.Get("key1")
	if ok {
		t.Error("Expected not to find key after clear")
	}
}

func TestMultiLevelCache_Stats(t *testing.T) {
	l1 := NewLRUCache(DefaultCacheConfig())
	l2 := NewLRUCache(DefaultCacheConfig())

	mlc := NewMultiLevelCache(l1, l2)

	mlc.Set("key1", "value1", 0)
	mlc.Set("key2", "value2", 0)

	stats := mlc.Stats()
	// Note: Set adds to both L1 and L2, so total size is 4 (2 in L1 + 2 in L2)
	if stats.Size != 4 {
		t.Errorf("Expected size 4 (2 in L1 + 2 in L2), got %d", stats.Size)
	}
}
