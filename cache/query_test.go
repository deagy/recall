package cache

import (
	"testing"
	"time"
)

func TestQueryCache_SetAndGet(t *testing.T) {
	qc := NewQueryCache(100)

	result := &QueryResult{
		Query:     "test query",
		Results:   []interface{}{"result1", "result2"},
		Timestamp: time.Now(),
		Latency:   10 * time.Millisecond,
	}

	qc.Set("test query", nil, result, 5*time.Minute)

	cached, ok := qc.Get("test query", nil)
	if !ok {
		t.Fatal("Expected to find cached result")
	}
	if cached.Query != "test query" {
		t.Errorf("Expected query 'test query', got %q", cached.Query)
	}
}

func TestQueryCache_GetNonExistent(t *testing.T) {
	qc := NewQueryCache(100)

	_, ok := qc.Get("nonexistent", nil)
	if ok {
		t.Error("Expected not to find nonexistent query")
	}
}

func TestQueryCache_Delete(t *testing.T) {
	qc := NewQueryCache(100)

	result := &QueryResult{
		Query:     "test query",
		Results:   []interface{}{"result1"},
		Timestamp: time.Now(),
	}

	qc.Set("test query", nil, result, 5*time.Minute)
	qc.Delete("test query", nil)

	_, ok := qc.Get("test query", nil)
	if ok {
		t.Error("Expected not to find deleted query")
	}
}

func TestQueryCache_Clear(t *testing.T) {
	qc := NewQueryCache(100)

	result := &QueryResult{
		Query:     "test query",
		Results:   []interface{}{"result1"},
		Timestamp: time.Now(),
	}

	qc.Set("test query", nil, result, 5*time.Minute)
	qc.Clear()

	_, ok := qc.Get("test query", nil)
	if ok {
		t.Error("Expected not to find query after clear")
	}
}

func TestQueryCache_Stats(t *testing.T) {
	qc := NewQueryCache(100)

	result := &QueryResult{
		Query:     "test query",
		Results:   []interface{}{"result1"},
		Timestamp: time.Now(),
	}

	qc.Set("test query", nil, result, 5*time.Minute)

	stats := qc.Stats()
	if stats.Size != 1 {
		t.Errorf("Expected size 1, got %d", stats.Size)
	}
}

func TestGenerateQueryKey(t *testing.T) {
	// Test with same query and filters (should produce same key)
	filters := map[string]interface{}{
		"limit":  10,
		"offset": 0,
	}

	key1 := GenerateQueryKey("test query", filters)
	key2 := GenerateQueryKey("test query", filters)

	if key1 != key2 {
		t.Errorf("Expected same key for same query and filters, got %q and %q", key1, key2)
	}

	// Test with different query (should produce different key)
	key3 := GenerateQueryKey("different query", filters)
	if key1 == key3 {
		t.Error("Expected different key for different query")
	}

	// Test with different filters (should produce different key)
	filters2 := map[string]interface{}{
		"limit":  20,
		"offset": 0,
	}
	key4 := GenerateQueryKey("test query", filters2)
	if key1 == key4 {
		t.Error("Expected different key for different filters")
	}
}
