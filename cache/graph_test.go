package cache

import (
	"testing"
	"time"

	"github.com/deagy/recall/graph"
)

func TestGraphCache_SetAndGet(t *testing.T) {
	gc := NewGraphCache(100)

	result := &GraphTraversalResult{
		Query: "find friends of alice",
		Results: []*graph.Entity{
			graph.NewEntity("bob", "Bob", graph.EntityPerson),
			graph.NewEntity("charlie", "Charlie", graph.EntityPerson),
		},
		Timestamp: time.Now(),
		Latency:   5 * time.Millisecond,
	}

	gc.Set("alice", "outgoing", 1, result, 5*time.Minute)

	cached, ok := gc.Get("alice", "outgoing", 1)
	if !ok {
		t.Fatal("Expected to find cached result")
	}
	if cached.Query != "find friends of alice" {
		t.Errorf("Expected query 'find friends of alice', got %q", cached.Query)
	}
	if len(cached.Results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(cached.Results))
	}
}

func TestGraphCache_GetNonExistent(t *testing.T) {
	gc := NewGraphCache(100)

	_, ok := gc.Get("nonexistent", "outgoing", 1)
	if ok {
		t.Error("Expected not to find nonexistent result")
	}
}

func TestGraphCache_Delete(t *testing.T) {
	gc := NewGraphCache(100)

	result := &GraphTraversalResult{
		Query:   "test",
		Results: []*graph.Entity{},
	}

	gc.Set("alice", "outgoing", 1, result, 5*time.Minute)
	gc.Delete("alice", "outgoing", 1)

	_, ok := gc.Get("alice", "outgoing", 1)
	if ok {
		t.Error("Expected not to find deleted result")
	}
}

func TestGraphCache_Clear(t *testing.T) {
	gc := NewGraphCache(100)

	result := &GraphTraversalResult{
		Query:   "test",
		Results: []*graph.Entity{},
	}

	gc.Set("alice", "outgoing", 1, result, 5*time.Minute)
	gc.Clear()

	_, ok := gc.Get("alice", "outgoing", 1)
	if ok {
		t.Error("Expected not to find result after clear")
	}
}

func TestGraphCache_Stats(t *testing.T) {
	gc := NewGraphCache(100)

	result := &GraphTraversalResult{
		Query:   "test",
		Results: []*graph.Entity{},
	}

	gc.Set("alice", "outgoing", 1, result, 5*time.Minute)

	stats := gc.Stats()
	if stats.Size != 1 {
		t.Errorf("Expected size 1, got %d", stats.Size)
	}
}

func TestGenerateGraphTraversalKey(t *testing.T) {
	// Test with same parameters (should produce same key)
	key1 := GenerateGraphTraversalKey("alice", "outgoing", 1)
	key2 := GenerateGraphTraversalKey("alice", "outgoing", 1)

	if key1 != key2 {
		t.Errorf("Expected same key for same parameters, got %q and %q", key1, key2)
	}

	// Test with different entity (should produce different key)
	key3 := GenerateGraphTraversalKey("bob", "outgoing", 1)
	if key1 == key3 {
		t.Error("Expected different key for different entity")
	}

	// Test with different depth (should produce different key)
	key4 := GenerateGraphTraversalKey("alice", "outgoing", 2)
	if key1 == key4 {
		t.Error("Expected different key for different depth")
	}
}
