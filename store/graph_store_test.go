package store

import (
	"context"
	"testing"

	"github.com/deagy/recall/graph"
)

func TestMemoryGraphStore_ExtractEntities(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	entities, err := s.ExtractEntities(ctx, "Alice works at Google in Mountain View", "chunk-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(entities) == 0 {
		t.Fatal("expected entities to be extracted")
	}

	// Check that Alice was extracted
	found := false
	for _, e := range entities {
		if e.ID == "alice" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected 'alice' entity to be extracted")
	}

	// Count should reflect extracted entities
	if s.Count() == 0 {
		t.Fatal("expected graph to have entities")
	}
}

func TestMemoryGraphStore_ExtractRelations(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	// First extract entities
	s.ExtractEntities(ctx, "Alice works at Google", "chunk-1")
	s.ExtractEntities(ctx, "Bob works at Microsoft", "chunk-2")

	// Use text with adjacent capitalized words for relation extraction
	relations, err := s.ExtractRelations(ctx, "Alice Google Bob Microsoft", "chunk-3")
	if err != nil {
		t.Fatal(err)
	}
	if len(relations) == 0 {
		t.Fatal("expected relations to be extracted")
	}
}

func TestMemoryGraphStore_FindPath(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice knows Bob", "chunk-1")
	s.ExtractEntities(ctx, "Bob knows Charlie", "chunk-2")

	// Manually add a relation for testing
	s.Graph().AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s.Graph().AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.8))

	path := s.FindPath("alice", "charlie")
	if path == nil {
		t.Fatal("expected path to exist")
	}
	if path.Length() != 3 {
		t.Fatalf("expected path length 3, got %d", path.Length())
	}
}

func TestMemoryGraphStore_TransitiveClosure(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Bob Charlie", "chunk-1")
	s.Graph().AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s.Graph().AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.8))

	closure := s.TransitiveClosure("alice")
	if len(closure) != 3 {
		t.Fatalf("expected 3 entities in closure, got %d", len(closure))
	}
}

func TestMemoryGraphStore_CommonNeighbors(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Bob Charlie Dave", "chunk-1")
	s.Graph().AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s.Graph().AddRelation(graph.NewRelation("alice", "charlie", "knows", 0.8))
	s.Graph().AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.7))
	s.Graph().AddRelation(graph.NewRelation("bob", "dave", "knows", 0.6))

	common := s.CommonNeighbors("alice", "bob")
	if len(common) != 1 {
		t.Fatalf("expected 1 common neighbor, got %d", len(common))
	}
	if common[0].ID != "charlie" {
		t.Fatalf("expected common neighbor 'charlie', got '%s'", common[0].ID)
	}
}

func TestMemoryGraphStore_ShortestPathLength(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Bob Charlie", "chunk-1")
	s.Graph().AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s.Graph().AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.8))

	if dist := s.ShortestPathLength("alice", "charlie"); dist != 2 {
		t.Fatalf("expected distance 2, got %d", dist)
	}
	if dist := s.ShortestPathLength("alice", "alice"); dist != 0 {
		t.Fatalf("expected distance 0, got %d", dist)
	}
}

func TestMemoryGraphStore_FindEntitiesByLabel(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Smith Bob Jones Alice Johnson Charlie", "chunk-1")

	// "Alice" appears twice but deduplicates, so we get 1 result for "alice"
	// Use a broader query to match multiple entities
	results := s.FindEntitiesByLabel("smith")
	if len(results) != 1 {
		t.Fatalf("expected 1 result for 'smith', got %d", len(results))
	}
}

func TestMemoryGraphStore_FindEntitiesByType(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Go Bob", "chunk-1")

	results := s.FindEntitiesByType(graph.EntityOther)
	if len(results) == 0 {
		t.Fatal("expected entities of type 'other'")
	}
}

func TestMemoryGraphStore_Neighbors(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Bob Charlie", "chunk-1")
	s.Graph().AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s.Graph().AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.8))

	neighbors := s.Neighbors("alice")
	if len(neighbors) != 1 {
		t.Fatalf("expected 1 neighbor, got %d", len(neighbors))
	}
	if neighbors[0].ID != "bob" {
		t.Fatalf("expected neighbor 'bob', got '%s'", neighbors[0].ID)
	}
}

func TestMemoryGraphStore_GetEntity(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice", "chunk-1")

	e, ok := s.GetEntity("alice")
	if !ok {
		t.Fatal("expected entity to be found")
	}
	if e.Label != "Alice" {
		t.Fatalf("expected label 'Alice', got '%s'", e.Label)
	}

	_, ok = s.GetEntity("nonexistent")
	if ok {
		t.Fatal("expected non-existent entity to not be found")
	}
}

func TestMemoryGraphStore_Relations(t *testing.T) {
	s := NewMemoryGraphStore()
	ctx := context.Background()

	s.ExtractEntities(ctx, "Alice Bob", "chunk-1")
	s.Graph().AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))

	if s.RelationCount() != 1 {
		t.Fatalf("expected 1 relation, got %d", s.RelationCount())
	}

	rels := s.Relations()
	if len(rels) != 1 {
		t.Fatalf("expected 1 relation, got %d", len(rels))
	}
}
