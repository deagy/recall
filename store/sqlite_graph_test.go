package store

import (
	"testing"

	"github.com/deagy/recall/graph"
)

func TestSQLiteGraphStore_AddEntity(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	e := graph.NewEntity("alice", "Alice", graph.EntityPerson)
	if err := s.AddEntity(e); err != nil {
		t.Fatal(err)
	}

	// Verify it's in the graph
	g := s.Graph()
	entities := g.FindEntitiesByLabel("alice")
	if len(entities) == 0 {
		t.Fatal("expected entity to be in graph")
	}
}

func TestSQLiteGraphStore_AddRelation(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// Add entities first
	s.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	s.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))

	rel := graph.NewRelation("alice", "bob", "knows", 0.9)
	if err := s.AddRelation(rel); err != nil {
		t.Fatal(err)
	}

	// Verify it's in the graph
	g := s.Graph()
	rels := g.Relations()
	if len(rels) == 0 {
		t.Fatal("expected relation to be in graph")
	}
}

func TestSQLiteGraphStore_Persistence(t *testing.T) {
	// Create store, add data, close, reopen, verify data
	tmpFile := t.TempDir() + "/test.db"

	s1, err := NewSQLiteGraphStore(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	s1.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	s1.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
	s1.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s1.Close()

	// Reopen
	s2, err := NewSQLiteGraphStore(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()

	if err := s2.LoadFromDB(); err != nil {
		t.Fatal(err)
	}

	g := s2.Graph()
	entities := g.FindEntitiesByLabel("alice")
	if len(entities) == 0 {
		t.Fatal("expected alice to be loaded from DB")
	}

	rels := g.Relations()
	if len(rels) == 0 {
		t.Fatal("expected knows relation to be loaded from DB")
	}
}

func TestSQLiteGraphStore_Clear(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	s.Clear()

	g := s.Graph()
	entities := g.FindEntitiesByLabel("alice")
	if len(entities) != 0 {
		t.Fatal("expected no entities after clear")
	}
}

func TestSQLiteGraphStore_EntityWithProperties(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	e := graph.NewEntity("alice", "Alice", graph.EntityPerson)
	e.SetProperty("age", "30")
	e.SetProperty("city", "SF")

	if err := s.AddEntity(e); err != nil {
		t.Fatal(err)
	}

	// Reload and verify properties
	if err := s.LoadFromDB(); err != nil {
		t.Fatal(err)
	}

	g := s.Graph()
	entities := g.FindEntitiesByLabel("alice")
	if len(entities) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(entities))
	}
	if entities[0].GetProperty("age") != "30" {
		t.Fatalf("expected age=30, got %s", entities[0].GetProperty("age"))
	}
}

func TestSQLiteGraphStore_RelationWithProperties(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	s.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))

	rel := graph.NewRelation("alice", "bob", "works_with", 0.8)
	rel.Properties["since"] = "2020"

	if err := s.AddRelation(rel); err != nil {
		t.Fatal(err)
	}

	// Reload and verify
	g := s.Graph()
	rels := g.Relations()
	if len(rels) != 1 {
		t.Fatalf("expected 1 relation, got %d", len(rels))
	}
	if rels[0].GetProperty("since") != "2020" {
		t.Fatalf("expected since=2020, got %s", rels[0].GetProperty("since"))
	}
}

// Ensure SQLiteGraphStore implements GraphPersistence.
var _ GraphPersistence = (*SQLiteGraphStore)(nil)