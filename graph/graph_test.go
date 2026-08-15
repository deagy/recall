package graph

import (
	"testing"
)

func TestKnowledgeGraph_AddEntity(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)

	if !g.AddEntity(e1) {
		t.Fatal("expected e1 to be added")
	}
	if !g.AddEntity(e2) {
		t.Fatal("expected e2 to be added")
	}
	if g.Count() != 2 {
		t.Fatalf("expected 2 entities, got %d", g.Count())
	}

	if g.AddEntity(e1) {
		t.Fatal("expected duplicate to be rejected")
	}
}

func TestKnowledgeGraph_AddRelation(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	g.AddEntity(e1)
	g.AddEntity(e2)

	r := NewRelation("e1", "e2", "knows", 0.9)
	if !g.AddRelation(r) {
		t.Fatal("expected relation to be added")
	}
	if g.RelationCount() != 1 {
		t.Fatalf("expected 1 relation, got %d", g.RelationCount())
	}

	r2 := NewRelation("e1", "nonexistent", "knows", 0.5)
	if g.AddRelation(r2) {
		t.Fatal("expected relation to non-existent entity to fail")
	}
}

func TestKnowledgeGraph_GetEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	e := NewEntity("e1", "Alice", EntityPerson)
	g.AddEntity(e)

	got, ok := g.GetEntity("e1")
	if !ok {
		t.Fatal("expected entity to be found")
	}
	if got.Label != "Alice" {
		t.Fatalf("expected label 'Alice', got '%s'", got.Label)
	}

	_, ok = g.GetEntity("nonexistent")
	if ok {
		t.Fatal("expected non-existent entity to not be found")
	}
}

func TestKnowledgeGraph_Neighbors(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	e3 := NewEntity("e3", "Charlie", EntityPerson)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddEntity(e3)

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.8))

	neighbors := g.Neighbors("e1")
	if len(neighbors) != 1 {
		t.Fatalf("expected 1 neighbor, got %d", len(neighbors))
	}
	if neighbors[0].ID != "e2" {
		t.Fatalf("expected neighbor e2, got %s", neighbors[0].ID)
	}

	neighbors2 := g.Neighbors("e2")
	if len(neighbors2) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(neighbors2))
	}
}

func TestKnowledgeGraph_FindPath(t *testing.T) {
	g := NewKnowledgeGraph()

	entities := []struct {
		id   string
		label string
		typ  EntityType
	}{
		{"e1", "Alice", EntityPerson},
		{"e2", "Bob", EntityPerson},
		{"e3", "Charlie", EntityPerson},
		{"e4", "Dave", EntityPerson},
	}
	for _, e := range entities {
		g.AddEntity(NewEntity(e.id, e.label, e.typ))
	}

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.8))
	g.AddRelation(NewRelation("e3", "e4", "knows", 0.7))

	path := g.FindPath("e1", "e4")
	if path == nil {
		t.Fatal("expected path to exist")
	}
	if path.Length() != 4 {
		t.Fatalf("expected path length 4, got %d", path.Length())
	}

	g2 := NewKnowledgeGraph()
	g2.AddEntity(NewEntity("a", "A", EntityOther))
	g2.AddEntity(NewEntity("b", "B", EntityOther))
	path2 := g2.FindPath("a", "b")
	if path2 != nil {
		t.Fatal("expected no path")
	}
}

func TestKnowledgeGraph_TransitiveClosure(t *testing.T) {
	g := NewKnowledgeGraph()

	entities := []struct {
		id   string
		label string
		typ  EntityType
	}{
		{"e1", "Alice", EntityPerson},
		{"e2", "Bob", EntityPerson},
		{"e3", "Charlie", EntityPerson},
	}
	for _, e := range entities {
		g.AddEntity(NewEntity(e.id, e.label, e.typ))
	}

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.8))

	closure := g.TransitiveClosure("e1")
	if len(closure) != 3 {
		t.Fatalf("expected 3 entities in closure, got %d", len(closure))
	}
}

func TestKnowledgeGraph_CommonNeighbors(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	e3 := NewEntity("e3", "Charlie", EntityPerson)
	e4 := NewEntity("e4", "Dave", EntityPerson)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddEntity(e3)
	g.AddEntity(e4)

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e1", "e3", "knows", 0.8))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.7))
	g.AddRelation(NewRelation("e2", "e4", "knows", 0.6))

	common := g.CommonNeighbors("e1", "e2")
	if len(common) != 1 {
		t.Fatalf("expected 1 common neighbor, got %d", len(common))
	}
	if common[0].ID != "e3" {
		t.Fatalf("expected common neighbor e3, got %s", common[0].ID)
	}
}

func TestKnowledgeGraph_ShortestPathLength(t *testing.T) {
	g := NewKnowledgeGraph()

	entities := []struct {
		id   string
		label string
		typ  EntityType
	}{
		{"e1", "Alice", EntityPerson},
		{"e2", "Bob", EntityPerson},
		{"e3", "Charlie", EntityPerson},
	}
	for _, e := range entities {
		g.AddEntity(NewEntity(e.id, e.label, e.typ))
	}

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.8))

	if dist := g.ShortestPathLength("e1", "e3"); dist != 2 {
		t.Fatalf("expected distance 2, got %d", dist)
	}
	if dist := g.ShortestPathLength("e1", "e1"); dist != 0 {
		t.Fatalf("expected distance 0, got %d", dist)
	}
	if dist := g.ShortestPathLength("e1", "nonexistent"); dist != -1 {
		t.Fatalf("expected distance -1, got %d", dist)
	}
}

func TestKnowledgeGraph_FindEntitiesByLabel(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice Smith", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob Jones", EntityPerson))
	g.AddEntity(NewEntity("e3", "Alice Johnson", EntityPerson))

	results := g.FindEntitiesByLabel("alice")
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestKnowledgeGraph_FindEntitiesByType(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Go", EntityConcept))
	g.AddEntity(NewEntity("e3", "Bob", EntityPerson))

	results := g.FindEntitiesByType(EntityPerson)
	if len(results) != 2 {
		t.Fatalf("expected 2 persons, got %d", len(results))
	}
}

func TestKnowledgeGraph_RemoveEntity(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))

	if !g.RemoveEntity("e1") {
		t.Fatal("expected e1 to be removed")
	}
	if g.Count() != 1 {
		t.Fatalf("expected 1 entity, got %d", g.Count())
	}
	if g.RelationCount() != 0 {
		t.Fatalf("expected 0 relations, got %d", g.RelationCount())
	}
}

func TestEntityProperties(t *testing.T) {
	e := NewEntity("e1", "Alice", EntityPerson)
	e.SetProperty("age", "30")
	e.SetProperty("city", "NYC")

	if e.GetProperty("age") != "30" {
		t.Fatalf("expected '30', got '%s'", e.GetProperty("age"))
	}
	if e.GetProperty("nonexistent") != "" {
		t.Fatal("expected empty string for nonexistent property")
	}
}

func TestRelationProperties(t *testing.T) {
	r := NewRelation("e1", "e2", "knows", 0.9)
	r.SetProperty("since", "2020")

	if r.GetProperty("since") != "2020" {
		t.Fatalf("expected '2020', got '%s'", r.GetProperty("since"))
	}
}

func TestPath_String(t *testing.T) {
	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	r := NewRelation("e1", "e2", "knows", 0.9)

	path := &Path{
		Entities:  []*Entity{e1, e2},
		Relations: []*Relation{r},
	}

	s := path.String()
	if s != "Alice --[knows]--> Bob" {
		t.Fatalf("expected 'Alice --[knows]--> Bob', got '%s'", s)
	}
}