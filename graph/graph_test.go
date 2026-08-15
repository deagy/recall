package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKnowledgeGraph_AddEntity(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)

	require.True(t, g.AddEntity(e1), "e1 should be added")
	require.True(t, g.AddEntity(e2), "e2 should be added")
	assert.Equal(t, 2, g.Count(), "expected 2 entities")

	assert.False(t, g.AddEntity(e1), "duplicate should be rejected")
}

func TestKnowledgeGraph_AddRelation(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	g.AddEntity(e1)
	g.AddEntity(e2)

	r := NewRelation("e1", "e2", "knows", 0.9)
	require.True(t, g.AddRelation(r), "relation should be added")
	assert.Equal(t, 1, g.RelationCount(), "expected 1 relation")

	r2 := NewRelation("e1", "nonexistent", "knows", 0.5)
	assert.False(t, g.AddRelation(r2), "relation to non-existent entity should fail")
}

func TestKnowledgeGraph_GetEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	e := NewEntity("e1", "Alice", EntityPerson)
	g.AddEntity(e)

	got, ok := g.GetEntity("e1")
	require.True(t, ok, "entity should be found")
	assert.Equal(t, "Alice", got.Label, "label should match")

	_, ok = g.GetEntity("nonexistent")
	assert.False(t, ok, "non-existent entity should not be found")
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
	require.Len(t, neighbors, 1, "expected 1 neighbor")
	assert.Equal(t, "e2", neighbors[0].ID, "expected neighbor e2")

	neighbors2 := g.Neighbors("e2")
	require.Len(t, neighbors2, 2, "expected 2 neighbors")
}

func TestKnowledgeGraph_FindPath(t *testing.T) {
	g := NewKnowledgeGraph()

	entities := []struct {
		id    string
		label string
		typ   EntityType
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
	require.NotNil(t, path, "expected path to exist")
	assert.Equal(t, 4, path.Length(), "expected path length 4")

	g2 := NewKnowledgeGraph()
	g2.AddEntity(NewEntity("a", "A", EntityOther))
	g2.AddEntity(NewEntity("b", "B", EntityOther))
	path2 := g2.FindPath("a", "b")
	assert.Nil(t, path2, "expected no path")
}

func TestKnowledgeGraph_TransitiveClosure(t *testing.T) {
	g := NewKnowledgeGraph()

	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.8))

	closure := g.TransitiveClosure("e1")
	require.Len(t, closure, 3, "expected 3 entities in transitive closure")
}

func TestKnowledgeGraph_CommonNeighbors(t *testing.T) {
	g := NewKnowledgeGraph()

	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))

	g.AddRelation(NewRelation("e1", "e3", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.6))

	common := g.CommonNeighbors("e1", "e2")
	require.Len(t, common, 1, "expected 1 common neighbor")
	assert.Equal(t, "e3", common[0].ID, "expected common neighbor e3")
}

func TestKnowledgeGraph_ShortestPathLength(t *testing.T) {
	g := NewKnowledgeGraph()

	entities := []struct {
		id    string
		label string
		typ   EntityType
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

	assert.Equal(t, 2, g.ShortestPathLength("e1", "e3"), "expected distance 2")
	assert.Equal(t, 0, g.ShortestPathLength("e1", "e1"), "expected distance 0")
	assert.Equal(t, -1, g.ShortestPathLength("e1", "nonexistent"), "expected distance -1")
}

func TestKnowledgeGraph_FindEntitiesByLabel(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice Smith", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob Jones", EntityPerson))
	g.AddEntity(NewEntity("e3", "Alice Johnson", EntityPerson))

	results := g.FindEntitiesByLabel("alice")
	require.Len(t, results, 2, "expected 2 results")
}

func TestKnowledgeGraph_FindEntitiesByType(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Go", EntityConcept))
	g.AddEntity(NewEntity("e3", "Bob", EntityPerson))

	results := g.FindEntitiesByType(EntityPerson)
	require.Len(t, results, 2, "expected 2 persons")
}

func TestKnowledgeGraph_RemoveEntity(t *testing.T) {
	g := NewKnowledgeGraph()

	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))

	require.True(t, g.RemoveEntity("e1"), "e1 should be removed")
	assert.Equal(t, 1, g.Count(), "expected 1 entity")
	assert.Equal(t, 0, g.RelationCount(), "expected 0 relations")
}

func TestEntityProperties(t *testing.T) {
	e := NewEntity("e1", "Alice", EntityPerson)
	e.SetProperty("age", "30")
	e.SetProperty("city", "NYC")

	assert.Equal(t, "30", e.GetProperty("age"), "age should match")
	assert.Equal(t, "NYC", e.GetProperty("city"), "city should match")
	assert.Empty(t, e.GetProperty("nonexistent"), "nonexistent property should be empty")
}

func TestRelationProperties(t *testing.T) {
	r := NewRelation("e1", "e2", "knows", 0.9)
	r.SetProperty("since", "2020")

	assert.Equal(t, "2020", r.GetProperty("since"), "since should match")
}

func TestPath_String(t *testing.T) {
	e1 := NewEntity("e1", "Alice", EntityPerson)
	e2 := NewEntity("e2", "Bob", EntityPerson)
	r := NewRelation("e1", "e2", "knows", 0.9)

	path := &Path{
		Entities:  []*Entity{e1, e2},
		Relations: []*Relation{r},
	}

	assert.Equal(t, "Alice --[knows]--> Bob", path.String(), "path string should match")
}
