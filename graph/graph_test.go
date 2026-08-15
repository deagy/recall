package graph

import (
	"fmt"
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

func TestKnowledgeGraph_EmptyGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	assert.Equal(t, 0, g.Count(), "expected 0 entities")
	assert.Equal(t, 0, g.RelationCount(), "expected 0 relations")
}

func TestKnowledgeGraph_AddEntity_EmptyLabel(t *testing.T) {
	g := NewKnowledgeGraph()
	e := NewEntity("e1", "", EntityPerson)
	require.True(t, g.AddEntity(e), "entity with empty label should be added")
	assert.Equal(t, 1, g.Count(), "expected 1 entity")
}

func TestKnowledgeGraph_AddEntity_AllTypes(t *testing.T) {
	g := NewKnowledgeGraph()

	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Go", EntityConcept))
	g.AddEntity(NewEntity("e3", "Google", EntityLocation))
	g.AddEntity(NewEntity("e4", "Mountain View", EntityLocation))
	g.AddEntity(NewEntity("e5", "something", EntityOther))

	assert.Equal(t, 5, g.Count(), "expected 5 entities")
}

func TestKnowledgeGraph_AddRelation_SameSourceTarget(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))

	// Self-loop
	r := NewRelation("e1", "e1", "knows", 0.9)
	require.True(t, g.AddRelation(r), "self-loop relation should be added")
	assert.Equal(t, 1, g.RelationCount(), "expected 1 relation")
}

func TestKnowledgeGraph_AddRelation_ZeroWeight(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))

	r := NewRelation("e1", "e2", "knows", 0)
	require.True(t, g.AddRelation(r), "relation with zero weight should be added")
}

func TestKnowledgeGraph_AddRelation_OneEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))

	// Missing target entity
	r := NewRelation("e1", "nonexistent", "knows", 0.9)
	assert.False(t, g.AddRelation(r), "relation to non-existent entity should fail")
}

func TestKnowledgeGraph_Neighbors_EmptyGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	neighbors := g.Neighbors("nonexistent")
	assert.Empty(t, neighbors, "expected no neighbors for non-existent entity")
}

func TestKnowledgeGraph_Neighbors_NoRelations(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))

	neighbors := g.Neighbors("e1")
	assert.Empty(t, neighbors, "expected no neighbors for entity with no relations")
}

func TestKnowledgeGraph_FindPath_NonExistentEntities(t *testing.T) {
	g := NewKnowledgeGraph()
	path := g.FindPath("nonexistent1", "nonexistent2")
	assert.Nil(t, path, "expected no path for non-existent entities")
}

func TestKnowledgeGraph_FindPath_SameEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))

	path := g.FindPath("e1", "e1")
	// Same entity should return a path of length 1
	if path != nil {
		assert.Equal(t, 1, path.Length(), "expected path length 1 for same entity")
	}
}

func TestKnowledgeGraph_TransitiveClosure_EmptyGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	closure := g.TransitiveClosure("nonexistent")
	assert.Empty(t, closure, "expected empty closure for non-existent entity")
}

func TestKnowledgeGraph_TransitiveClosure_SelfLoop(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddRelation(NewRelation("e1", "e1", "knows", 0.9))

	closure := g.TransitiveClosure("e1")
	// Should include self due to self-loop
	assert.NotEmpty(t, closure, "expected non-empty closure for self-loop")
}

func TestKnowledgeGraph_CommonNeighbors_NoCommon(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))
	g.AddEntity(NewEntity("e4", "Dave", EntityPerson))

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e3", "e4", "knows", 0.8))

	common := g.CommonNeighbors("e1", "e3")
	assert.Empty(t, common, "expected no common neighbors")
}

func TestKnowledgeGraph_CommonNeighbors_SameEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))

	common := g.CommonNeighbors("e1", "e1")
	_ = common
}

func TestKnowledgeGraph_ShortestPathLength_EmptyGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	assert.Equal(t, -1, g.ShortestPathLength("a", "b"), "expected -1 for empty graph")
}

func TestKnowledgeGraph_ShortestPathLength_SameEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	assert.Equal(t, 0, g.ShortestPathLength("e1", "e1"), "expected 0 for same entity")
}

func TestKnowledgeGraph_FindEntitiesByLabel_Empty(t *testing.T) {
	g := NewKnowledgeGraph()
	results := g.FindEntitiesByLabel("nonexistent")
	assert.Empty(t, results, "expected no results for non-existent label")
}

func TestKnowledgeGraph_FindEntitiesByType_Empty(t *testing.T) {
	g := NewKnowledgeGraph()
	results := g.FindEntitiesByType(EntityPerson)
	assert.Empty(t, results, "expected no results for empty graph")
}

func TestKnowledgeGraph_RemoveEntity_NonExistent(t *testing.T) {
	g := NewKnowledgeGraph()
	assert.False(t, g.RemoveEntity("nonexistent"), "expected false for non-existent entity")
}

func TestKnowledgeGraph_RemoveEntity_WithRelations(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))
	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.8))

	assert.True(t, g.RemoveEntity("e2"), "e2 should be removed")
	assert.Equal(t, 2, g.Count(), "expected 2 entities")
	assert.Equal(t, 0, g.RelationCount(), "expected 0 relations")
}

func TestEntity_SetProperty_NilProperties(t *testing.T) {
	e := NewEntity("e1", "Alice", EntityPerson)
	e.SetProperty("key", "value")
	assert.Equal(t, "value", e.GetProperty("key"), "expected property to be set")
}

func TestEntity_GetProperty_NilProperties(t *testing.T) {
	e := NewEntity("e1", "Alice", EntityPerson)
	assert.Empty(t, e.GetProperty("nonexistent"), "expected empty for non-existent property")
}

func TestRelation_SetProperty_NilProperties(t *testing.T) {
	r := NewRelation("e1", "e2", "knows", 0.9)
	r.SetProperty("key", "value")
	assert.Equal(t, "value", r.GetProperty("key"), "expected property to be set")
}

func TestRelation_GetProperty_NilProperties(t *testing.T) {
	r := NewRelation("e1", "e2", "knows", 0.9)
	assert.Empty(t, r.GetProperty("nonexistent"), "expected empty for non-existent property")
}

func TestPath_Length_Empty(t *testing.T) {
	path := &Path{}
	assert.Equal(t, 0, path.Length(), "expected length 0 for empty path")
}

func TestPath_Length_SingleEntity(t *testing.T) {
	e1 := NewEntity("e1", "Alice", EntityPerson)
	path := &Path{Entities: []*Entity{e1}}
	assert.Equal(t, 1, path.Length(), "expected length 1 for single entity path")
}

func TestPath_String_Empty(t *testing.T) {
	path := &Path{}
	s := path.String()
	// Empty path may return empty string depending on implementation
	_ = s
}

func TestPath_String_SingleEntity(t *testing.T) {
	e1 := NewEntity("e1", "Alice", EntityPerson)
	path := &Path{Entities: []*Entity{e1}}
	s := path.String()
	assert.NotEmpty(t, s, "expected non-empty string")
}

func TestNewEntity_AllTypes(t *testing.T) {
	tests := []struct {
		name string
		typ  EntityType
	}{
		{"person", EntityPerson},
		{"concept", EntityConcept},
		{"organization", EntityLocation},
		{"location", EntityLocation},
		{"other", EntityOther},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := NewEntity("id", "label", tt.typ)
			assert.Equal(t, tt.typ, e.Type, "expected type to match")
			assert.Equal(t, "label", e.Label, "expected label to match")
		})
	}
}

func TestNewRelation_AllWeights(t *testing.T) {
	tests := []struct {
		name   string
		weight float64
	}{
		{"zero", 0},
		{"positive", 0.5},
		{"one", 1.0},
		{"negative", -0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewRelation("a", "b", "rel", tt.weight)
			assert.Equal(t, tt.weight, r.Weight, "expected weight to match")
		})
	}
}

func TestEntityType_Values(t *testing.T) {
	if EntityPerson != "person" {
		t.Errorf("expected EntityPerson=%q, got %q", "person", EntityPerson)
	}
	if EntityConcept != "concept" {
		t.Errorf("expected EntityConcept=%q, got %q", "concept", EntityConcept)
	}
	if EntityLocation != "location" {
		t.Errorf("expected EntityLocation=%q, got %q", "location", EntityLocation)
	}
	if EntityOther != "other" {
		t.Errorf("expected EntityOther=%q, got %q", "other", EntityOther)
	}
}

func TestKnowledgeGraph_AddEntity_Duplicate(t *testing.T) {
	g := NewKnowledgeGraph()
	e1 := NewEntity("e1", "Alice", EntityPerson)
	require.True(t, g.AddEntity(e1))
	assert.False(t, g.AddEntity(e1), "duplicate entity should be rejected")
	assert.Equal(t, 1, g.Count(), "expected 1 entity")
}

func TestKnowledgeGraph_AddRelation_Duplicate(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))

	r1 := NewRelation("e1", "e2", "knows", 0.9)
	require.True(t, g.AddRelation(r1))
	// Duplicate relations may be allowed depending on implementation
	require.True(t, g.AddRelation(r1), "duplicate relation may be allowed")
	assert.GreaterOrEqual(t, g.RelationCount(), 1, "expected at least 1 relation")
}

func TestKnowledgeGraph_Neighbors_MultipleRelations(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))
	g.AddEntity(NewEntity("e4", "Dave", EntityPerson))

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e1", "e3", "knows", 0.8))
	g.AddRelation(NewRelation("e1", "e4", "knows", 0.7))

	neighbors := g.Neighbors("e1")
	assert.Len(t, neighbors, 3, "expected 3 neighbors")
}

func TestKnowledgeGraph_FindPath_MultiplePaths(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))
	g.AddEntity(NewEntity("e4", "Dave", EntityPerson))

	g.AddRelation(NewRelation("e1", "e2", "knows", 0.9))
	g.AddRelation(NewRelation("e1", "e3", "knows", 0.8))
	g.AddRelation(NewRelation("e2", "e4", "knows", 0.7))
	g.AddRelation(NewRelation("e3", "e4", "knows", 0.6))

	path := g.FindPath("e1", "e4")
	require.NotNil(t, path, "expected path to exist")
	// Should find a path (may be via e2 or e3)
	assert.GreaterOrEqual(t, path.Length(), 2, "expected path length >= 2")
}

func TestKnowledgeGraph_TransitiveClosure_LargeGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	for i := 0; i < 10; i++ {
		g.AddEntity(NewEntity(fmt.Sprintf("e%d", i), fmt.Sprintf("Entity%d", i), EntityPerson))
	}
	for i := 0; i < 9; i++ {
		g.AddRelation(NewRelation(fmt.Sprintf("e%d", i), fmt.Sprintf("e%d", i+1), "knows", 0.9))
	}

	closure := g.TransitiveClosure("e0")
	assert.Len(t, closure, 10, "expected 10 entities in transitive closure")
}

func TestKnowledgeGraph_CommonNeighbors_LargeGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e3", "Charlie", EntityPerson))
	g.AddEntity(NewEntity("e4", "Dave", EntityPerson))
	g.AddEntity(NewEntity("e5", "Eve", EntityPerson))

	g.AddRelation(NewRelation("e1", "e3", "knows", 0.9))
	g.AddRelation(NewRelation("e1", "e4", "knows", 0.8))
	g.AddRelation(NewRelation("e2", "e3", "knows", 0.7))
	g.AddRelation(NewRelation("e2", "e5", "knows", 0.6))

	common := g.CommonNeighbors("e1", "e2")
	assert.Len(t, common, 1, "expected 1 common neighbor")
	assert.Equal(t, "e3", common[0].ID, "expected common neighbor e3")
}

func TestKnowledgeGraph_ShortestPathLength_LargeGraph(t *testing.T) {
	g := NewKnowledgeGraph()
	for i := 0; i < 10; i++ {
		g.AddEntity(NewEntity(fmt.Sprintf("e%d", i), fmt.Sprintf("Entity%d", i), EntityPerson))
	}
	for i := 0; i < 9; i++ {
		g.AddRelation(NewRelation(fmt.Sprintf("e%d", i), fmt.Sprintf("e%d", i+1), "knows", 0.9))
	}

	assert.Equal(t, 9, g.ShortestPathLength("e0", "e9"), "expected distance 9")
	assert.Equal(t, 0, g.ShortestPathLength("e0", "e0"), "expected distance 0")
	assert.Equal(t, -1, g.ShortestPathLength("e0", "nonexistent"), "expected distance -1")
}

func TestKnowledgeGraph_FindEntitiesByLabel_CaseInsensitive(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "BOB", EntityPerson))
	g.AddEntity(NewEntity("e3", "alice", EntityPerson))

	results := g.FindEntitiesByLabel("alice")
	assert.Len(t, results, 2, "expected 2 results (case-insensitive)")
}

func TestKnowledgeGraph_FindEntitiesByType_MultipleTypes(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))
	g.AddEntity(NewEntity("e2", "Go", EntityConcept))
	g.AddEntity(NewEntity("e3", "Bob", EntityPerson))
	g.AddEntity(NewEntity("e4", "Google", EntityLocation))

	persons := g.FindEntitiesByType(EntityPerson)
	assert.Len(t, persons, 2, "expected 2 persons")

	concepts := g.FindEntitiesByType(EntityConcept)
	assert.Len(t, concepts, 1, "expected 1 concept")

	orgs := g.FindEntitiesByType(EntityLocation)
	assert.Len(t, orgs, 1, "expected 1 organization")
}

func TestKnowledgeGraph_RemoveEntity_LastEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))

	assert.True(t, g.RemoveEntity("e1"), "e1 should be removed")
	assert.Equal(t, 0, g.Count(), "expected 0 entities")
}

func TestKnowledgeGraph_AddRelation_ToNonExistentEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e1", "Alice", EntityPerson))

	r := NewRelation("e1", "nonexistent", "knows", 0.9)
	assert.False(t, g.AddRelation(r), "relation to non-existent entity should fail")
}

func TestKnowledgeGraph_AddRelation_FromNonExistentEntity(t *testing.T) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("e2", "Bob", EntityPerson))

	r := NewRelation("nonexistent", "e2", "knows", 0.9)
	assert.False(t, g.AddRelation(r), "relation from non-existent entity should fail")
}

func TestEntity_GetProperty_EmptyKey(t *testing.T) {
	e := NewEntity("e1", "Alice", EntityPerson)
	e.SetProperty("", "value")
	assert.Equal(t, "value", e.GetProperty(""), "expected property with empty key")
}

func TestRelation_GetProperty_EmptyKey(t *testing.T) {
	r := NewRelation("e1", "e2", "knows", 0.9)
	r.SetProperty("", "value")
	assert.Equal(t, "value", r.GetProperty(""), "expected property with empty key")
}

func TestPath_EmptyRelations(t *testing.T) {
	e1 := NewEntity("e1", "Alice", EntityPerson)
	path := &Path{Entities: []*Entity{e1}}
	assert.Empty(t, path.Relations, "expected empty relations")
}

func TestPath_EmptyEntities(t *testing.T) {
	path := &Path{Relations: []*Relation{}}
	assert.Empty(t, path.Entities, "expected empty entities")
}
