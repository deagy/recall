package store

import (
	"testing"

	"github.com/deagy/recall/graph"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQLiteGraphStore_AddEntity(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	require.NoError(t, err)
	defer s.Close()

	e := graph.NewEntity("alice", "Alice", graph.EntityPerson)
	require.NoError(t, s.AddEntity(e))

	// Verify it's in the graph
	g := s.Graph()
	entities := g.FindEntitiesByLabel("alice")
	require.NotEmpty(t, entities, "expected entity to be in graph")
}

func TestSQLiteGraphStore_AddRelation(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	require.NoError(t, err)
	defer s.Close()

	// Add entities first
	require.NoError(t, s.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson)))
	require.NoError(t, s.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson)))

	rel := graph.NewRelation("alice", "bob", "knows", 0.9)
	require.NoError(t, s.AddRelation(rel))

	// Verify it's in the graph
	g := s.Graph()
	rels := g.Relations()
	require.NotEmpty(t, rels, "expected relation to be in graph")
}

func TestSQLiteGraphStore_Persistence(t *testing.T) {
	// Create store, add data, close, reopen, verify data
	tmpFile := t.TempDir() + "/test.db"

	s1, err := NewSQLiteGraphStore(tmpFile)
	require.NoError(t, err)

	s1.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	s1.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
	s1.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	s1.Close()

	// Reopen
	s2, err := NewSQLiteGraphStore(tmpFile)
	require.NoError(t, err)
	defer s2.Close()

	require.NoError(t, s2.LoadFromDB())

	g := s2.Graph()
	entities := g.FindEntitiesByLabel("alice")
	require.NotEmpty(t, entities, "expected alice to be loaded from DB")

	rels := g.Relations()
	require.NotEmpty(t, rels, "expected knows relation to be loaded from DB")
}

func TestSQLiteGraphStore_Clear(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	require.NoError(t, err)
	defer s.Close()

	s.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	require.NoError(t, s.Clear())

	g := s.Graph()
	entities := g.FindEntitiesByLabel("alice")
	assert.Empty(t, entities, "expected no entities after clear")
}

func TestSQLiteGraphStore_EntityWithProperties(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	require.NoError(t, err)
	defer s.Close()

	e := graph.NewEntity("alice", "Alice", graph.EntityPerson)
	e.SetProperty("age", "30")
	e.SetProperty("city", "SF")

	require.NoError(t, s.AddEntity(e))

	// Reload and verify properties
	require.NoError(t, s.LoadFromDB())

	g := s.Graph()
	entities := g.FindEntitiesByLabel("alice")
	require.Len(t, entities, 1, "expected 1 entity")
	assert.Equal(t, "30", entities[0].GetProperty("age"), "age should be 30")
	assert.Equal(t, "SF", entities[0].GetProperty("city"), "city should be SF")
}

func TestSQLiteGraphStore_RelationWithProperties(t *testing.T) {
	s, err := NewSQLiteGraphStore(":memory:")
	require.NoError(t, err)
	defer s.Close()

	s.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	s.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))

	rel := graph.NewRelation("alice", "bob", "works_with", 0.8)
	rel.Properties["since"] = "2020"

	require.NoError(t, s.AddRelation(rel))

	// Reload and verify
	g := s.Graph()
	rels := g.Relations()
	require.Len(t, rels, 1, "expected 1 relation")
	assert.Equal(t, "2020", rels[0].GetProperty("since"), "since should be 2020")
}

// Ensure SQLiteGraphStore implements GraphPersistence.
var _ GraphPersistence = (*SQLiteGraphStore)(nil)
