package graph

import (
	"testing"
)

func BenchmarkKnowledgeGraph_AddEntity(b *testing.B) {
	g := NewKnowledgeGraph()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.AddEntity(NewEntity("entity"+string(rune('a'+i%26)), "Entity", EntityPerson))
	}
}

func BenchmarkKnowledgeGraph_AddRelation(b *testing.B) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("from", "From", EntityPerson))
	g.AddEntity(NewEntity("to", "To", EntityPerson))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.AddRelation(NewRelation("from", "to", "rel", 0.5))
	}
}

func BenchmarkKnowledgeGraph_FindPath(b *testing.B) {
	g := NewKnowledgeGraph()
	// Create a chain: a -> b -> c -> d -> e
	entities := []string{"a", "b", "c", "d", "e"}
	for _, id := range entities {
		g.AddEntity(NewEntity(id, id, EntityPerson))
	}
	for i := 0; i < len(entities)-1; i++ {
		g.AddRelation(NewRelation(entities[i], entities[i+1], "next", 1.0))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.FindPath("a", "e")
	}
}

func BenchmarkKnowledgeGraph_Neighbors(b *testing.B) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("center", "Center", EntityPerson))
	for i := 0; i < 100; i++ {
		id := string(rune('a' + i%26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		g.AddRelation(NewRelation("center", id, "connected", 1.0))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Neighbors("center")
	}
}

func BenchmarkKnowledgeGraph_TransitiveClosure(b *testing.B) {
	g := NewKnowledgeGraph()
	// Create a chain: a -> b -> c -> d -> e
	entities := []string{"a", "b", "c", "d", "e"}
	for _, id := range entities {
		g.AddEntity(NewEntity(id, id, EntityPerson))
	}
	for i := 0; i < len(entities)-1; i++ {
		g.AddRelation(NewRelation(entities[i], entities[i+1], "next", 1.0))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.TransitiveClosure("a")
	}
}

func BenchmarkKnowledgeGraph_CommonNeighbors(b *testing.B) {
	g := NewKnowledgeGraph()
	// Create a star graph: center connected to all
	g.AddEntity(NewEntity("c1", "C1", EntityPerson))
	g.AddEntity(NewEntity("c2", "C2", EntityPerson))
	for i := 0; i < 50; i++ {
		id := string(rune('a' + i%26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		g.AddRelation(NewRelation(id, "c1", "connected", 1.0))
		g.AddRelation(NewRelation(id, "c2", "connected", 1.0))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.CommonNeighbors("c1", "c2")
	}
}