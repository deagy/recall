package graph

import (
	"testing"
)

func BenchmarkKnowledgeGraph_DisconnectedComponents(b *testing.B) {
	g := NewKnowledgeGraph()
	// Create 10 disconnected components of 10 entities each
	for comp := 0; comp < 10; comp++ {
		for i := 0; i < 10; i++ {
			id := string(rune('a' + comp*10 + i))
			g.AddEntity(NewEntity(id, id, EntityPerson))
			if i > 0 {
				prev := string(rune('a' + comp*10 + i - 1))
				g.AddRelation(NewRelation(prev, id, "next", 1.0))
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.FindPath(string(rune('a')), string(rune('a'+9)))
	}
}

func BenchmarkKnowledgeGraph_Cycles(b *testing.B) {
	g := NewKnowledgeGraph()
	// Create a cycle: a -> b -> c -> d -> e -> a
	entities := []string{"a", "b", "c", "d", "e"}
	for _, id := range entities {
		g.AddEntity(NewEntity(id, id, EntityPerson))
	}
	for i := 0; i < len(entities); i++ {
		next := (i + 1) % len(entities)
		g.AddRelation(NewRelation(entities[i], entities[next], "next", 1.0))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.FindPath("a", "a")
	}
}

func BenchmarkKnowledgeGraph_LargeGraph_FindPath(b *testing.B) {
	g := NewKnowledgeGraph()
	n := 200
	// Create a long chain
	for i := 0; i < n; i++ {
		id := string(rune('a'+i%26)) + string(rune('0'+i/26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		if i > 0 {
			prev := string(rune('a'+(i-1)%26)) + string(rune('0'+(i-1)/26))
			g.AddRelation(NewRelation(prev, id, "next", 1.0))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.FindPath(string(rune('a')), string(rune('a'))+string(rune('0'+(n-1)/26)))
	}
}

func BenchmarkKnowledgeGraph_LargeGraph_TransitiveClosure(b *testing.B) {
	g := NewKnowledgeGraph()
	n := 50
	// Create a chain
	for i := 0; i < n; i++ {
		id := string(rune('a' + i%26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		if i > 0 {
			prev := string(rune('a' + (i-1)%26))
			g.AddRelation(NewRelation(prev, id, "next", 1.0))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.TransitiveClosure(string(rune('a')))
	}
}

func BenchmarkKnowledgeGraph_DenseGraph_CommonNeighbors(b *testing.B) {
	g := NewKnowledgeGraph()
	// Create a dense graph: center connected to many entities, all connected to each other
	centerID := "center"
	g.AddEntity(NewEntity(centerID, "Center", EntityPerson))
	n := 100
	for i := 0; i < n; i++ {
		id := string(rune('a' + i%26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		g.AddRelation(NewRelation(centerID, id, "connected", 1.0))
		g.AddRelation(NewRelation(id, centerID, "connected", 1.0))
		// Connect to a few others
		for j := 1; j <= 3 && i+j < n; j++ {
			other := string(rune('a' + (i+j)%26))
			g.AddRelation(NewRelation(id, other, "linked", 0.8))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.CommonNeighbors(string(rune('a')), string(rune('b')))
	}
}

func BenchmarkKnowledgeGraph_LargeGraph_Neighbors(b *testing.B) {
	g := NewKnowledgeGraph()
	g.AddEntity(NewEntity("center", "Center", EntityPerson))
	n := 500
	for i := 0; i < n; i++ {
		id := string(rune('a' + i%26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		g.AddRelation(NewRelation("center", id, "connected", 1.0))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Neighbors("center")
	}
}

func BenchmarkKnowledgeGraph_DenseGraph_FindPath(b *testing.B) {
	g := NewKnowledgeGraph()
	n := 50
	// Create a dense graph with many connections
	for i := 0; i < n; i++ {
		id := string(rune('a' + i%26))
		g.AddEntity(NewEntity(id, id, EntityPerson))
		// Connect to next 5 entities
		for j := 1; j <= 5; j++ {
			next := (i + j) % n
			neighbor := string(rune('a' + next%26))
			g.AddRelation(NewRelation(id, neighbor, "linked", 1.0))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.FindPath(string(rune('a')), string(rune('a'+49%26)))
	}
}
