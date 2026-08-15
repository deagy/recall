package reasoning

import (
	"testing"

	"github.com/deagy/recall/graph"
)

func BenchmarkInverseRule_Apply(b *testing.B) {
	rule := &InverseRule{MinWeight: 0.5}
	rel := graph.NewRelation("alice", "bob", "knows", 0.9)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rule.Apply(rel)
	}
}

func BenchmarkCompositionRule_Apply(b *testing.B) {
	rule := &CompositionRule{
		Rules: map[string]string{
			"located_in|works_at": "works_in",
		},
		MinWeight: 0.5,
	}
	rel := graph.NewRelation("alice", "paris", "located_in", 0.9)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rule.Apply(rel)
	}
}

func BenchmarkCompositionRule_ComposeRelations(b *testing.B) {
	rule := &CompositionRule{
		Rules: map[string]string{
			"located_in|works_at": "works_in",
		},
		MinWeight: 0.5,
	}
	rel1 := graph.NewRelation("alice", "paris", "located_in", 0.9)
	rel2 := graph.NewRelation("paris", "france", "works_at", 0.8)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rule.ComposeRelations(rel1, rel2)
	}
}

func BenchmarkProductAggregator_Aggregate(b *testing.B) {
	agg := &ProductAggregator{Decay: 0.9}
	scores := []float64{0.9, 0.8, 0.7, 0.6, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg.Aggregate(scores)
	}
}

func BenchmarkMinAggregator_Aggregate(b *testing.B) {
	agg := &MinAggregator{Decay: 0.9}
	scores := []float64{0.9, 0.8, 0.7, 0.6, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg.Aggregate(scores)
	}
}

func BenchmarkAverageAggregator_Aggregate(b *testing.B) {
	agg := &AverageAggregator{Decay: 0.9}
	scores := []float64{0.9, 0.8, 0.7, 0.6, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg.Aggregate(scores)
	}
}

func BenchmarkEntityExtractor_ExtractEntities(b *testing.B) {
	extractor := NewEntityExtractor()
	text := "Alice works at Google in New York. Bob works at Microsoft in Seattle."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractEntities(text)
	}
}

func BenchmarkEntityExtractor_ExpandQuery(b *testing.B) {
	extractor := NewEntityExtractor()
	query := "Go programming language"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExpandQuery(query)
	}
}

func BenchmarkEngine_ExplorePaths(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	for i := 0; i < 100; i++ {
		g.AddEntity(graph.NewEntity(string(rune('a'+i%26)), string(rune('a'+i%26)), graph.EntityPerson))
		if i > 0 {
			g.AddRelation(graph.NewRelation(string(rune('a'+(i-1)%26)), string(rune('a'+i%26)), "knows", 0.9))
		}
	}

	cfg := DefaultConfig()
	cfg.MaxDepth = 3
	engine := NewEngine(g, cfg)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.ExplorePaths(string(rune('a')), string(rune('a'+99%26)))
	}
}

func BenchmarkEngine_InferRelations(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	for i := 0; i < 50; i++ {
		g.AddEntity(graph.NewEntity(string(rune('a'+i%26)), string(rune('a'+i%26)), graph.EntityPerson))
		if i > 0 {
			g.AddRelation(graph.NewRelation(string(rune('a'+(i-1)%26)), string(rune('a'+i%26)), "knows", 0.9))
		}
	}

	cfg := DefaultConfig()
	engine := NewEngine(g, cfg)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.InferRelations()
	}
}

func BenchmarkEngine_Reason(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	for i := 0; i < 50; i++ {
		g.AddEntity(graph.NewEntity(string(rune('a'+i%26)), string(rune('a'+i%26)), graph.EntityPerson))
		if i > 0 {
			g.AddRelation(graph.NewRelation(string(rune('a'+(i-1)%26)), string(rune('a'+i%26)), "knows", 0.9))
		}
	}

	cfg := DefaultConfig()
	engine := NewEngine(g, cfg)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Reason("Alice knows Bob", 3)
	}
}
