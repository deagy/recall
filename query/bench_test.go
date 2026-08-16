package query

import (
	"context"
	"testing"

	"github.com/deagy/recall/graph"
)

func BenchmarkDefaultParser_Parse_Factual(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "What is Go programming language?")
	}
}

func BenchmarkDefaultParser_Parse_Comparative(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "How does Go compare to Python?")
	}
}

func BenchmarkDefaultParser_Parse_Temporal(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "What happened in 2020?")
	}
}

func BenchmarkDefaultParser_Parse_Causal(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "Why did the market crash?")
	}
}

func BenchmarkDefaultParser_Parse_Procedural(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "How do I implement a linked list?")
	}
}

func BenchmarkDefaultParser_Parse_Existential(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "Does Go support generics?")
	}
}

func BenchmarkDefaultParser_ExtractEntities(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "What is Go programming language?")
	}
}

func BenchmarkDefaultParser_DecomposeQuery(b *testing.B) {
	parser := NewDefaultParser(nil)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser.Parse(ctx, "Go vs Python performance")
	}
}

func BenchmarkGraphExpander_Expand_Synonyms(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	expander := NewGraphExpander(g)
	ctx := context.Background()

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "language", Confidence: 0.9},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		expander.Expand(ctx, parsed)
	}
}

func BenchmarkGraphExpander_Expand_WithGraph(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	e1 := graph.NewEntity("go", "Go", graph.EntityConcept)
	e2 := graph.NewEntity("python", "Python", graph.EntityConcept)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddRelation(graph.NewRelation("go", "python", "compare", 0.8))

	expander := NewGraphExpander(g)
	ctx := context.Background()

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "language", Confidence: 0.9},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		expander.Expand(ctx, parsed)
	}
}

func BenchmarkGraphExpander_Expand_MultipleEntities(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	expander := NewGraphExpander(g)
	ctx := context.Background()

	parsed := &ParsedQuery{
		Original: "Go and Python and JavaScript",
		Intent:   IntentComparative,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "language", Confidence: 0.9},
			{Text: "Python", Type: "language", Confidence: 0.9},
			{Text: "JavaScript", Type: "language", Confidence: 0.9},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		expander.Expand(ctx, parsed)
	}
}

func BenchmarkGraphExpander_Expand_WithRelations(b *testing.B) {
	g := graph.NewKnowledgeGraph()
	e1 := graph.NewEntity("go", "Go", graph.EntityConcept)
	e2 := graph.NewEntity("python", "Python", graph.EntityConcept)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddRelation(graph.NewRelation("go", "python", "compare", 0.8))

	expander := NewGraphExpander(g)
	ctx := context.Background()

	parsed := &ParsedQuery{
		Original: "Go and Python",
		Intent:   IntentComparative,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "language", Confidence: 0.9},
			{Text: "Python", Type: "language", Confidence: 0.9},
		},
		Relations: []ExtractedRelation{
			{From: "Go", To: "Python", Type: "compare", Confidence: 0.8},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		expander.Expand(ctx, parsed)
	}
}
