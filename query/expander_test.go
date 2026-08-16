package query

import (
	"context"
	"testing"

	"github.com/deagy/recall/graph"
)

func TestGraphExpander_Expand_WithSynonyms(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	expander := NewGraphExpander(g)

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "concept", Confidence: 0.9},
		},
		Relations:  nil,
		SubQueries: []string{"What is Go?"},
		Filters:    nil,
		Confidence: 0.8,
	}

	expanded, err := expander.Expand(context.Background(), parsed)
	if err != nil {
		t.Fatal(err)
	}

	// Should have expanded entities with synonyms
	if len(expanded.Entities) <= len(parsed.Entities) {
		t.Error("expected expanded entities to be more than original")
	}

	// Check that synonyms were added
	foundSynonym := false
	for _, e := range expanded.Entities {
		if e.Text == "golang" {
			foundSynonym = true
			break
		}
	}
	if !foundSynonym {
		t.Error("expected 'golang' synonym to be added")
	}
}

func TestGraphExpander_Expand_WithGraph(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	e1 := graph.NewEntity("go", "Go", graph.EntityConcept)
	e2 := graph.NewEntity("python", "Python", graph.EntityConcept)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddRelation(graph.NewRelation("go", "python", "compare", 0.8))

	expander := NewGraphExpander(g)

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "concept", Confidence: 0.9},
		},
		Relations:  nil,
		SubQueries: []string{"What is Go?"},
		Filters:    nil,
		Confidence: 0.8,
	}

	expanded, err := expander.Expand(context.Background(), parsed)
	if err != nil {
		t.Fatal(err)
	}

	// Should have expanded with related entities
	if len(expanded.Entities) <= len(parsed.Entities) {
		t.Error("expected expanded entities to include related entities")
	}
}

func TestGraphExpander_Expand_Nil(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	expander := NewGraphExpander(g)

	expanded, err := expander.Expand(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if expanded != nil {
		t.Error("expected nil for nil input")
	}
}

func TestGraphExpander_Expand_Relations(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	e1 := graph.NewEntity("go", "Go", graph.EntityConcept)
	e2 := graph.NewEntity("python", "Python", graph.EntityConcept)
	g.AddEntity(e1)
	g.AddEntity(e2)
	g.AddRelation(graph.NewRelation("go", "python", "compare", 0.8))

	expander := NewGraphExpander(g)

	parsed := &ParsedQuery{
		Original: "Go and Python",
		Intent:   IntentComparative,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "concept", Confidence: 0.9},
			{Text: "Python", Type: "concept", Confidence: 0.9},
		},
		Relations: []ExtractedRelation{
			{From: "Go", To: "Python", Type: "compare", Confidence: 0.8},
		},
		SubQueries: []string{"What is Go?", "What is Python?"},
		Filters:    nil,
		Confidence: 0.9,
	}

	expanded, err := expander.Expand(context.Background(), parsed)
	if err != nil {
		t.Fatal(err)
	}

	// Should have at least the original relations
	if len(expanded.Relations) < len(parsed.Relations) {
		t.Error("expected expanded relations to include original relations")
	}
}

func TestGraphExpander_Expand_ConfidenceBoost(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	expander := NewGraphExpander(g)

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "concept", Confidence: 0.9},
		},
		Relations:  nil,
		SubQueries: []string{"What is Go?"},
		Filters:    nil,
		Confidence: 0.8,
	}

	expanded, err := expander.Expand(context.Background(), parsed)
	if err != nil {
		t.Fatal(err)
	}

	// Confidence should be boosted due to expansion
	if expanded.Confidence <= parsed.Confidence {
		t.Errorf("expected confidence boost, got %f <= %f", expanded.Confidence, parsed.Confidence)
	}
}

func TestGraphExpander_MaxExpansions(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	expander := NewGraphExpander(g)
	expander.MaxExpansions = 2

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "concept", Confidence: 0.9},
		},
		Relations:  nil,
		SubQueries: []string{"What is Go?"},
		Filters:    nil,
		Confidence: 0.8,
	}

	expanded, err := expander.Expand(context.Background(), parsed)
	if err != nil {
		t.Fatal(err)
	}

	// Should not exceed max expansions
	if len(expanded.Entities) > 1+expander.MaxExpansions {
		t.Errorf("expected at most %d expanded entities, got %d", 1+expander.MaxExpansions, len(expanded.Entities))
	}
}
