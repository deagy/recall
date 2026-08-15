package reasoning

import (
	"math"
	"testing"

	"github.com/deagy/recall/graph"
)

func TestTransitiveRule_Apply(t *testing.T) {
	rule := &TransitiveRule{MinWeight: 0.5}

	rel := graph.NewRelation("a", "b", "knows", 0.9)
	inferred, ok := rule.Apply(rel)
	if !ok {
		t.Fatal("expected rule to apply")
	}
	if inferred.From != "b" || inferred.To != "a" {
		t.Fatalf("expected reverse relation, got %s -> %s", inferred.From, inferred.To)
	}
	if math.Abs(inferred.Weight-0.72) > 0.001 {
		t.Fatalf("expected weight ~0.72, got %f", inferred.Weight)
	}
}

func TestTransitiveRule_ApplyBelowMinWeight(t *testing.T) {
	rule := &TransitiveRule{MinWeight: 0.5}

	rel := graph.NewRelation("a", "b", "knows", 0.3)
	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected rule to not apply below min weight")
	}
}

func TestSymmetricRule_Apply(t *testing.T) {
	rule := &SymmetricRule{RelationTypes: []string{"knows", "friend"}}

	rel := graph.NewRelation("a", "b", "knows", 0.9)
	inferred, ok := rule.Apply(rel)
	if !ok {
		t.Fatal("expected rule to apply")
	}
	if inferred.From != "b" || inferred.To != "a" {
		t.Fatalf("expected reverse relation, got %s -> %s", inferred.From, inferred.To)
	}
}

func TestSymmetricRule_NotApplicable(t *testing.T) {
	rule := &SymmetricRule{RelationTypes: []string{"knows"}}

	rel := graph.NewRelation("a", "b", "works_at", 0.9)
	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected rule to not apply for non-symmetric relation")
	}
}

func TestAntiSymmetricRule_Apply(t *testing.T) {
	rule := &AntiSymmetricRule{RelationTypes: []string{"works_at", "parent_of"}}

	rel := graph.NewRelation("a", "b", "works_at", 0.9)
	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected anti-symmetric rule to not infer reverse")
	}
}

func TestDefaultRules(t *testing.T) {
	rules := DefaultRules()
	if len(rules) != 3 {
		t.Fatalf("expected 3 default rules, got %d", len(rules))
	}
	if rules[0].Name() != "transitive" {
		t.Fatalf("expected first rule to be 'transitive', got '%s'", rules[0].Name())
	}
}

func TestEngine_InferRelations(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("b", "Bob", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("c", "Charlie", graph.EntityPerson))
	g.AddRelation(graph.NewRelation("a", "b", "knows", 0.9))
	g.AddRelation(graph.NewRelation("b", "c", "knows", 0.8))

	cfg := DefaultConfig()
	cfg.MaxDepth = 1
	eng := NewEngine(g, cfg)

	inferred := eng.InferRelations()
	if len(inferred) == 0 {
		t.Fatal("expected inferred relations")
	}
}

func TestEngine_ExplorePaths(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("b", "Bob", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("c", "Charlie", graph.EntityPerson))
	g.AddRelation(graph.NewRelation("a", "b", "knows", 0.9))
	g.AddRelation(graph.NewRelation("b", "c", "knows", 0.8))

	eng := NewEngine(g, DefaultConfig())
	paths := eng.ExplorePaths("a", "c")
	if len(paths) == 0 {
		t.Fatal("expected at least one path")
	}
	if paths[0].Length() != 3 {
		t.Fatalf("expected path length 3, got %d", paths[0].Length())
	}
}

func TestEngine_ExplorePathsNoPath(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("b", "Bob", graph.EntityPerson))

	eng := NewEngine(g, DefaultConfig())
	paths := eng.ExplorePaths("a", "b")
	if len(paths) != 0 {
		t.Fatalf("expected no paths, got %d", len(paths))
	}
}

func TestEngine_Reason(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("charlie", "Charlie", graph.EntityPerson))
	g.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	g.AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.8))

	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("Alice knows Charlie", 2)
	if len(results) == 0 {
		t.Fatal("expected inferred relations from reasoning")
	}
}

func TestEngine_ReasonEmptyQuery(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("no entities here", 2)
	if results != nil {
		t.Fatal("expected nil for query with no entities")
	}
}

func TestInferredRelation_String(t *testing.T) {
	ir := &InferredRelation{
		From:       "alice",
		To:         "charlie",
		Type:       "knows",
		Confidence: 0.72,
		Rule:       "symmetric",
	}

	s := ir.String()
	if s == "" {
		t.Fatal("expected non-empty string")
	}
}
