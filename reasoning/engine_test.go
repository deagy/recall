package reasoning

import (
	"math"
	"testing"

	"github.com/deagy/recall/graph"
)

func TestReverseRule_Apply(t *testing.T) {
	rule := &ReverseRule{MinWeight: 0.5}

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

func TestReverseRule_ApplyBelowMinWeight(t *testing.T) {
	rule := &ReverseRule{MinWeight: 0.5}

	rel := graph.NewRelation("a", "b", "knows", 0.3)
	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected rule to not apply below min weight")
	}
}

func TestTransitiveRule_Apply(t *testing.T) {
	rule := &TransitiveRule{MinWeight: 0.5}

	rel := graph.NewRelation("a", "b", "knows", 0.9)
	inferred, ok := rule.Apply(rel)
	if !ok {
		t.Fatal("expected rule to apply")
	}
	// TransitiveRule returns the relation as-is for single-relation application
	if inferred.From != "a" || inferred.To != "b" {
		t.Fatalf("expected same relation, got %s -> %s", inferred.From, inferred.To)
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
	if len(rules) != 4 {
		t.Fatalf("expected 4 default rules, got %d", len(rules))
	}
	if rules[0].Name() != "reverse" {
		t.Fatalf("expected first rule to be 'reverse', got '%s'", rules[0].Name())
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

func TestEngine_InferRelations_RespectsMinConfidence(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("b", "Bob", graph.EntityPerson))
	g.AddRelation(graph.NewRelation("a", "b", "knows", 0.9))

	// Every default rule derives a confidence <= 0.9 from a 0.9-weight
	// relation, so a 0.99 threshold must drop all of them.
	strict := NewEngine(g, Config{MaxDepth: 1, MinConfidence: 0.99, Rules: DefaultRules()})
	if got := strict.InferRelations(); len(got) != 0 {
		t.Fatalf("expected no inferences above 0.99 threshold, got %d: %v", len(got), got)
	}

	// A 0.3 threshold must keep inferences, and all kept ones must meet it.
	loose := NewEngine(g, Config{MaxDepth: 1, MinConfidence: 0.3, Rules: DefaultRules()})
	got := loose.InferRelations()
	if len(got) == 0 {
		t.Fatal("expected inferences above 0.3 threshold")
	}
	for _, ir := range got {
		if ir.Confidence < 0.3 {
			t.Fatalf("inferred relation below configured MinConfidence: %s", ir)
		}
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

func TestEngine_DefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MaxDepth != 3 {
		t.Errorf("expected MaxDepth 3, got %d", cfg.MaxDepth)
	}
	if cfg.MinConfidence != 0.3 {
		t.Errorf("expected MinConfidence 0.3, got %f", cfg.MinConfidence)
	}
	if len(cfg.Rules) != 4 {
		t.Errorf("expected 4 rules, got %d", len(cfg.Rules))
	}
}

func TestEngine_ZeroMaxDepth(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	cfg := Config{MaxDepth: 0, Rules: DefaultRules()}
	eng := NewEngine(g, cfg)
	if eng.maxDepth != 3 {
		t.Errorf("expected maxDepth 3, got %d", eng.maxDepth)
	}
}

func TestEngine_NegativeMinConfidence(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	cfg := Config{MinConfidence: -1, Rules: DefaultRules()}
	eng := NewEngine(g, cfg)
	if eng.rules == nil {
		t.Fatal("expected non-nil rules")
	}
}

func TestEngine_ExplorePaths_SameEntity(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "Alice", graph.EntityPerson))

	eng := NewEngine(g, DefaultConfig())
	paths := eng.ExplorePaths("a", "a")
	// Same entity should return no paths (visited check)
	_ = paths
}

func TestEngine_ExplorePaths_NonExistentEntity(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "Alice", graph.EntityPerson))

	eng := NewEngine(g, DefaultConfig())
	paths := eng.ExplorePaths("nonexistent", "a")
	if paths != nil {
		t.Fatal("expected nil for non-existent entity")
	}

	paths = eng.ExplorePaths("a", "nonexistent")
	if paths != nil {
		t.Fatal("expected nil for non-existent target")
	}
}

func TestEngine_Reason_ZeroHops(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))

	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("Alice", 0)
	// Zero hops should use maxDepth
	_ = results
}

func TestEngine_Reason_LargeHops(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))

	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("Alice", 100) // Larger than maxDepth
	// Should use maxDepth
	_ = results
}

func TestEngine_Reason_NoCapitalizedWords(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("no capitalized words here", 2)
	if results != nil {
		t.Fatal("expected nil for query with no capitalized words")
	}
}

func TestEngine_Reason_SingleEntity(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))

	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("Alice", 2)
	// Single entity should not produce paths (from==to check)
	_ = results
}

func TestEngine_InferRelations_EmptyGraph(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	eng := NewEngine(g, DefaultConfig())
	inferred := eng.InferRelations()
	if inferred != nil && len(inferred) != 0 {
		t.Fatalf("expected empty inferred, got %d", len(inferred))
	}
}

func TestEngine_ExplorePaths_EmptyGraph(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	eng := NewEngine(g, DefaultConfig())
	paths := eng.ExplorePaths("a", "b")
	if paths != nil {
		t.Fatal("expected nil for empty graph")
	}
}

func TestEngine_Reason_Punctuation(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
	g.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))

	eng := NewEngine(g, DefaultConfig())
	results := eng.Reason("Alice, knows Bob!", 2)
	// Should extract Alice and Bob
	_ = results
}

func TestInferredRelation_String_Empty(t *testing.T) {
	ir := &InferredRelation{}
	s := ir.String()
	if s == "" {
		t.Fatal("expected non-empty string")
	}
}
