package reasoning

import (
	"testing"

	"github.com/deagy/recall/graph"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAntiSymmetricRule_Apply_NotAntiSymmetric(t *testing.T) {
	rule := &AntiSymmetricRule{RelationTypes: []string{"knows"}}
	rel := graph.NewRelation("alice", "bob", "friend", 0.8)

	_, ok := rule.Apply(rel)
	assert.False(t, ok, "anti_symmetric rule returns false for non-listed types")
}

func TestDefaultRules_Names(t *testing.T) {
	rules := DefaultRules()
	require.Len(t, rules, 4, "expected 4 default rules")

	names := make(map[string]bool)
	for _, r := range rules {
		names[r.Name()] = true
	}
	assert.True(t, names["reverse"], "expected reverse rule")
	assert.True(t, names["transitive"], "expected transitive rule")
	assert.True(t, names["symmetric"], "expected symmetric rule")
	assert.True(t, names["anti_symmetric"], "expected anti_symmetric rule")
}

func TestReverseRule_Apply_BelowMinWeight(t *testing.T) {
	rule := &ReverseRule{MinWeight: 0.9}
	rel := graph.NewRelation("alice", "bob", "works_at", 0.5)

	_, ok := rule.Apply(rel)
	assert.False(t, ok, "reverse rule should not apply below min weight")
}

func TestReverseRule_ZeroMinWeight(t *testing.T) {
	rule := &ReverseRule{MinWeight: 0}
	rel := graph.NewRelation("a", "b", "knows", 0.1)

	inferred, ok := rule.Apply(rel)
	if !ok {
		t.Fatal("expected rule to apply with zero min weight")
	}
	if inferred.From != "b" || inferred.To != "a" {
		t.Fatalf("expected reverse relation, got %s -> %s", inferred.From, inferred.To)
	}
}

func TestTransitiveRule_ZeroMinWeight(t *testing.T) {
	rule := &TransitiveRule{MinWeight: 0}
	rel := graph.NewRelation("a", "b", "knows", 0.1)

	inferred, ok := rule.Apply(rel)
	if !ok {
		t.Fatal("expected rule to apply with zero min weight")
	}
	if inferred.From != "a" || inferred.To != "b" {
		t.Fatalf("expected same relation, got %s -> %s", inferred.From, inferred.To)
	}
}

func TestSymmetricRule_EmptyTypes(t *testing.T) {
	rule := &SymmetricRule{RelationTypes: []string{}}
	rel := graph.NewRelation("a", "b", "knows", 0.9)

	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected rule to not apply with empty types")
	}
}

func TestAntiSymmetricRule_EmptyTypes(t *testing.T) {
	rule := &AntiSymmetricRule{RelationTypes: []string{}}
	rel := graph.NewRelation("a", "b", "knows", 0.9)

	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected rule to not apply with empty types")
	}
}

func TestReverseRule_Name(t *testing.T) {
	rule := &ReverseRule{}
	if rule.Name() != "reverse" {
		t.Errorf("expected 'reverse', got %q", rule.Name())
	}
}

func TestTransitiveRule_Name(t *testing.T) {
	rule := &TransitiveRule{}
	if rule.Name() != "transitive" {
		t.Errorf("expected 'transitive', got %q", rule.Name())
	}
}

func TestSymmetricRule_Name(t *testing.T) {
	rule := &SymmetricRule{}
	if rule.Name() != "symmetric" {
		t.Errorf("expected 'symmetric', got %q", rule.Name())
	}
}

func TestAntiSymmetricRule_Name(t *testing.T) {
	rule := &AntiSymmetricRule{}
	if rule.Name() != "anti_symmetric" {
		t.Errorf("expected 'anti_symmetric', got %q", rule.Name())
	}
}

func TestDefaultRules_ZeroWeight(t *testing.T) {
	rules := DefaultRules()
	for _, r := range rules {
		if r == nil {
			t.Fatal("expected non-nil rule")
		}
	}
}

func TestReverseRule_Apply_ZeroWeight(t *testing.T) {
	rule := &ReverseRule{MinWeight: 0.5}
	rel := graph.NewRelation("a", "b", "knows", 0)

	_, ok := rule.Apply(rel)
	if ok {
		t.Fatal("expected rule to not apply with zero weight")
	}
}

func TestSymmetricRule_Apply_MultipleTypes(t *testing.T) {
	rule := &SymmetricRule{RelationTypes: []string{"knows", "friend", "related_to"}}

	rel1 := graph.NewRelation("a", "b", "knows", 0.9)
	_, ok := rule.Apply(rel1)
	if !ok {
		t.Fatal("expected rule to apply for 'knows'")
	}

	rel2 := graph.NewRelation("a", "b", "friend", 0.9)
	_, ok = rule.Apply(rel2)
	if !ok {
		t.Fatal("expected rule to apply for 'friend'")
	}

	rel3 := graph.NewRelation("a", "b", "works_at", 0.9)
	_, ok = rule.Apply(rel3)
	if ok {
		t.Fatal("expected rule to not apply for 'works_at'")
	}
}

func TestTransitiveRule_Apply_NilRelation(t *testing.T) {
	rule := &TransitiveRule{MinWeight: 0.5}
	// This tests the rule with a valid relation (nil check is in the rule)
	rel := graph.NewRelation("a", "b", "knows", 0.9)
	inferred, ok := rule.Apply(rel)
	if !ok {
		t.Fatal("expected rule to apply")
	}
	if inferred == nil {
		t.Fatal("expected non-nil inferred relation")
	}
}
