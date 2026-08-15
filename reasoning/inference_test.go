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

func TestInverseRule_Apply(t *testing.T) {
	rule := &InverseRule{MinWeight: 0.5}
	rel := graph.NewRelation("alice", "bob", "knows", 0.9)

	inferred, ok := rule.Apply(rel)
	require.True(t, ok, "inverse rule should apply")
	require.Equal(t, "knows_reverse", inferred.Type, "expected inverse type")
	require.Equal(t, "bob", inferred.From, "expected reversed from")
	require.Equal(t, "alice", inferred.To, "expected reversed to")
	require.InDelta(t, 0.81, inferred.Weight, 0.01, "expected 0.9*0.9 weight")
}

func TestInverseRule_Apply_BelowMinWeight(t *testing.T) {
	rule := &InverseRule{MinWeight: 0.5}
	rel := graph.NewRelation("alice", "bob", "knows", 0.3)

	_, ok := rule.Apply(rel)
	require.False(t, ok, "inverse rule should not apply below min weight")
}

func TestCompositionRule_Apply(t *testing.T) {
	rule := &CompositionRule{
		Rules: map[string]string{
			"located_in|works_at": "works_in",
		},
		MinWeight: 0.5,
	}

	// Test single relation (should return as-is)
	rel := graph.NewRelation("alice", "bob", "works_at", 0.9)
	inferred, ok := rule.Apply(rel)
	require.True(t, ok, "composition rule should apply to single relation")
	require.Equal(t, rel, inferred, "expected same relation for single relation")
}

func TestCompositionRule_ComposeRelations(t *testing.T) {
	rule := &CompositionRule{
		Rules: map[string]string{
			"located_in|works_at": "works_in",
		},
		MinWeight: 0.5,
	}

	rel1 := graph.NewRelation("alice", "paris", "located_in", 0.9)
	rel2 := graph.NewRelation("paris", "france", "works_at", 0.8)

	inferred, ok := rule.ComposeRelations(rel1, rel2)
	require.True(t, ok, "composition should succeed")
	require.Equal(t, "works_in", inferred.Type, "expected composed type")
	require.Equal(t, "alice", inferred.From, "expected from")
	require.Equal(t, "france", inferred.To, "expected to")
	require.InDelta(t, 0.72, inferred.Weight, 0.01, "expected 0.9*0.8 weight")
}

func TestCompositionRule_ComposeRelations_NoMatch(t *testing.T) {
	rule := &CompositionRule{
		Rules: map[string]string{
			"located_in|works_at": "works_in",
		},
		MinWeight: 0.5,
	}

	rel1 := graph.NewRelation("alice", "bob", "knows", 0.9)
	rel2 := graph.NewRelation("charlie", "dave", "knows", 0.8)

	_, ok := rule.ComposeRelations(rel1, rel2)
	require.False(t, ok, "composition should fail for non-matching relations")
}

func TestCommonInterestRule_FindCommonInterests(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("charlie", "Charlie", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("go", "Go", graph.EntityConcept))

	g.AddRelation(graph.NewRelation("alice", "go", "uses", 0.9))
	g.AddRelation(graph.NewRelation("bob", "go", "uses", 0.8))

	rule := &CommonInterestRule{MinCommonRelations: 1, MinWeight: 0.5}
	common := rule.FindCommonInterests(g, "alice")

	require.NotEmpty(t, common, "expected common interests")
}

func TestHierarchyRule_Apply(t *testing.T) {
	rule := &HierarchyRule{
		HierarchyTypes: []string{"is_a", "part_of"},
		MinWeight: 0.5,
	}

	rel := graph.NewRelation("dog", "animal", "is_a", 0.9)
	inferred, ok := rule.Apply(rel)
	require.True(t, ok, "hierarchy rule should apply to hierarchical relation")
	require.Equal(t, rel, inferred, "expected same relation")
}

func TestHierarchyRule_Apply_NonHierarchy(t *testing.T) {
	rule := &HierarchyRule{
		HierarchyTypes: []string{"is_a", "part_of"},
		MinWeight: 0.5,
	}

	rel := graph.NewRelation("alice", "bob", "knows", 0.9)
	_, ok := rule.Apply(rel)
	require.False(t, ok, "hierarchy rule should not apply to non-hierarchical relation")
}

func TestProductAggregator_Aggregate(t *testing.T) {
	agg := &ProductAggregator{Decay: 0.9}
	scores := []float64{0.9, 0.8, 0.7}

	result := agg.Aggregate(scores)
	expected := 0.9 * 0.8 * 0.7 * 0.9 * 0.9
	require.InDelta(t, expected, result, 0.01, "expected product with decay")
}

func TestProductAggregator_Aggregate_Empty(t *testing.T) {
	agg := &ProductAggregator{Decay: 0.9}
	result := agg.Aggregate([]float64{})
	require.Equal(t, 0.0, result, "expected 0 for empty scores")
}

func TestMinAggregator_Aggregate(t *testing.T) {
	agg := &MinAggregator{Decay: 0.9}
	scores := []float64{0.9, 0.5, 0.7}

	result := agg.Aggregate(scores)
	expected := 0.5 * 0.9 * 0.9
	require.InDelta(t, expected, result, 0.01, "expected minimum with decay")
}

func TestAverageAggregator_Aggregate(t *testing.T) {
	agg := &AverageAggregator{Decay: 0.9}
	scores := []float64{0.9, 0.8, 0.7}

	result := agg.Aggregate(scores)
	expected := (0.9 + 0.8 + 0.7) / 3.0 * 0.9
	require.InDelta(t, expected, result, 0.01, "expected average with decay")
}

func TestDefaultAggregator(t *testing.T) {
	agg := DefaultAggregator()
	require.NotNil(t, agg, "expected non-nil aggregator")
	require.IsType(t, &ProductAggregator{}, agg, "expected ProductAggregator")
}

func TestEntityExtractor_ExtractEntities(t *testing.T) {
	extractor := NewEntityExtractor()
	text := "Alice works at Google in New York"

	entities := extractor.ExtractEntities(text)
	require.NotEmpty(t, entities, "expected entities")
}

func TestEntityExtractor_ExpandQuery(t *testing.T) {
	extractor := NewEntityExtractor()
	query := "Go programming language"

	expanded := extractor.ExpandQuery(query)
	require.NotEmpty(t, expanded, "expected expanded query")
	require.Contains(t, expanded, "go", "expected 'go' in expanded query")
}

func TestEntityExtractor_ExpandQuery_WithSynonyms(t *testing.T) {
	extractor := NewEntityExtractor()
	query := "Go"

	expanded := extractor.ExpandQuery(query)
	require.Contains(t, expanded, "golang", "expected 'golang' synonym")
	require.Contains(t, expanded, "gopher", "expected 'gopher' synonym")
}
