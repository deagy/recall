package reasoning

import (
	"testing"

	"github.com/deagy/recall/graph"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests cover branches not exercised by inference_test.go: the
// rule Name() methods, the inverse mapping branch, the composition
// default-type/below-min-weight branches, CommonInterestRule.Apply,
// HierarchyRule.InferTransitiveClosure, and the Min/Average empty
// aggregator inputs.

func TestInverseRule_Name(t *testing.T) {
	assert.Equal(t, "inverse", (&InverseRule{}).Name())
}

func TestInverseRule_Apply_WithMapping(t *testing.T) {
	r := &InverseRule{Mappings: map[string]string{"works_at": "works_for"}}
	in, ok := r.Apply(graph.NewRelation("alice", "acme", "works_at", 0.9))
	require.True(t, ok)
	assert.Equal(t, "works_for", in.Type, "explicit mapping must win over the _reverse default")
	assert.Equal(t, "acme", in.From)
	assert.Equal(t, "alice", in.To)
	assert.InDelta(t, 0.81, in.Weight, 1e-9)
}

func TestCompositionRule_Name(t *testing.T) {
	assert.Equal(t, "composition", (&CompositionRule{}).Name())
}

func TestCompositionRule_ComposeRelations_DefaultAndMinWeight(t *testing.T) {
	r := &CompositionRule{MinWeight: 0.5}
	out, ok := r.ComposeRelations(
		graph.NewRelation("a", "b", "x", 0.8),
		graph.NewRelation("b", "c", "y", 0.9))
	require.True(t, ok)
	assert.Equal(t, "x_via_y", out.Type, "unknown pattern must default to type1_via_type2")
	assert.InDelta(t, 0.72, out.Weight, 1e-9)

	// 0.6 * 0.6 = 0.36 < MinWeight 0.5
	_, ok = r.ComposeRelations(
		graph.NewRelation("a", "b", "x", 0.6),
		graph.NewRelation("b", "c", "y", 0.6))
	assert.False(t, ok, "confidence product below MinWeight must not compose")
}

func TestCommonInterestRule_Name(t *testing.T) {
	assert.Equal(t, "common_interest", (&CommonInterestRule{}).Name())
}

func TestCommonInterestRule_Apply(t *testing.T) {
	r := &CommonInterestRule{MinWeight: 0.3}
	rel := graph.NewRelation("a", "b", "enjoyed", 0.9)
	out, ok := r.Apply(rel)
	require.True(t, ok)
	assert.Same(t, rel, out, "single-relation application returns the relation unchanged")

	_, ok = r.Apply(graph.NewRelation("a", "b", "enjoyed", 0.1))
	assert.False(t, ok)
}

func TestHierarchyRule_Name(t *testing.T) {
	assert.Equal(t, "hierarchy", (&HierarchyRule{}).Name())
}

// A is_a B, B is_a C ⇒ A is_a C must be inferred — and nothing else:
// no self-loops, and no backward edges (the bidirectional
// KnowledgeGraph.TransitiveClosure used to infer "B is_a A" too).
func TestHierarchyRule_InferTransitiveClosure(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("dog", "Dog", graph.EntityConcept))
	g.AddEntity(graph.NewEntity("mammal", "Mammal", graph.EntityConcept))
	g.AddEntity(graph.NewEntity("animal", "Animal", graph.EntityConcept))
	g.AddRelation(graph.NewRelation("dog", "mammal", "is_a", 0.9))
	g.AddRelation(graph.NewRelation("mammal", "animal", "is_a", 0.9))

	r := &HierarchyRule{HierarchyTypes: []string{"is_a"}}
	inferred := r.InferTransitiveClosure(g)

	require.Len(t, inferred, 1)
	assert.Equal(t, "dog", inferred[0].From)
	assert.Equal(t, "animal", inferred[0].To)
	assert.Equal(t, "is_a", inferred[0].Type)
	assert.InDelta(t, 0.72, inferred[0].Weight, 1e-9)
	for _, rel := range inferred {
		assert.NotEqual(t, rel.From, rel.To, "self-loops must never be inferred")
	}
}

func TestHierarchyRule_InferTransitiveClosure_NoHierarchyRelations(t *testing.T) {
	g := graph.NewKnowledgeGraph()
	g.AddEntity(graph.NewEntity("a", "A", graph.EntityConcept))
	g.AddEntity(graph.NewEntity("b", "B", graph.EntityConcept))
	g.AddRelation(graph.NewRelation("a", "b", "knows", 0.9))

	r := &HierarchyRule{HierarchyTypes: []string{"is_a"}}
	assert.Empty(t, r.InferTransitiveClosure(g))
}

func TestAggregators_MinAndAverageEmptyInput(t *testing.T) {
	assert.Equal(t, 0.0, (&MinAggregator{}).Aggregate(nil))
	assert.Equal(t, 0.0, (&AverageAggregator{}).Aggregate(nil))
}
