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
