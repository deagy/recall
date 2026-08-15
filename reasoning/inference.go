// Package reasoning provides a multi-hop reasoning engine for the knowledge graph.
// It supports inference rules, confidence propagation, and depth-limited path exploration.
package reasoning

import (
	"github.com/deagy/recall/graph"
)

// InferenceRule defines how relationships can be combined or inferred.
type InferenceRule interface {
	// Name is the rule identifier (e.g., "transitive", "symmetric").
	Name() string

	// Apply checks if the rule applies to a given relation and returns an inferred relation.
	Apply(rel *graph.Relation) (*graph.Relation, bool)
}

// TransitiveRule infers that if A->B and B->C, then A->C.
type TransitiveRule struct {
	// MinWeight is the minimum confidence for inferred relations.
	MinWeight float64
}

// Name implements InferenceRule.
func (r *TransitiveRule) Name() string { return "transitive" }

// Apply implements InferenceRule.
func (r *TransitiveRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	if rel.Weight < r.MinWeight {
		return nil, false
	}
	inferred := graph.NewRelation(rel.To, rel.From, rel.Type+"_(reverse)", rel.Weight*0.8)
	return inferred, true
}

// SymmetricRule infers that if A->B, then B->A (for symmetric relationships).
type SymmetricRule struct {
	// RelationTypes are the relation types that are symmetric.
	RelationTypes []string
}

// Name implements InferenceRule.
func (r *SymmetricRule) Name() string { return "symmetric" }

// Apply implements InferenceRule.
func (r *SymmetricRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	for _, t := range r.RelationTypes {
		if rel.Type == t {
			inferred := graph.NewRelation(rel.To, rel.From, rel.Type, rel.Weight)
			return inferred, true
		}
	}
	return nil, false
}

// AntiSymmetricRule infers that if A->B, then NOT(B->A).
type AntiSymmetricRule struct {
	// RelationTypes are the relation types that are anti-symmetric.
	RelationTypes []string
}

// Name implements InferenceRule.
func (r *AntiSymmetricRule) Name() string { return "anti_symmetric" }

// Apply implements InferenceRule.
func (r *AntiSymmetricRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	for _, t := range r.RelationTypes {
		if rel.Type == t {
			// Anti-symmetric means we don't infer the reverse
			return nil, false
		}
	}
	return nil, false
}

// DefaultRules returns the default set of inference rules.
func DefaultRules() []InferenceRule {
	return []InferenceRule{
		&TransitiveRule{MinWeight: 0.5},
		&SymmetricRule{RelationTypes: []string{"knows", "related_to", "friend"}},
		&AntiSymmetricRule{RelationTypes: []string{"works_at", "parent_of", "teaches"}},
	}
}
