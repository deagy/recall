package reasoning

import (
	"strings"

	"github.com/deagy/recall/graph"
)

// InverseRule infers inverse relations (e.g., works_at → works_for).
type InverseRule struct {
	// Mappings defines the inverse relation mappings.
	// Key is the original relation type, value is the inverse type.
	Mappings map[string]string

	// MinWeight is the minimum confidence for inferred relations.
	MinWeight float64
}

// Name implements InferenceRule.
func (r *InverseRule) Name() string { return "inverse" }

// Apply implements InferenceRule.
func (r *InverseRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	if rel.Weight < r.MinWeight {
		return nil, false
	}

	inverseType, ok := r.Mappings[rel.Type]
	if !ok {
		// Default: append "_reverse" if no mapping
		inverseType = rel.Type + "_reverse"
	}

	inferred := graph.NewRelation(rel.To, rel.From, inverseType, rel.Weight*0.9)
	return inferred, true
}

// CompositionRule composes relations (e.g., located_in + works_at → works_in_location).
type CompositionRule struct {
	// Rules defines composition patterns.
	// Key is "type1|type2", value is the composed type.
	Rules map[string]string

	// MinWeight is the minimum confidence for inferred relations.
	MinWeight float64
}

// Name implements InferenceRule.
func (r *CompositionRule) Name() string { return "composition" }

// Apply implements InferenceRule.
// Note: CompositionRule requires two relations to compose.
// When applied to a single relation, it returns the relation unchanged if weight is sufficient.
func (r *CompositionRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	if rel.Weight < r.MinWeight {
		return nil, false
	}
	// For single-relation application, return as-is
	return rel, true
}

// ComposeRelations composes two relations according to the composition rules.
func (r *CompositionRule) ComposeRelations(rel1, rel2 *graph.Relation) (*graph.Relation, bool) {
	if rel1.To != rel2.From {
		return nil, false
	}

	key := rel1.Type + "|" + rel2.Type
	composedType, ok := r.Rules[key]
	if !ok {
		composedType = rel1.Type + "_via_" + rel2.Type
	}

	// Confidence is product of both weights
	confidence := rel1.Weight * rel2.Weight
	if confidence < r.MinWeight {
		return nil, false
	}

	inferred := graph.NewRelation(rel1.From, rel2.To, composedType, confidence)
	return inferred, true
}

// CommonInterestRule infers common interests based on shared relations.
type CommonInterestRule struct {
	// MinCommonRelations is the minimum number of common relations to infer.
	MinCommonRelations int

	// MinWeight is the minimum confidence for inferred relations.
	MinWeight float64
}

// Name implements InferenceRule.
func (r *CommonInterestRule) Name() string { return "common_interest" }

// Apply implements InferenceRule.
func (r *CommonInterestRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	if rel.Weight < r.MinWeight {
		return nil, false
	}
	// Common interest requires two relations from the same source
	return rel, true
}

// FindCommonInterests finds entities with common interests.
func (r *CommonInterestRule) FindCommonInterests(g *graph.KnowledgeGraph, entityID string) []*graph.Entity {
	_, ok := g.GetEntity(entityID)
	if !ok {
		return nil
	}

	relations := g.OutgoingRelations(entityID)
	if len(relations) < r.MinCommonRelations {
		return nil
	}

	// Find other entities that share relations with this entity
	relatedEntities := make(map[string]*graph.Entity)
	for _, rel := range relations {
		toEntity, ok := g.GetEntity(rel.To)
		if ok {
			relatedEntities[rel.To] = toEntity
		}
	}

	var common []*graph.Entity
	for id, ent := range relatedEntities {
		if id != entityID {
			common = append(common, ent)
		}
	}

	return common
}

// HierarchyRule handles hierarchical relations (is_a, part_of).
type HierarchyRule struct {
	// HierarchyTypes are the relation types that represent hierarchy.
	HierarchyTypes []string

	// MinWeight is the minimum confidence for inferred relations.
	MinWeight float64
}

// Name implements InferenceRule.
func (r *HierarchyRule) Name() string { return "hierarchy" }

// Apply implements InferenceRule.
func (r *HierarchyRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
	if rel.Weight < r.MinWeight {
		return nil, false
	}

	for _, hType := range r.HierarchyTypes {
		if rel.Type == hType {
			// For hierarchical relations, infer transitive closure
			return rel, true
		}
	}

	return nil, false
}

// InferTransitiveClosure infers transitive closure for hierarchical relations.
func (r *HierarchyRule) InferTransitiveClosure(g *graph.KnowledgeGraph) []*graph.Relation {
	var inferred []*graph.Relation

	for _, rel := range g.Relations() {
		if !r.isHierarchyType(rel.Type) {
			continue
		}

		// Walk only outgoing hierarchy edges: for "dog is_a mammal", the
		// ancestors of dog are reached by following is_a edges away from
		// it. KnowledgeGraph.TransitiveClosure is bidirectional and would
		// treat children as ancestors too.
		ancestors := map[string]bool{}
		queue := []string{rel.From}
		for len(queue) > 0 {
			curr := queue[0]
			queue = queue[1:]
			for _, out := range g.OutgoingRelations(curr) {
				if !r.isHierarchyType(out.Type) || ancestors[out.To] || out.To == rel.From {
					continue
				}
				ancestors[out.To] = true
				queue = append(queue, out.To)
			}
		}

		for id := range ancestors {
			if id == rel.To {
				continue
			}
			// Check if relation already exists
			if _, ok := g.GetRelation(rel.From, id, rel.Type); !ok {
				inferredRel := graph.NewRelation(rel.From, id, rel.Type, rel.Weight*0.8)
				inferred = append(inferred, inferredRel)
			}
		}
	}

	return inferred
}

func (r *HierarchyRule) isHierarchyType(relType string) bool {
	for _, hType := range r.HierarchyTypes {
		if relType == hType {
			return true
		}
	}
	return false
}

// ConfidenceAggregator defines how confidence is aggregated along paths.
type ConfidenceAggregator interface {
	// Aggregate combines multiple confidence scores.
	Aggregate(scores []float64) float64
}

// ProductAggregator multiplies confidence scores.
type ProductAggregator struct {
	// Decay is the confidence decay per hop.
	Decay float64
}

// Aggregate implements ConfidenceAggregator.
func (a *ProductAggregator) Aggregate(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}

	result := scores[0]
	for i := 1; i < len(scores); i++ {
		result *= scores[i]
		// Apply decay for longer paths
		result *= a.Decay
	}

	return result
}

// MinAggregator takes the minimum confidence score.
type MinAggregator struct {
	// Decay is the confidence decay per hop.
	Decay float64
}

// Aggregate implements ConfidenceAggregator.
func (a *MinAggregator) Aggregate(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}

	result := scores[0]
	for i := 1; i < len(scores); i++ {
		if scores[i] < result {
			result = scores[i]
		}
		// Apply decay
		result *= a.Decay
	}

	return result
}

// AverageAggregator takes the average confidence score.
type AverageAggregator struct {
	// Decay is the confidence decay per hop.
	Decay float64
}

// Aggregate implements ConfidenceAggregator.
func (a *AverageAggregator) Aggregate(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}

	sum := 0.0
	for _, s := range scores {
		sum += s
	}

	result := sum / float64(len(scores))
	// Apply decay
	result *= a.Decay

	return result
}

// DefaultAggregator returns the default confidence aggregator (product with decay).
func DefaultAggregator() ConfidenceAggregator {
	return &ProductAggregator{Decay: 0.9}
}

// EntityExtractor extracts entities from natural language text.
type EntityExtractor struct {
	// Patterns are regex patterns for entity extraction.
	Patterns map[string][]string

	// Synonyms maps entity names to alternative names.
	Synonyms map[string][]string
}

// NewEntityExtractor creates a new entity extractor with default patterns.
func NewEntityExtractor() *EntityExtractor {
	return &EntityExtractor{
		Patterns: map[string][]string{
			"person":       {`\b[A-Z][a-z]+\b`},
			"location":     {`\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b`},
			"organization": {`\b[A-Z][A-Z\s]+\b`},
		},
		Synonyms: map[string][]string{
			"go":         {"golang", "gopher"},
			"python":     {"py"},
			"javascript": {"js"},
		},
	}
}

// ExtractEntities extracts entities from text.
func (e *EntityExtractor) ExtractEntities(text string) []*graph.Entity {
	var entities []*graph.Entity
	seen := make(map[string]bool)

	words := strings.Fields(text)
	for _, word := range words {
		cleaned := strings.Trim(word, ".,;:!?\"'()[]{}")
		if len(cleaned) < 2 {
			continue
		}

		// Check if it's a known entity
		if _, ok := seen[cleaned]; !ok {
			seen[cleaned] = true
			entity := graph.NewEntity(strings.ToLower(cleaned), cleaned, graph.EntityOther)
			entities = append(entities, entity)
		}
	}

	return entities
}

// ExpandQuery expands a query with synonyms.
func (e *EntityExtractor) ExpandQuery(query string) []string {
	var expanded []string
	words := strings.Fields(query)

	for _, word := range words {
		cleaned := strings.ToLower(strings.Trim(word, ".,;:!?\"'()[]{}"))
		expanded = append(expanded, cleaned)

		// Add synonyms
		if synonyms, ok := e.Synonyms[cleaned]; ok {
			expanded = append(expanded, synonyms...)
		}
	}

	return expanded
}
