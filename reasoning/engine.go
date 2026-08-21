// Package reasoning provides a multi-hop reasoning engine for the knowledge graph.
// It supports inference rules, confidence propagation, and depth-limited path exploration.
package reasoning

import (
	"fmt"
	"strings"

	"github.com/deagy/recall/graph"
)

// Engine provides multi-hop reasoning over the knowledge graph.
type Engine struct {
	graph         *graph.KnowledgeGraph
	rules         []InferenceRule
	maxDepth      int
	minConfidence float64
}

// Config holds configuration for the reasoning engine.
type Config struct {
	// MaxDepth is the maximum depth for path exploration (default: 3).
	MaxDepth int

	// MinConfidence is the minimum confidence threshold for inferred relations (default: 0.3).
	MinConfidence float64

	// Rules are the inference rules to apply.
	Rules []InferenceRule
}

// DefaultConfig returns a config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxDepth:      3,
		MinConfidence: 0.3,
		Rules:         DefaultRules(),
	}
}

// NewEngine creates a new reasoning engine.
func NewEngine(g *graph.KnowledgeGraph, cfg Config) *Engine {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = 3
	}
	if cfg.MinConfidence < 0 {
		cfg.MinConfidence = 0.3
	}
	if len(cfg.Rules) == 0 {
		cfg.Rules = DefaultRules()
	}
	return &Engine{
		graph:         g,
		rules:         cfg.Rules,
		maxDepth:      cfg.MaxDepth,
		minConfidence: cfg.MinConfidence,
	}
}

// Graph returns the knowledge graph this engine reasons over. It is nil when
// the engine was created with a nil graph.
func (e *Engine) Graph() *graph.KnowledgeGraph {
	return e.graph
}

// InferredRelation represents a relation inferred through reasoning.
type InferredRelation struct {
	// From is the source entity ID.
	From string

	// To is the target entity ID.
	To string

	// Type is the inferred relation type.
	Type string

	// Confidence is the propagated confidence score.
	Confidence float64

	// Path is the chain of relations that led to this inference.
	Path []*graph.Relation

	// Rule is the rule that produced this inference.
	Rule string
}

// String returns a human-readable representation.
func (ir *InferredRelation) String() string {
	return fmt.Sprintf("InferredRelation{%s --[%s, conf=%.2f]--> %s (via %s)}", ir.From, ir.Type, ir.Confidence, ir.To, ir.Rule)
}

// InferRelations finds all relations that can be inferred from the graph using
// the configured rules. Inferred relations with a confidence below the
// engine's configured MinConfidence are dropped.
func (e *Engine) InferRelations() []*InferredRelation {
	var inferred []*InferredRelation
	seen := make(map[string]bool)

	relations := e.graph.Relations()
	for _, rel := range relations {
		for _, rule := range e.rules {
			if ir, ok := rule.Apply(rel); ok {
				key := ir.From + "|" + ir.To + "|" + ir.Type
				if !seen[key] && ir.Weight >= e.minConfidence {
					seen[key] = true
					inferred = append(inferred, &InferredRelation{
						From:       ir.From,
						To:         ir.To,
						Type:       ir.Type,
						Confidence: ir.Weight,
						Path:       []*graph.Relation{rel},
						Rule:       rule.Name(),
					})
				}
			}
		}
	}

	return inferred
}

// ExplorePaths finds shortest-hop paths between two entities up to maxDepth
// hops. Entities are visited at most once per search, so the result is a
// breadth-first search tree: at most one path per intermediate entity, not
// the full set of all simple paths between the two endpoints.
func (e *Engine) ExplorePaths(from, to string) []*graph.Path {
	g := e.graph
	if _, ok := g.GetEntity(from); !ok {
		return nil
	}
	if _, ok := g.GetEntity(to); !ok {
		return nil
	}

	type queueItem struct {
		entityID string
		path     *graph.Path
		depth    int
	}

	visited := make(map[string]bool)
	fromEntity, _ := g.GetEntity(from)
	queue := []queueItem{{from, &graph.Path{Entities: []*graph.Entity{fromEntity}}, 0}}
	visited[from] = true

	var results []*graph.Path

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if curr.depth >= e.maxDepth {
			continue
		}

		for _, r := range g.OutgoingRelations(curr.entityID) {
			if visited[r.To] {
				continue
			}
			visited[r.To] = true

			toEntity, _ := g.GetEntity(r.To)
			newPath := &graph.Path{
				Entities:  append(append([]*graph.Entity{}, curr.path.Entities...), toEntity),
				Relations: append(append([]*graph.Relation{}, curr.path.Relations...), r),
			}

			if r.To == to {
				results = append(results, newPath)
			}

			if curr.depth+1 < e.maxDepth {
				queue = append(queue, queueItem{r.To, newPath, curr.depth + 1})
			}
		}

		// Also explore incoming edges (bidirectional)
		for _, r := range g.IncomingRelations(curr.entityID) {
			if visited[r.From] {
				continue
			}
			visited[r.From] = true

			fromEntity2, _ := g.GetEntity(r.From)
			newPath := &graph.Path{
				Entities:  append([]*graph.Entity{fromEntity2}, curr.path.Entities...),
				Relations: append([]*graph.Relation{r}, curr.path.Relations...),
			}

			if r.From == to {
				results = append(results, newPath)
			}

			if curr.depth+1 < e.maxDepth {
				queue = append(queue, queueItem{r.From, newPath, curr.depth + 1})
			}
		}
	}

	return results
}

// Reason finds inferred answers to a natural language query by exploring the graph.
// It combines vector similarity (via entity matching) with graph traversal.
// The maxHops argument is accepted for API symmetry with path exploration;
// traversal depth is bounded by the engine's configured MaxDepth.
func (e *Engine) Reason(query string, maxHops int) []*InferredRelation {
	// Extract entities from the query (simple heuristic: capitalized words)
	words := strings.Fields(query)
	var queryEntities []string
	for _, w := range words {
		if len(w) > 1 && w[0] >= 'A' && w[0] <= 'Z' {
			cleaned := strings.Trim(w, ".,;:!?\"'()[]{}")
			if len(cleaned) > 1 {
				queryEntities = append(queryEntities, strings.ToLower(cleaned))
			}
		}
	}

	if len(queryEntities) == 0 {
		return nil
	}

	// For each pair of query entities, find paths
	var results []*InferredRelation
	seen := make(map[string]bool)

	for _, from := range queryEntities {
		for _, to := range queryEntities {
			if from == to {
				continue
			}
			paths := e.ExplorePaths(from, to)
			for _, p := range paths {
				if len(p.Relations) > 0 {
					// Calculate confidence as product of relation weights
					confidence := 1.0
					for _, r := range p.Relations {
						confidence *= r.Weight
					}
					key := from + "|" + to
					if !seen[key] && confidence >= 0.1 {
						seen[key] = true
						results = append(results, &InferredRelation{
							From:       from,
							To:         to,
							Type:       "reasoned",
							Confidence: confidence,
							Path:       p.Relations,
							Rule:       "path_exploration",
						})
					}
				}
			}
		}
	}

	return results
}
