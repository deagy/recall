package query

import (
	"context"
	"strings"

	"github.com/deagy/recall/graph"
)

// Expander defines the interface for expanding queries.
type Expander interface {
	// Expand takes a parsed query and returns an expanded version.
	Expand(ctx context.Context, parsed *ParsedQuery) (*ParsedQuery, error)
}

// GraphExpander expands queries using knowledge graph relations and synonyms.
type GraphExpander struct {
	Graph *graph.KnowledgeGraph

	// Synonyms maps entity text to alternative representations.
	Synonyms map[string][]string

	// MaxExpansions limits the number of expansions per entity.
	MaxExpansions int
}

// NewGraphExpander creates a new GraphExpander.
func NewGraphExpander(g *graph.KnowledgeGraph) *GraphExpander {
	return &GraphExpander{
		Graph:         g,
		Synonyms:      defaultSynonyms(),
		MaxExpansions: 5,
	}
}

// Expand expands a parsed query using knowledge graph relations.
func (e *GraphExpander) Expand(ctx context.Context, parsed *ParsedQuery) (*ParsedQuery, error) {
	if parsed == nil {
		return nil, nil
	}

	expanded := &ParsedQuery{
		Original:   parsed.Original,
		Intent:     parsed.Intent,
		Entities:   make([]ExtractedEntity, len(parsed.Entities)),
		Relations:  make([]ExtractedRelation, len(parsed.Relations)),
		SubQueries: append([]string{}, parsed.SubQueries...),
		Filters:    append([]Filter{}, parsed.Filters...),
		Confidence: parsed.Confidence,
	}
	copy(expanded.Entities, parsed.Entities)
	copy(expanded.Relations, parsed.Relations)

	// Expand entities using synonyms and relations
	var expandedEntities []ExtractedEntity
	for _, ent := range parsed.Entities {
		expandedEntities = append(expandedEntities, ent)

		// Add synonyms
		if syns, ok := e.Synonyms[strings.ToLower(ent.Text)]; ok {
			for _, syn := range syns {
				if len(expandedEntities) < e.MaxExpansions {
					expandedEntities = append(expandedEntities, ExtractedEntity{
						Text:       syn,
						Type:       ent.Type,
						Confidence: ent.Confidence * 0.8, // Lower confidence for synonyms
					})
				}
			}
		}

		// Add related entities from graph
		if e.Graph != nil {
			entityID := strings.ToLower(ent.Text)
			if _, ok := e.Graph.GetEntity(entityID); ok {
				// Get relations for this entity
				relations := e.Graph.OutgoingRelations(entityID)
				for _, rel := range relations {
					if len(expandedEntities) < e.MaxExpansions {
						if targetEntity, ok := e.Graph.GetEntity(rel.To); ok {
							expandedEntities = append(expandedEntities, ExtractedEntity{
								Text:       targetEntity.Label,
								Type:       string(targetEntity.Type),
								Confidence: ent.Confidence * rel.Weight * 0.7,
							})
						}
					}
				}
			}
		}
	}

	expanded.Entities = expandedEntities

	// Expand relations
	var expandedRelations []ExtractedRelation
	for _, rel := range parsed.Relations {
		expandedRelations = append(expandedRelations, rel)

		// Add inverse relations if available
		if e.Graph != nil {
			fromID := strings.ToLower(rel.From)
			toID := strings.ToLower(rel.To)

			// Check for inverse relation
			if invRel, ok := e.Graph.GetRelation(toID, fromID, ""); ok {
				expandedRelations = append(expandedRelations, ExtractedRelation{
					From:       rel.To,
					To:         rel.From,
					Type:       invRel.Type,
					Confidence: rel.Confidence * 0.9,
				})
			}
		}
	}

	expanded.Relations = expandedRelations

	// Update confidence
	if len(expandedEntities) > len(parsed.Entities) || len(expandedRelations) > len(parsed.Relations) {
		expanded.Confidence = (parsed.Confidence + 0.1) // Slight confidence boost from expansion
		if expanded.Confidence > 1.0 {
			expanded.Confidence = 1.0
		}
	}

	return expanded, nil
}

// defaultSynonyms returns default synonym mappings.
func defaultSynonyms() map[string][]string {
	return map[string][]string{
		"go":         {"golang", "gopher", "go language"},
		"python":     {"py", "python language"},
		"javascript": {"js", "javascript language"},
		"typescript": {"ts", "typescript language"},
		"react":      {"reactjs", "react.js"},
		"node":       {"nodejs", "node.js", "node.js runtime"},
		"docker":     {"docker container", "docker engine"},
		"kubernetes": {"k8s", "kubernetes cluster"},
		"sql":        {"structured query language", "relational database"},
		"nosql":      {"non-relational database", "document database"},
	}
}
