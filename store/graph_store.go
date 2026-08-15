// Package store provides the interface and implementations for the knowledge store.
package store

import (
	"context"
	"fmt"
	"strings"

	"github.com/deagy/recall/graph"
)

// GraphStore provides access to the knowledge graph for entity/relation operations.
type GraphStore interface {
	ExtractEntities(ctx context.Context, text string, sourceChunkID string) ([]*graph.Entity, error)
	ExtractRelations(ctx context.Context, text string, sourceChunkID string) ([]*graph.Relation, error)
	GetEntity(id string) (*graph.Entity, bool)
	GetRelation(from, to, relType string) (*graph.Relation, bool)
	FindPath(from, to string) *graph.Path
	TransitiveClosure(entityID string) []*graph.Entity
	CommonNeighbors(id1, id2 string) []*graph.Entity
	ShortestPathLength(from, to string) int
	Entities() graph.Entities
	Relations() graph.Relations
	Count() int
	RelationCount() int
	Neighbors(entityID string) []*graph.Entity
	FindEntitiesByLabel(query string) graph.Entities
	FindEntitiesByType(typ graph.EntityType) graph.Entities
	Graph() *graph.KnowledgeGraph
}

// MemoryGraphStore is an in-memory implementation of GraphStore.
type MemoryGraphStore struct {
	graph *graph.KnowledgeGraph
}

// NewMemoryGraphStore creates a new in-memory graph store.
func NewMemoryGraphStore() *MemoryGraphStore {
	return &MemoryGraphStore{
		graph: graph.NewKnowledgeGraph(),
	}
}

// ExtractEntities extracts entities from text using simple heuristics.
// Identifies capitalized words/phrases as potential entities with stopword filtering.
func (s *MemoryGraphStore) ExtractEntities(_ context.Context, text string, sourceChunkID string) ([]*graph.Entity, error) {
	ner := graph.NewHeuristicNER()
	entities, err := ner.Extract(text)
	if err != nil {
		return nil, fmt.Errorf("extracting entities: %w", err)
	}

	for _, e := range entities {
		e.AddSourceChunk(sourceChunkID)
		if s.graph.AddEntity(e) {
			// Re-add to graph to update source chunks
			s.graph.AddEntity(e)
		}
	}
	return entities, nil
}

// ExtractRelations extracts relations from text using pattern matching.
// Falls back to adjacent capitalized word detection if no patterns match.
func (s *MemoryGraphStore) ExtractRelations(_ context.Context, text string, sourceChunkID string) ([]*graph.Relation, error) {
	// Extract relations using patterns
	relExtractor := &graph.PatternRelationExtractor{Patterns: graph.DefaultPatterns()}
	relations := relExtractor.ExtractRelations(text)

	// If no pattern matches, fall back to adjacent capitalized word detection
	if len(relations) == 0 {
		words := strings.Fields(text)
		for i := 0; i < len(words)-1; i++ {
			w1 := strings.Trim(words[i], ".,;:!?\"'()[]{}")
			w2 := strings.Trim(words[i+1], ".,;:!?\"'()[]{}")
			if len(w1) <= 1 || len(w2) <= 1 {
				continue
			}
			if w1[0] >= 'A' && w1[0] <= 'Z' && w2[0] >= 'A' && w2[0] <= 'Z' {
				fromID := strings.ToLower(w1)
				toID := strings.ToLower(w2)
				if fromID != toID {
					rel := graph.NewRelation(fromID, toID, "related_to", 0.5)
					rel.AddSourceChunk(sourceChunkID)
					if s.graph.AddRelation(rel) {
						relations = append(relations, rel)
					}
				}
			}
		}
	} else {
		for _, r := range relations {
			r.AddSourceChunk(sourceChunkID)
			if s.graph.AddRelation(r) {
				// Re-add to graph to update source chunks
				s.graph.AddRelation(r)
			}
		}
	}
	return relations, nil
}

// --- GraphStore interface implementations ---

func (s *MemoryGraphStore) GetEntity(id string) (*graph.Entity, bool) {
	return s.graph.GetEntity(id)
}

func (s *MemoryGraphStore) GetRelation(from, to, relType string) (*graph.Relation, bool) {
	return s.graph.GetRelation(from, to, relType)
}

func (s *MemoryGraphStore) FindPath(from, to string) *graph.Path {
	return s.graph.FindPath(from, to)
}

func (s *MemoryGraphStore) TransitiveClosure(entityID string) []*graph.Entity {
	return s.graph.TransitiveClosure(entityID)
}

func (s *MemoryGraphStore) CommonNeighbors(id1, id2 string) []*graph.Entity {
	return s.graph.CommonNeighbors(id1, id2)
}

func (s *MemoryGraphStore) ShortestPathLength(from, to string) int {
	return s.graph.ShortestPathLength(from, to)
}

func (s *MemoryGraphStore) Entities() graph.Entities {
	return s.graph.Entities()
}

func (s *MemoryGraphStore) Relations() graph.Relations {
	return s.graph.Relations()
}

func (s *MemoryGraphStore) Count() int {
	return s.graph.Count()
}

func (s *MemoryGraphStore) RelationCount() int {
	return s.graph.RelationCount()
}

func (s *MemoryGraphStore) Neighbors(entityID string) []*graph.Entity {
	return s.graph.Neighbors(entityID)
}

func (s *MemoryGraphStore) FindEntitiesByLabel(query string) graph.Entities {
	return s.graph.FindEntitiesByLabel(query)
}

func (s *MemoryGraphStore) FindEntitiesByType(typ graph.EntityType) graph.Entities {
	return s.graph.FindEntitiesByType(typ)
}

func (s *MemoryGraphStore) Graph() *graph.KnowledgeGraph {
	return s.graph
}
