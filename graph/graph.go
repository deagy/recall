// Package graph provides a knowledge graph for storing and querying
// structured relationships between entities extracted from text.
package graph

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// KnowledgeGraph stores entities and their relationships, supporting
// traversal, querying, and inference operations.
type KnowledgeGraph struct {
	mu        sync.RWMutex
	entities  map[string]*Entity
	relations []*Relation
	outEdges  map[string][]*Relation // from -> relations
	inEdges   map[string][]*Relation // to -> relations
}

// NewKnowledgeGraph creates a new empty knowledge graph.
func NewKnowledgeGraph() *KnowledgeGraph {
	return &KnowledgeGraph{
		entities: make(map[string]*Entity),
		outEdges: make(map[string][]*Relation),
		inEdges:  make(map[string][]*Relation),
	}
}

// AddEntity adds an entity to the graph. Returns true if added, false if already exists.
func (g *KnowledgeGraph) AddEntity(e *Entity) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, exists := g.entities[e.ID]; exists {
		return false
	}
	g.entities[e.ID] = e
	return true
}

// AddRelation adds a relation to the graph. Returns true if added.
func (g *KnowledgeGraph) AddRelation(r *Relation) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Ensure source/target entities exist
	if _, ok := g.entities[r.From]; !ok {
		return false
	}
	if _, ok := g.entities[r.To]; !ok {
		return false
	}

	g.relations = append(g.relations, r)
	g.outEdges[r.From] = append(g.outEdges[r.From], r)
	g.inEdges[r.To] = append(g.inEdges[r.To], r)
	return true
}

// GetEntity returns an entity by ID.
func (g *KnowledgeGraph) GetEntity(id string) (*Entity, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	e, ok := g.entities[id]
	return e, ok
}

// GetRelation returns a relation matching from/to/type.
func (g *KnowledgeGraph) GetRelation(from, to, relType string) (*Relation, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	for _, r := range g.relations {
		if r.From == from && r.To == to && r.Type == relType {
			return r, true
		}
	}
	return nil, false
}

// Entities returns all entities sorted by label.
func (g *KnowledgeGraph) Entities() Entities {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make(Entities, 0, len(g.entities))
	for _, e := range g.entities {
		result = append(result, e)
	}
	sort.Sort(result)
	return result
}

// Relations returns all relations.
func (g *KnowledgeGraph) Relations() Relations {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make(Relations, len(g.relations))
	copy(result, g.relations)
	return result
}

// Count returns the number of entities.
func (g *KnowledgeGraph) Count() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.entities)
}

// RelationCount returns the number of relations.
func (g *KnowledgeGraph) RelationCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.relations)
}

// OutgoingRelations returns all relations from an entity.
func (g *KnowledgeGraph) OutgoingRelations(entityID string) []*Relation {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make([]*Relation, len(g.outEdges[entityID]))
	copy(result, g.outEdges[entityID])
	return result
}

// IncomingRelations returns all relations to an entity.
func (g *KnowledgeGraph) IncomingRelations(entityID string) []*Relation {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make([]*Relation, len(g.inEdges[entityID]))
	copy(result, g.inEdges[entityID])
	return result
}

// Neighbors returns all entities directly connected to the given entity.
func (g *KnowledgeGraph) Neighbors(entityID string) []*Entity {
	g.mu.RLock()
	defer g.mu.RUnlock()

	seen := make(map[string]bool)
	var result []*Entity

	for _, r := range g.outEdges[entityID] {
		if !seen[r.To] {
			seen[r.To] = true
			if e, ok := g.entities[r.To]; ok {
				result = append(result, e)
			}
		}
	}
	for _, r := range g.inEdges[entityID] {
		if !seen[r.From] {
			seen[r.From] = true
			if e, ok := g.entities[r.From]; ok {
				result = append(result, e)
			}
		}
	}
	return result
}

// FindEntitiesByType returns all entities of a given type.
func (g *KnowledgeGraph) FindEntitiesByType(typ EntityType) Entities {
	g.mu.RLock()
	defer g.mu.RUnlock()
	var result Entities
	for _, e := range g.entities {
		if e.Type == typ {
			result = append(result, e)
		}
	}
	sort.Sort(result)
	return result
}

// FindEntitiesByLabel returns all entities whose label contains the query (case-insensitive).
func (g *KnowledgeGraph) FindEntitiesByLabel(query string) Entities {
	g.mu.RLock()
	defer g.mu.RUnlock()
	var result Entities
	lower := strings.ToLower(query)
	for _, e := range g.entities {
		if strings.Contains(strings.ToLower(e.Label), lower) {
			result = append(result, e)
		}
	}
	sort.Sort(result)
	return result
}

// RemoveEntity removes an entity and all its relations.
func (g *KnowledgeGraph) RemoveEntity(id string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, ok := g.entities[id]; !ok {
		return false
	}
	delete(g.entities, id)

	// Remove outgoing relations
	for _, r := range g.outEdges[id] {
		// Remove from target's in-edges
		var newIn []*Relation
		for _, ir := range g.inEdges[r.To] {
			if ir != r {
				newIn = append(newIn, ir)
			}
		}
		g.inEdges[r.To] = newIn
		// Remove from global relations
		var newRel []*Relation
		for _, gr := range g.relations {
			if gr != r {
				newRel = append(newRel, gr)
			}
		}
		g.relations = newRel
	}
	delete(g.outEdges, id)

	// Remove incoming relations
	for _, r := range g.inEdges[id] {
		var newOut2 []*Relation
		for _, or := range g.outEdges[r.From] {
			if or != r {
				newOut2 = append(newOut2, or)
			}
		}
		g.outEdges[r.From] = newOut2
		var newRel []*Relation
		for _, gr := range g.relations {
			if gr != r {
				newRel = append(newRel, gr)
			}
		}
		g.relations = newRel
	}
	delete(g.inEdges, id)

	return true
}

// String returns a summary of the graph.
func (g *KnowledgeGraph) String() string {
	return fmt.Sprintf("KnowledgeGraph{entities: %d, relations: %d}", g.Count(), g.RelationCount())
}
