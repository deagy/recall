// Package graph provides a knowledge graph for storing and querying
// structured relationships between entities extracted from text.
package graph

import (
	"fmt"
	"sort"
	"strings"
)

// Inference provides graph traversal and inference operations.

// Path represents a sequence of entities connected by relations.
type Path struct {
	Entities []*Entity
	Relations []*Relation
}

// String returns a human-readable representation of the path.
func (p *Path) String() string {
	var b strings.Builder
	for i, e := range p.Entities {
		if i > 0 && i-1 < len(p.Relations) {
			fmt.Fprintf(&b, " --[%s]--> ", p.Relations[i-1].Type)
		}
		fmt.Fprintf(&b, "%s", e.Label)
	}
	return b.String()
}

// Length returns the number of entities in the path.
func (p *Path) Length() int {
	return len(p.Entities)
}

// FindPath finds a path between two entities using BFS.
// Returns nil if no path exists.
func (g *KnowledgeGraph) FindPath(from, to string) *Path {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if _, ok := g.entities[from]; !ok {
		return nil
	}
	if _, ok := g.entities[to]; !ok {
		return nil
	}
	if from == to {
		e := g.entities[from]
		return &Path{Entities: []*Entity{e}, Relations: []*Relation{}}
	}

	// BFS
	type queueItem struct {
		entityID string
		path     *Path
	}

	visited := make(map[string]bool)
	queue := []queueItem{{from, &Path{Entities: []*Entity{g.entities[from]}}}}
	visited[from] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		// Explore neighbors
		for _, r := range g.outEdges[curr.entityID] {
			if visited[r.To] {
				continue
			}
			visited[r.To] = true

			newPath := &Path{
				Entities:  append(append([]*Entity{}, curr.path.Entities...), g.entities[r.To]),
				Relations: append(append([]*Relation{}, curr.path.Relations...), r),
			}

			if r.To == to {
				return newPath
			}

			queue = append(queue, queueItem{r.To, newPath})
		}

		// Also explore incoming edges (bidirectional traversal)
		for _, r := range g.inEdges[curr.entityID] {
			if visited[r.From] {
				continue
			}
			visited[r.From] = true

			newPath := &Path{
				Entities:  append(append([]*Entity{}, curr.path.Entities...), g.entities[r.From]),
				Relations: append([]*Relation{r}, curr.path.Relations...),
			}

			if r.From == to {
				return newPath
			}

			queue = append(queue, queueItem{r.From, newPath})
		}
	}

	return nil
}

// TransitiveClosure computes all entities reachable from a given entity.
func (g *KnowledgeGraph) TransitiveClosure(entityID string) []*Entity {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if _, ok := g.entities[entityID]; !ok {
		return nil
	}

	visited := make(map[string]bool)
	var result []*Entity
	queue := []string{entityID}
	visited[entityID] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		result = append(result, g.entities[curr])

		for _, r := range g.outEdges[curr] {
			if !visited[r.To] {
				visited[r.To] = true
				queue = append(queue, r.To)
			}
		}
		for _, r := range g.inEdges[curr] {
			if !visited[r.From] {
				visited[r.From] = true
				queue = append(queue, r.From)
			}
		}
	}

	return result
}

// CommonNeighbors finds entities that are neighbors of both given entities.
func (g *KnowledgeGraph) CommonNeighbors(id1, id2 string) []*Entity {
	g.mu.RLock()
	defer g.mu.RUnlock()

	neighbors1 := make(map[string]bool)
	for _, r := range g.outEdges[id1] {
		neighbors1[r.To] = true
	}
	for _, r := range g.inEdges[id1] {
		neighbors1[r.From] = true
	}

	neighbors2 := make(map[string]bool)
	for _, r := range g.outEdges[id2] {
		neighbors2[r.To] = true
	}
	for _, r := range g.inEdges[id2] {
		neighbors2[r.From] = true
	}

	var common []*Entity
	for id := range neighbors1 {
		if neighbors2[id] && id != id1 && id != id2 {
			if e, ok := g.entities[id]; ok {
				common = append(common, e)
			}
		}
	}
	sort.Sort(Entities(common))
	return common
}

// ShortestPathLength returns the number of edges in the shortest path between two entities.
// Returns -1 if no path exists.
func (g *KnowledgeGraph) ShortestPathLength(from, to string) int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if _, ok := g.entities[from]; !ok {
		return -1
	}
	if _, ok := g.entities[to]; !ok {
		return -1
	}
	if from == to {
		return 0
	}

	visited := make(map[string]bool)
	queue := []string{from}
	visited[from] = true
	length := 0

	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			curr := queue[0]
			queue = queue[1:]

			for _, r := range g.outEdges[curr] {
				if r.To == to {
					return length + 1
				}
				if !visited[r.To] {
					visited[r.To] = true
					queue = append(queue, r.To)
				}
			}
			for _, r := range g.inEdges[curr] {
				if r.From == to {
					return length + 1
				}
				if !visited[r.From] {
					visited[r.From] = true
					queue = append(queue, r.From)
				}
			}
		}
		length++
	}

	return -1
}