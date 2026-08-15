// Package graph provides a knowledge graph for storing and querying
// structured relationships between entities extracted from text.
package graph

import (
	"fmt"
)

// Relation represents a directed edge between two entities in the knowledge graph.
type Relation struct {
	// From is the source entity ID.
	From string

	// To is the target entity ID.
	To string

	// Type classifies the relationship (e.g., "works_at", "part_of", "related_to").
	Type string

	// Weight is the confidence or strength of the relationship (0-1).
	Weight float64

	// Properties are arbitrary key-value metadata.
	Properties map[string]string

	// SourceChunks tracks which chunks this relation was extracted from.
	SourceChunks []string
}

// NewRelation creates a new relation with the given parameters.
func NewRelation(from, to, relType string, weight float64) *Relation {
	return &Relation{
		From:         from,
		To:           to,
		Type:         relType,
		Weight:       weight,
		Properties:   make(map[string]string),
		SourceChunks: make([]string, 0),
	}
}

// SetProperty sets a property on the relation.
func (r *Relation) SetProperty(key, value string) {
	r.Properties[key] = value
}

// GetProperty returns a property value, or empty string if not present.
func (r *Relation) GetProperty(key string) string {
	return r.Properties[key]
}

// AddSourceChunk adds a chunk ID to the relation's source list.
func (r *Relation) AddSourceChunk(chunkID string) {
	for _, id := range r.SourceChunks {
		if id == chunkID {
			return
		}
	}
	r.SourceChunks = append(r.SourceChunks, chunkID)
}

// String returns a human-readable representation.
func (r *Relation) String() string {
	return fmt.Sprintf("%s --[%s(%.2f)]--> %s", r.From, r.Type, r.Weight, r.To)
}

// Relations is a sortable slice of *Relation.
type Relations []*Relation

func (r Relations) Len() int           { return len(r) }
func (r Relations) Less(i, j int) bool { return r[i].Type < r[j].Type }
func (r Relations) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
