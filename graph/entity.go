// Package graph provides a knowledge graph for storing and querying
// structured relationships between entities extracted from text.
package graph

import (
	"fmt"
	"strings"
)

// EntityType classifies an entity.
type EntityType string

const (
	EntityPerson    EntityType = "person"
	EntityConcept   EntityType = "concept"
	EntityDocument  EntityType = "document"
	EntityLocation  EntityType = "location"
	EntityOrganizer EntityType = "organization"
	EntityOther     EntityType = "other"
)

// Entity represents a node in the knowledge graph.
type Entity struct {
	// ID is a unique identifier.
	ID string

	// Label is a human-readable name.
	Label string

	// Type classifies the entity.
	Type EntityType

	// Properties are arbitrary key-value metadata.
	Properties map[string]string

	// SourceChunks tracks which chunks this entity was extracted from.
	SourceChunks []string
}

// NewEntity creates a new entity with the given ID, label, and type.
func NewEntity(id, label string, typ EntityType) *Entity {
	return &Entity{
		ID:           id,
		Label:        label,
		Type:         typ,
		Properties:   make(map[string]string),
		SourceChunks: make([]string, 0),
	}
}

// SetProperty sets a property on the entity.
func (e *Entity) SetProperty(key, value string) {
	e.Properties[key] = value
}

// GetProperty returns a property value, or empty string if not present.
func (e *Entity) GetProperty(key string) string {
	return e.Properties[key]
}

// AddSourceChunk adds a chunk ID to the entity's source list.
func (e *Entity) AddSourceChunk(chunkID string) {
	for _, id := range e.SourceChunks {
		if id == chunkID {
			return
		}
	}
	e.SourceChunks = append(e.SourceChunks, chunkID)
}

// String returns a human-readable representation.
func (e *Entity) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s (%s)", e.Label, e.Type)
	if len(e.SourceChunks) > 0 {
		fmt.Fprintf(&b, " [from %d chunks]", len(e.SourceChunks))
	}
	return b.String()
}

// Entities is a sortable slice of *Entity.
type Entities []*Entity

func (e Entities) Len() int           { return len(e) }
func (e Entities) Less(i, j int) bool { return e[i].Label < e[j].Label }
func (e Entities) Swap(i, j int)      { e[i], e[j] = e[j], e[i] }