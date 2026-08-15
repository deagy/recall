package graph

import (
	"regexp"
	"sort"
	"strings"
)

// NERExtractor defines the interface for named entity recognition.
// Users can implement this to plug in custom NER models (e.g., OpenAI, spaCy, etc.).
type NERExtractor interface {
	// Extract identifies entities in the given text and returns them.
	Extract(text string) ([]*Entity, error)
}

// PatternRelationExtractor extracts relations based on configurable text patterns.
type PatternRelationExtractor struct {
	// Patterns are regex patterns that capture relation types and entity positions.
	// Each pattern should have named capture groups: "rel" for relation type,
	// and "from"/"to" for source/target entities.
	Patterns []*RelationPattern
}

// RelationPattern defines a regex pattern for extracting a specific relation type.
type RelationPattern struct {
	// Name is the relation type (e.g., "works_at", "located_in").
	Name string

	// Regex is the compiled regex pattern.
	Regex *regexp.Regexp

	// FromIndex is the 1-based capture group index for the source entity.
	FromIndex int

	// ToIndex is the 1-based capture group index for the target entity.
	ToIndex int
}

// NewRelationPattern creates a new relation pattern from a regex string.
func NewRelationPattern(name, regexStr string, fromIndex, toIndex int) *RelationPattern {
	return &RelationPattern{
		Name:      name,
		Regex:     regexp.MustCompile(regexStr),
		FromIndex: fromIndex,
		ToIndex:   toIndex,
	}
}

// DefaultPatterns returns common relation extraction patterns.
func DefaultPatterns() []*RelationPattern {
	return []*RelationPattern{
		NewRelationPattern("works_at", `(?i)(\w+)\s+works?\s+(?:at|for)\s+(\w+)`, 1, 2),
		NewRelationPattern("located_in", `(?i)(\w+)\s+is\s+(?:located\s+)?in\s+(\w+)`, 1, 2),
		NewRelationPattern("founded_by", `(?i)(\w+)\s+was?\s+founded\s+by\s+(\w+)`, 1, 2),
		NewRelationPattern("part_of", `(?i)(\w+)\s+(?:is\s+)?part\s+of\s+(\w+)`, 1, 2),
		NewRelationPattern("related_to", `(?i)(\w+)\s+(?:is\s+)?related\s+to\s+(\w+)`, 1, 2),
		NewRelationPattern("taught_by", `(?i)(\w+)\s+was?\s+taught\s+by\s+(\w+)`, 1, 2),
		NewRelationPattern("parent_of", `(?i)(\w+)\s+is\s+the\s+parent\s+of\s+(\w+)`, 1, 2),
		NewRelationPattern("ceo_of", `(?i)(\w+)\s+is\s+the\s+CEO\s+of\s+(\w+)`, 1, 2),
	}
}

// ExtractRelations extracts relations from text using the configured patterns.
func (e *PatternRelationExtractor) ExtractRelations(text string) []*Relation {
	var relations []*Relation

	for _, pattern := range e.Patterns {
		matches := pattern.Regex.FindAllStringSubmatch(text, -1)
		for _, match := range matches {
			from := strings.TrimSpace(match[pattern.FromIndex])
			to := strings.TrimSpace(match[pattern.ToIndex])

			if from == "" || to == "" || from == to {
				continue
			}

			fromID := strings.ToLower(from)
			toID := strings.ToLower(to)

			rel := NewRelation(fromID, toID, pattern.Name, 0.8)
			relations = append(relations, rel)
		}
	}

	return relations
}

// ExtractEntitiesWithPatterns extracts entities from text using relation patterns.
// Entities are those that appear in any pattern match.
func ExtractEntitiesWithPatterns(text string, patterns []*RelationPattern) []*Entity {
	seen := make(map[string]bool)
	var entities []*Entity

	for _, pattern := range patterns {
		matches := pattern.Regex.FindAllStringSubmatch(text, -1)
		for _, match := range matches {
			from := strings.TrimSpace(match[pattern.FromIndex])
			to := strings.TrimSpace(match[pattern.ToIndex])

			for _, word := range []string{from, to} {
				if word == "" {
					continue
				}
				id := strings.ToLower(word)
				if !seen[id] {
					seen[id] = true
					entities = append(entities, NewEntity(id, word, EntityOther))
				}
			}
		}
	}

	sort.Sort(Entities(entities))
	return entities
}
