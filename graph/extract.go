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

// HeuristicNER extracts entities using capitalized word detection with
// stopword filtering and multi-word grouping.
type HeuristicNER struct {
	Stopwords      map[string]bool
	MinLength      int
	MinGroupLength int // minimum total length for multi-word grouping (default 8)
}

// DefaultStopwords is a list of common English stopwords that are unlikely
// to be meaningful entity names on their own.
var DefaultStopwords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true,
	"but": true, "in": true, "on": true, "at": true, "to": true,
	"for": true, "of": true, "with": true, "by": true, "from": true,
	"is": true, "are": true, "was": true, "were": true, "be": true,
	"been": true, "being": true, "have": true, "has": true, "had": true,
	"do": true, "does": true, "did": true, "will": true, "would": true,
	"could": true, "should": true, "may": true, "might": true, "shall": true,
	"can": true, "it": true, "its": true, "this": true, "that": true,
	"these": true, "those": true, "i": true, "we": true, "they": true,
	"he": true, "she": true, "you": true, "me": true, "my": true,
	"your": true, "his": true, "her": true, "our": true, "their": true,
	"what": true, "which": true, "who": true, "whom": true, "where": true,
	"when": true, "how": true, "not": true, "no": true, "nor": true,
}

// NewHeuristicNER creates a HeuristicNER with default settings.
func NewHeuristicNER() *HeuristicNER {
	return &HeuristicNER{
		Stopwords:      DefaultStopwords,
		MinLength:      2,
		MinGroupLength: 25, // only group if total length >= 25 (e.g., "New York City" = 13, "John Fitzgerald Kennedy" = 23)
	}
}

// Extract identifies capitalized words and groups consecutive capitalized words
// into multi-word entities, filtering stopwords and short tokens.
func (h *HeuristicNER) Extract(text string) ([]*Entity, error) {
	words := strings.Fields(text)
	var entities []*Entity
	seen := make(map[string]bool)

	// Group consecutive capitalized words into multi-word entities
	i := 0
	for i < len(words) {
		word := words[i]
		cleaned := strings.Trim(word, ".,;:!?\"'()[]{}")
		if len(cleaned) < h.MinLength {
			i++
			continue
		}

		// Check if word starts with uppercase
		if word[0] >= 'A' && word[0] <= 'Z' {
			// Group consecutive capitalized words
			var group []string
			for i < len(words) {
				w := words[i]
				c := strings.Trim(w, ".,;:!?\"'()[]{}")
				if len(c) >= h.MinLength && w[0] >= 'A' && w[0] <= 'Z' {
					group = append(group, c)
					i++
				} else {
					break
				}
			}

			// Skip if all words in group are stopwords
			allStop := true
			for _, g := range group {
				if !h.Stopwords[strings.ToLower(g)] {
					allStop = false
					break
				}
			}
			if allStop {
				continue
			}

			// Create entity from group
			var entityLabel string
			var entityID string

			if len(group) == 1 {
				// Single-word entity
				entityLabel = group[0]
				entityID = strings.ToLower(entityLabel)
			} else {
				// Multi-word entity: only group if total length meets threshold
				entityLabel = strings.Join(group, " ")
				entityID = strings.ToLower(entityLabel)

				// If group is too short, treat each word as a separate entity
				if len(entityLabel) < h.MinGroupLength {
					for _, g := range group {
						id := strings.ToLower(g)
						if !seen[id] {
							seen[id] = true
							entities = append(entities, NewEntity(id, g, EntityOther))
						}
					}
					continue
				}
			}

			if !seen[entityID] {
				seen[entityID] = true
				entities = append(entities, NewEntity(entityID, entityLabel, EntityOther))
			}
		} else {
			i++
		}
	}

	return entities, nil
}
