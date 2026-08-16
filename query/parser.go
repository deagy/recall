// Package query provides advanced query processing capabilities including
// intent detection, entity extraction, and query expansion.
package query

import (
	"context"
	"strings"
	"unicode"

	"github.com/deagy/recall/graph"
)

// Intent represents the type of query being asked.
type Intent string

const (
	IntentFactual     Intent = "factual"     // "What is X?"
	IntentComparative Intent = "comparative" // "How does X compare to Y?"
	IntentTemporal    Intent = "temporal"    // "What happened in 2020?"
	IntentCausal      Intent = "causal"      // "Why did X happen?"
	IntentProcedural  Intent = "procedural"  // "How do I do X?"
	IntentExistential Intent = "existential" // "Does X exist?"
	IntentUnknown     Intent = "unknown"
)

// String returns the string representation of the intent.
func (i Intent) String() string {
	return string(i)
}

// ParsedQuery represents a parsed and analyzed query.
type ParsedQuery struct {
	// Original is the raw query string.
	Original string

	// Intent is the detected query intent.
	Intent Intent

	// Entities are the extracted entities with confidence scores.
	Entities []ExtractedEntity

	// Relations are the extracted relations.
	Relations []ExtractedRelation

	// SubQueries are decomposed sub-queries for complex questions.
	SubQueries []string

	// Filters are structured filters to apply during retrieval.
	Filters []Filter

	// Confidence is the overall confidence in the parsing (0.0 to 1.0).
	Confidence float64
}

// ExtractedEntity represents an entity extracted from a query.
type ExtractedEntity struct {
	// Text is the original text span.
	Text string

	// Type is the entity type (person, location, organization, etc.).
	Type string

	// Confidence is the confidence score (0.0 to 1.0).
	Confidence float64
}

// ExtractedRelation represents a relation extracted from a query.
type ExtractedRelation struct {
	// From is the source entity text.
	From string

	// To is the target entity text.
	To string

	// Type is the relation type.
	Type string

	// Confidence is the confidence score (0.0 to 1.0).
	Confidence float64
}

// Filter represents a structured filter for retrieval.
type Filter struct {
	// Key is the filter attribute name.
	Key string

	// Op is the comparison operator (eq, ne, gt, lt, gte, lte, in, contains).
	Op string

	// Value is the filter value.
	Value interface{}
}

// QueryParser defines the interface for parsing and analyzing queries.
type QueryParser interface {
	// Parse analyzes a query and returns a ParsedQuery.
	Parse(ctx context.Context, query string) (*ParsedQuery, error)
}

// DefaultParser is the default query parser with intent detection and entity extraction.
type DefaultParser struct {
	// Graph is an optional knowledge graph for entity disambiguation.
	Graph *graph.KnowledgeGraph

	// Stopwords are words to ignore during parsing.
	Stopwords map[string]bool

	// TemporalKeywords are words that indicate temporal queries.
	TemporalKeywords []string

	// CausalKeywords are words that indicate causal queries.
	CausalKeywords []string

	// ComparativeKeywords are words that indicate comparative queries.
	ComparativeKeywords []string

	// ProceduralKeywords are words that indicate procedural queries.
	ProceduralKeywords []string
}

// NewDefaultParser creates a new DefaultParser with sensible defaults.
func NewDefaultParser(g *graph.KnowledgeGraph) *DefaultParser {
	return &DefaultParser{
		Graph: g,
		Stopwords: map[string]bool{
			"the": true, "a": true, "an": true, "is": true, "are": true,
			"was": true, "were": true, "be": true, "been": true, "being": true,
			"have": true, "has": true, "had": true, "do": true, "does": true,
			"did": true, "will": true, "would": true, "could": true, "should": true,
			"may": true, "might": true, "shall": true, "can": true, "need": true,
			"must": true, "to": true, "of": true, "in": true, "for": true,
			"on": true, "with": true, "at": true, "by": true, "from": true,
			"as": true, "into": true, "through": true, "during": true,
		},
		TemporalKeywords: []string{"when", "where", "year", "month", "day", "date",
			"time", "era", "century", "decade", "age", "period", "duration"},
		CausalKeywords: []string{"why", "because", "reason", "cause", "effect",
			"consequence", "result", "lead", "result", "due", "since", "therefore"},
		ComparativeKeywords: []string{"compare", "difference", "similar", "versus",
			"vs", "unlike", "whereas", "compared", "relative", "contrast"},
		ProceduralKeywords: []string{"how", "process", "method", "way", "step",
			"procedure", "algorithm", "implement", "create", "build", "make"},
	}
}

// Parse analyzes a query and returns a ParsedQuery.
func (p *DefaultParser) Parse(ctx context.Context, query string) (*ParsedQuery, error) {
	if query == "" {
		return nil, nil
	}

	// Detect intent
	intent := p.detectIntent(query)

	// Extract entities
	entities := p.extractEntities(query)

	// Extract relations (if graph is available)
	var relations []ExtractedRelation
	if p.Graph != nil && len(entities) >= 2 {
		relations = p.extractRelations(entities)
	}

	// Decompose query if needed
	subQueries := p.decomposeQuery(query, intent)

	// Extract filters
	filters := p.extractFilters(query)

	// Calculate confidence
	confidence := p.calculateConfidence(intent, entities, relations)

	return &ParsedQuery{
		Original:   query,
		Intent:     intent,
		Entities:   entities,
		Relations:  relations,
		SubQueries: subQueries,
		Filters:    filters,
		Confidence: confidence,
	}, nil
}

// detectIntent determines the intent of the query.
func (p *DefaultParser) detectIntent(query string) Intent {
	lower := strings.ToLower(query)

	// Check for temporal keywords
	for _, kw := range p.TemporalKeywords {
		if strings.Contains(lower, kw) {
			return IntentTemporal
		}
	}

	// Check for causal keywords
	for _, kw := range p.CausalKeywords {
		if strings.Contains(lower, kw) {
			return IntentCausal
		}
	}

	// Check for comparative keywords
	for _, kw := range p.ComparativeKeywords {
		if strings.Contains(lower, kw) {
			return IntentComparative
		}
	}

	// Check for procedural keywords
	for _, kw := range p.ProceduralKeywords {
		if strings.Contains(lower, kw) {
			return IntentProcedural
		}
	}

	// Check for existential queries
	if strings.HasPrefix(lower, "does ") || strings.HasPrefix(lower, "is ") ||
		strings.HasPrefix(lower, "are ") || strings.HasPrefix(lower, "has ") ||
		strings.HasPrefix(lower, "have ") || strings.HasPrefix(lower, "did ") ||
		strings.HasPrefix(lower, "do ") || strings.HasPrefix(lower, "will ") ||
		strings.HasPrefix(lower, "can ") || strings.HasPrefix(lower, "is there") ||
		strings.HasPrefix(lower, "are there") {
		return IntentExistential
	}

	// Check for comparative patterns (X vs Y, compare X and Y)
	if strings.Contains(lower, " vs ") || strings.Contains(lower, " versus ") ||
		strings.Contains(lower, " compare ") || strings.Contains(lower, " and ") {
		return IntentComparative
	}

	// Default to factual
	return IntentFactual
}

// extractEntities extracts entities from the query using capitalized words.
func (p *DefaultParser) extractEntities(query string) []ExtractedEntity {
	var entities []ExtractedEntity
	words := strings.Fields(query)

	for _, word := range words {
		// Clean the word
		cleaned := strings.Trim(word, ".,;:!?\"'()[]{}")
		if len(cleaned) < 2 {
			continue
		}

		// Skip stopwords
		if p.Stopwords[strings.ToLower(cleaned)] {
			continue
		}

		// Check if word starts with uppercase (potential entity)
		if len(cleaned) > 0 && unicode.IsUpper(rune(cleaned[0])) {
			entityType := "unknown"
			confidence := 0.7

			// Try to disambiguate entity type using graph
			if p.Graph != nil {
				if _, ok := p.Graph.GetEntity(strings.ToLower(cleaned)); ok {
					confidence = 0.9
				}
			}

			entities = append(entities, ExtractedEntity{
				Text:       cleaned,
				Type:       entityType,
				Confidence: confidence,
			})
		}
	}

	return entities
}

// extractRelations extracts relations between entities.
func (p *DefaultParser) extractRelations(entities []ExtractedEntity) []ExtractedRelation {
	var relations []ExtractedRelation

	for i := 0; i < len(entities); i++ {
		for j := i + 1; j < len(entities); j++ {
			// Check if relation exists in graph
			if p.Graph != nil {
				fromID := strings.ToLower(entities[i].Text)
				toID := strings.ToLower(entities[j].Text)

				rel, ok := p.Graph.GetRelation(fromID, toID, "")
				if ok {
					relations = append(relations, ExtractedRelation{
						From:       entities[i].Text,
						To:         entities[j].Text,
						Type:       rel.Type,
						Confidence: rel.Weight,
					})
				}
			}
		}
	}

	return relations
}

// decomposeQuery breaks down complex queries into sub-queries.
func (p *DefaultParser) decomposeQuery(query string, intent Intent) []string {
	// For simple queries, return the original
	if intent == IntentFactual || intent == IntentExistential {
		return []string{query}
	}

	// For comparative queries, split into two parts
	if intent == IntentComparative {
		parts := strings.Split(query, " vs ")
		if len(parts) == 2 {
			return []string{"What is " + parts[0] + "?", "What is " + parts[1] + "?"}
		}
	}

	// For causal queries, split into cause and effect
	if intent == IntentCausal {
		return []string{"What caused this?", "What was the effect?"}
	}

	return []string{query}
}

// extractFilters extracts structured filters from the query.
func (p *DefaultParser) extractFilters(query string) []Filter {
	var filters []Filter
	lower := strings.ToLower(query)

	// Extract year filters (e.g., "in 2020")
	if idx := strings.Index(lower, " in "); idx >= 0 {
		after := query[idx+4:]
		for i := 0; i < len(after); i++ {
			if !unicode.IsDigit(rune(after[i])) {
				continue
			}
			yearStart := i
			for j := i; j < len(after) && unicode.IsDigit(rune(after[j])); j++ {
				i = j
			}
			year := after[yearStart : i+1]
			if len(year) == 4 {
				filters = append(filters, Filter{
					Key:   "year",
					Op:    "eq",
					Value: year,
				})
				break
			}
		}
	}

	return filters
}

// calculateConfidence calculates the overall confidence in the parsing.
func (p *DefaultParser) calculateConfidence(intent Intent, entities []ExtractedEntity, relations []ExtractedRelation) float64 {
	confidence := 0.5 // Base confidence

	// Increase confidence if entities are found
	if len(entities) > 0 {
		confidence += 0.2
	}

	// Increase confidence if relations are found
	if len(relations) > 0 {
		confidence += 0.2
	}

	// Increase confidence for clear intents
	if intent != IntentUnknown {
		confidence += 0.1
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}
