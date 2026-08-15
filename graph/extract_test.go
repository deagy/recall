package graph

import (
	"testing"
)

func TestDefaultPatterns(t *testing.T) {
	patterns := DefaultPatterns()
	if len(patterns) == 0 {
		t.Fatal("expected default patterns")
	}
	for _, p := range patterns {
		if p.Name == "" {
			t.Fatal("pattern name should not be empty")
		}
		if p.Regex == nil {
			t.Fatal("pattern regex should not be nil")
		}
	}
}

func TestPatternRelationExtractor_ExtractRelations(t *testing.T) {
	extractor := &PatternRelationExtractor{Patterns: DefaultPatterns()}

	relations := extractor.ExtractRelations("Alice works at Google in Mountain View")
	if len(relations) == 0 {
		t.Fatal("expected relations to be extracted")
	}

	// Check that we found a "works_at" relation
	found := false
	for _, r := range relations {
		if r.Type == "works_at" {
			found = true
			if r.From != "alice" || r.To != "google" {
				t.Fatalf("expected alice -> google, got %s -> %s", r.From, r.To)
			}
		}
	}
	if !found {
		t.Fatal("expected 'works_at' relation")
	}
}

func TestPatternRelationExtractor_ExtractRelations_Multiple(t *testing.T) {
	extractor := &PatternRelationExtractor{Patterns: DefaultPatterns()}

	text := "Alice works at Google. Bob is the CEO of Microsoft."
	relations := extractor.ExtractRelations(text)

	worksAt := 0
	ceoOf := 0
	for _, r := range relations {
		switch r.Type {
		case "works_at":
			worksAt++
		case "ceo_of":
			ceoOf++
		}
	}
	if worksAt < 1 {
		t.Fatal("expected at least one works_at relation")
	}
	if ceoOf < 1 {
		t.Fatal("expected at least one ceo_of relation")
	}
}

func TestPatternRelationExtractor_NoMatch(t *testing.T) {
	extractor := &PatternRelationExtractor{Patterns: DefaultPatterns()}

	relations := extractor.ExtractRelations("the quick brown fox jumps over the lazy dog")
	if len(relations) != 0 {
		t.Fatalf("expected no relations, got %d", len(relations))
	}
}

func TestExtractEntitiesWithPatterns(t *testing.T) {
	patterns := DefaultPatterns()
	entities := ExtractEntitiesWithPatterns("Alice works at Google", patterns)

	if len(entities) == 0 {
		t.Fatal("expected entities to be extracted")
	}

	// Check that alice and google are extracted
	ids := make(map[string]bool)
	for _, e := range entities {
		ids[e.ID] = true
	}
	if !ids["alice"] {
		t.Fatal("expected 'alice' entity")
	}
	if !ids["google"] {
		t.Fatal("expected 'google' entity")
	}
}

func TestExtractEntitiesWithPatterns_Deduplication(t *testing.T) {
	patterns := DefaultPatterns()
	entities := ExtractEntitiesWithPatterns("Alice works at Google and Google is in California", patterns)

	// Google should only appear once
	googleCount := 0
	for _, e := range entities {
		if e.ID == "google" {
			googleCount++
		}
	}
	if googleCount != 1 {
		t.Fatalf("expected google to appear once, got %d", googleCount)
	}
}

func TestNERExtractor_Interface(t *testing.T) {
	// Verify that PatternRelationExtractor can be used as a NERExtractor
	// by extracting entities from patterns
	var _ NERExtractor = &patternNERExtractor{}
}

// patternNERExtractor is a test implementation of NERExtractor using patterns.
type patternNERExtractor struct {
	patterns []*RelationPattern
}

func (e *patternNERExtractor) Extract(text string) ([]*Entity, error) {
	return ExtractEntitiesWithPatterns(text, e.patterns), nil
}

func TestNERExtractor_PatternBased(t *testing.T) {
	extractor := &patternNERExtractor{patterns: DefaultPatterns()}

	entities, err := extractor.Extract("Alice works at Google")
	if err != nil {
		t.Fatal(err)
	}
	if len(entities) < 2 {
		t.Fatalf("expected at least 2 entities, got %d", len(entities))
	}
}