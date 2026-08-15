package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Pattern tests ---

func TestDefaultPatterns(t *testing.T) {
	patterns := DefaultPatterns()
	require.NotEmpty(t, patterns, "expected default patterns")
	for _, p := range patterns {
		assert.NotEmpty(t, p.Name, "pattern name should not be empty")
		assert.NotNil(t, p.Regex, "pattern regex should not be nil")
	}
}

func TestPatternRelationExtractor_ExtractRelations(t *testing.T) {
	extractor := &PatternRelationExtractor{Patterns: DefaultPatterns()}

	relations := extractor.ExtractRelations("Alice works at Google in Mountain View")
	require.NotEmpty(t, relations, "expected relations to be extracted")

	// Check that we found a "works_at" relation
	found := false
	for _, r := range relations {
		if r.Type == "works_at" {
			found = true
			assert.Equal(t, "alice", r.From, "from should be alice")
			assert.Equal(t, "google", r.To, "to should be google")
		}
	}
	require.True(t, found, "expected 'works_at' relation")
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
	assert.GreaterOrEqual(t, worksAt, 1, "expected at least one works_at relation")
	assert.GreaterOrEqual(t, ceoOf, 1, "expected at least one ceo_of relation")
}

func TestPatternRelationExtractor_NoMatch(t *testing.T) {
	extractor := &PatternRelationExtractor{Patterns: DefaultPatterns()}

	relations := extractor.ExtractRelations("the quick brown fox jumps over the lazy dog")
	assert.Empty(t, relations, "expected no relations")
}

func TestExtractEntitiesWithPatterns(t *testing.T) {
	patterns := DefaultPatterns()
	entities := ExtractEntitiesWithPatterns("Alice works at Google", patterns)

	require.NotEmpty(t, entities, "expected entities to be extracted")

	// Check that alice and google are extracted
	ids := make(map[string]bool)
	for _, e := range entities {
		ids[e.ID] = true
	}
	assert.True(t, ids["alice"], "expected 'alice' entity")
	assert.True(t, ids["google"], "expected 'google' entity")
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
	assert.Equal(t, 1, googleCount, "expected google to appear once")
}

// --- NERExtractor mock tests ---

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
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(entities), 2, "expected at least 2 entities")
}

// --- Mockery-generated NERExtractor mock tests ---

func TestMockNERExtractor_Extract(t *testing.T) {
	m := new(MockNERExtractor)
	expected := []*Entity{
		NewEntity("alice", "Alice", EntityPerson),
	}
	m.On("Extract", "Alice works at Google").Return(expected, nil)

	entities, err := m.Extract("Alice works at Google")

	require.NoError(t, err)
	assert.Len(t, entities, 1)
	assert.Equal(t, "alice", entities[0].ID)
	m.AssertExpectations(t)
}

func TestMockNERExtractor_Extract_Error(t *testing.T) {
	m := new(MockNERExtractor)
	m.On("Extract", "fail").Return([]*Entity{}, assert.AnError)

	entities, err := m.Extract("fail")

	assert.Error(t, err)
	assert.Empty(t, entities)
	m.AssertExpectations(t)
}
