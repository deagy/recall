package hitl

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnnotationStore_AddAndGet(t *testing.T) {
	s := NewAnnotationStore()
	a := NewAnnotation("c1", AnnotationRelevance, "relevant")
	a.Score = 0.9
	s.Add(a)

	assert.Equal(t, 1, s.Count())
	got, ok := s.Get(a.ID)
	require.True(t, ok)
	assert.Equal(t, "c1", got.ChunkID)
	assert.InDelta(t, 0.9, got.Score, 1e-9)
}

func TestAnnotationStore_ForChunk(t *testing.T) {
	s := NewAnnotationStore()
	a1 := NewAnnotation("c1", AnnotationRelevance, "r1")
	a2 := NewAnnotation("c1", AnnotationCorrection, "fixed text")
	a3 := NewAnnotation("c2", AnnotationFeedback, "looks good")
	s.Add(a1)
	s.Add(a2)
	s.Add(a3)

	assert.Len(t, s.ForChunk("c1"), 2)
	assert.Len(t, s.ForChunk("c2"), 1)
	assert.Empty(t, s.ForChunk("missing"))
}

func TestAnnotationStore_RelevanceFor_MostRecent(t *testing.T) {
	s := NewAnnotationStore()
	old := NewAnnotation("c1", AnnotationRelevance, "low")
	old.Score = 0.2
	old.CreatedAt = time.Now().Add(-time.Hour).UTC()
	s.Add(old)
	newer := NewAnnotation("c1", AnnotationRelevance, "high")
	newer.Score = 0.9
	newer.CreatedAt = time.Now().UTC()
	s.Add(newer)

	// A non-relevance annotation must not affect relevance lookup.
	s.Add(NewAnnotation("c1", AnnotationCorrection, "note"))

	got, ok := s.RelevanceFor("c1")
	require.True(t, ok)
	assert.InDelta(t, 0.9, got, 1e-9)

	_, ok = s.RelevanceFor("missing")
	assert.False(t, ok)
}

func TestAnnotationStore_Chunks_Sorted(t *testing.T) {
	s := NewAnnotationStore()
	s.Add(NewAnnotation("zeta", AnnotationFeedback, ""))
	s.Add(NewAnnotation("alpha", AnnotationFeedback, ""))
	s.Add(NewAnnotation("alpha", AnnotationFeedback, ""))
	assert.Equal(t, []string{"alpha", "zeta"}, s.Chunks())
}

func TestAnnotationStore_ReplaceByID_ChangeChunk(t *testing.T) {
	s := NewAnnotationStore()
	a := NewAnnotation("c1", AnnotationRelevance, "x")
	s.Add(a)
	// Replace same ID but re-point to a different chunk.
	a.ChunkID = "c2"
	s.Add(a)

	assert.Len(t, s.ForChunk("c1"), 0)
	assert.Len(t, s.ForChunk("c2"), 1)
	assert.Equal(t, 1, s.Count())
}

func TestAnnotationType_String(t *testing.T) {
	assert.Equal(t, "relevance", AnnotationRelevance.String())
	assert.Equal(t, "correction", AnnotationCorrection.String())
	assert.Equal(t, "feedback", AnnotationFeedback.String())
	assert.Equal(t, "unknown", AnnotationType(99).String())
}
