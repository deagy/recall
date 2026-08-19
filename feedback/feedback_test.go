package feedback

import (
	"math"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFeedback_RelevantIrrelevant(t *testing.T) {
	fb := NewFeedback("what is recall", map[string]Label{
		"c3": LabelRelevant,
		"c1": LabelRelevant,
		"c2": LabelNotRelevant,
		"c0": LabelUnlabeled,
	})
	assert.Equal(t, []string{"c1", "c3"}, fb.Relevant())
	assert.Equal(t, []string{"c2"}, fb.Irrelevant())
	assert.True(t, fb.HasJudgment())
	assert.NotEmpty(t, fb.ID)
	assert.False(t, fb.Time.IsZero())
}

func TestFeedback_NoJudgment(t *testing.T) {
	fb := NewFeedback("q", map[string]Label{"c1": LabelUnlabeled})
	assert.False(t, fb.HasJudgment())
	assert.Empty(t, fb.Relevant())
	assert.Empty(t, fb.Irrelevant())
}

func TestFeedback_ToMetadata(t *testing.T) {
	fb := NewFeedback("what is recall", map[string]Label{
		"c1": LabelRelevant,
		"c2": LabelNotRelevant,
	})
	md := fb.ToMetadata()
	assert.Equal(t, "what is recall", core.ToString(md["feedback.query"]))
	assert.Equal(t, "c1", core.ToString(md["feedback.relevant"]))
	assert.Equal(t, "c2", core.ToString(md["feedback.irrelevant"]))
	assert.NotEmpty(t, core.ToString(md["feedback.id"]))
}

func TestCollector_Basic(t *testing.T) {
	c := NewCollector()
	c.Add(NewFeedback("q1", map[string]Label{"a": LabelRelevant}))
	c.Add(NewFeedback("q1", map[string]Label{"b": LabelRelevant, "a": LabelNotRelevant}))
	c.Add(NewFeedback("q2", map[string]Label{"c": LabelRelevant}))

	assert.Equal(t, 3, c.Count())
	assert.Len(t, c.All(), 3)
	assert.Len(t, c.ByQuery("q1"), 2)
	assert.Len(t, c.ByQuery("q2"), 1)
	assert.Empty(t, c.ByQuery("nope"))
}

func TestCollector_UnionFor(t *testing.T) {
	c := NewCollector()
	c.Add(NewFeedback("q", map[string]Label{"a": LabelRelevant, "x": LabelNotRelevant}))
	c.Add(NewFeedback("q", map[string]Label{"b": LabelRelevant, "a": LabelRelevant}))
	assert.Equal(t, []string{"a", "b"}, c.RelevantFor("q"))
	assert.Equal(t, []string{"x"}, c.IrrelevantFor("q"))
	assert.Empty(t, c.RelevantFor("other"))
}

func TestCollector_AddNil(t *testing.T) {
	c := NewCollector()
	c.Add(nil)
	assert.Equal(t, 0, c.Count())
}

func TestRocchio_VectorShift(t *testing.T) {
	query := []float32{1, 0}
	rel := [][]float32{{2, 0}, {4, 0}} // mean [3,0]
	irr := [][]float32{{0, 2}}         // mean [0,2]
	p := RocchioParams{Alpha: 1, Beta: 1, Gamma: 1, Normalize: false}

	got := Rocchio(query, rel, irr, p)
	// Q' = 1*[1,0] + 1*[3,0] - 1*[0,2] = [4,-2]
	assert.InDelta(t, 4, got[0], 1e-6)
	assert.InDelta(t, -2, got[1], 1e-6)
}

func TestRocchio_Normalizes(t *testing.T) {
	query := []float32{1, 0}
	p := RocchioParams{Alpha: 1, Beta: 0, Gamma: 0, Normalize: true}
	got := Rocchio(query, nil, nil, p)
	norm := math.Sqrt(float64(got[0]*got[0] + got[1]*got[1]))
	assert.InDelta(t, 1.0, norm, 1e-6)
}

func TestRocchio_EmptyFeedback(t *testing.T) {
	query := []float32{2, 0}
	p := RocchioParams{Alpha: 1, Beta: 0.5, Gamma: 0.3, Normalize: false}
	got := Rocchio(query, nil, nil, p)
	assert.InDelta(t, 2, got[0], 1e-6)
	assert.InDelta(t, 0, got[1], 1e-6)
}

func TestRocchio_EmptyQueryInfersDim(t *testing.T) {
	rel := [][]float32{{2, 0}, {4, 0}}
	p := RocchioParams{Alpha: 0, Beta: 1, Gamma: 0, Normalize: false}
	got := Rocchio(nil, rel, nil, p)
	require.Len(t, got, 2)
	assert.InDelta(t, 3, got[0], 1e-6)
	assert.InDelta(t, 0, got[1], 1e-6)
}

func TestRocchio_NoVectorsAtAll(t *testing.T) {
	got := Rocchio(nil, nil, nil, DefaultRocchioParams())
	assert.Empty(t, got)
}

func TestMeanVectors(t *testing.T) {
	m := MeanVectors([][]float32{{2, 4}, {4, 8}})
	assert.InDelta(t, 3, m[0], 1e-6)
	assert.InDelta(t, 6, m[1], 1e-6)
	assert.Empty(t, MeanVectors(nil))
}

func TestL2Normalize(t *testing.T) {
	got := L2Normalize([]float32{3, 4})
	assert.InDelta(t, 0.6, got[0], 1e-6)
	assert.InDelta(t, 0.8, got[1], 1e-6)
	// zero vector unchanged
	z := L2Normalize([]float32{0, 0})
	assert.Equal(t, []float32{0, 0}, z)
}

func TestCosineSimilarity(t *testing.T) {
	assert.InDelta(t, 1.0, CosineSimilarity([]float32{1, 0}, []float32{5, 0}), 1e-6)
	assert.InDelta(t, 0.0, CosineSimilarity([]float32{1, 0}, []float32{0, 1}), 1e-6)
	assert.InDelta(t, 0.0, CosineSimilarity([]float32{0, 0}, []float32{1, 1}), 1e-6)
	assert.InDelta(t, -1.0, CosineSimilarity([]float32{1, 0}, []float32{-5, 0}), 1e-6)
}

func TestRocchioTerms(t *testing.T) {
	p := TermRocchioParams{Beta: 1, Gamma: 1, MaxTerms: 10}
	got, weights := RocchioTerms(
		"the cat sat",
		[]string{"cat cat dog", "cat bird"},
		[]string{"fish fish fish"},
		p,
	)
	// cat boosted, sat kept, bird/dog added, fish demoted to 0 and dropped.
	assert.InDelta(t, 2.5, weights["cat"], 1e-9)
	assert.InDelta(t, 1.0, weights["sat"], 1e-9)
	assert.InDelta(t, 0.5, weights["bird"], 1e-9)
	assert.InDelta(t, 0.5, weights["dog"], 1e-9)
	_, hasFish := weights["fish"]
	assert.False(t, hasFish, "fish should be dropped (demoted to zero)")
	assert.Equal(t, "cat sat bird dog", got)
	assert.NotContains(t, got, "the", "stopword should be dropped")
}

func TestRocchioTerms_MaxTerms(t *testing.T) {
	p := TermRocchioParams{Beta: 0, Gamma: 0, MaxTerms: 2}
	got, _ := RocchioTerms("alpha beta gamma delta", nil, nil, p)
	// Only query terms, top 2 by weight (all 1) -> alphabetical: alpha, beta
	assert.Equal(t, "alpha beta", got)
}

func TestBoostRelevant(t *testing.T) {
	results := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "C"}},
		{Chunk: &core.Chunk{ID: "B"}},
		{Chunk: &core.Chunk{ID: "A"}},
	}
	out := BoostRelevant(results, []string{"A"})
	ids := []string{out[0].Chunk.ID, out[1].Chunk.ID, out[2].Chunk.ID}
	assert.Equal(t, []string{"A", "C", "B"}, ids)
}
