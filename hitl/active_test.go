package hitl

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUncertaintyFromScores(t *testing.T) {
	u := UncertaintyFromScores([]float64{0.9, 0.5, 0.1})
	// normalized = (s-min)/(max-min); uncertainty = 1 - normalized
	assert.InDelta(t, 0.0, u[0], 1e-9) // top score -> no uncertainty
	assert.InDelta(t, 0.5, u[1], 1e-9) // (0.5-0.1)/(0.9-0.1) = 0.5
	assert.InDelta(t, 1.0, u[2], 1e-9) // bottom score -> max uncertainty
}

func TestUncertaintyFromScores_AllEqual(t *testing.T) {
	u := UncertaintyFromScores([]float64{0.5, 0.5, 0.5})
	for _, v := range u {
		assert.InDelta(t, 0.5, v, 1e-9)
	}
}

func TestUncertaintyFromScores_SingleAndEmpty(t *testing.T) {
	assert.InDelta(t, 0.5, UncertaintyFromScores([]float64{0.3})[0], 1e-9)
	assert.Nil(t, UncertaintyFromScores(nil))
}

func TestMargin(t *testing.T) {
	assert.Equal(t, 0.0, Margin(nil))
	assert.Equal(t, 0.0, Margin([]float64{0.9}))
	assert.InDelta(t, 0.4, Margin([]float64{0.9, 0.5, 0.2}), 1e-9)
	// unsorted: finds true top two
	assert.InDelta(t, 0.1, Margin([]float64{0.4, 0.9, 0.8}), 1e-9)
}

func TestActiveLearning_Select(t *testing.T) {
	q := NewReviewQueue()
	al := NewActiveLearning(q, 2)
	cands := []Candidate{
		{ChunkID: "low", Uncertainty: 0.1},
		{ChunkID: "high", Uncertainty: 0.9},
		{ChunkID: "mid", Uncertainty: 0.5},
	}
	got := al.Select(cands)
	// batch size 2 -> only the two most uncertain, most uncertain first
	assert.Equal(t, []string{"high", "mid"}, got)
	assert.Equal(t, 2, q.Count())
}

func TestActiveLearning_Select_SkipsReviewed(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("done", "r", 0.9)
	q.MarkReviewed("done", true) // already approved

	al := NewActiveLearning(q, 10)
	got := al.Select([]Candidate{
		{ChunkID: "done", Uncertainty: 0.99},
		{ChunkID: "open", Uncertainty: 0.4},
	})
	assert.Equal(t, []string{"open"}, got)
	assert.Equal(t, 1, q.Count(), "only the open item should be enqueued")
}

func TestActiveLearning_Select_NoCap(t *testing.T) {
	q := NewReviewQueue()
	al := NewActiveLearning(q, 0) // no cap
	got := al.Select([]Candidate{
		{ChunkID: "a", Uncertainty: 0.3},
		{ChunkID: "b", Uncertainty: 0.7},
	})
	assert.Equal(t, 2, len(got))
	assert.Equal(t, 2, q.Count())
}

func TestActiveLearning_Select_Empty(t *testing.T) {
	q := NewReviewQueue()
	al := NewActiveLearning(q, 5)
	assert.Empty(t, al.Select(nil))
	assert.Empty(t, (&ActiveLearning{}).Select([]Candidate{{"x", 0.5}}), "nil queue")
}
