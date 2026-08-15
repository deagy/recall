package fuse

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// --- WeightedFusion tests ---

func TestWeightedFusion_Basic(t *testing.T) {
	f := NewWeightedFusion(0.7)

	s1 := map[string]float64{"a": 10, "b": 20}
	s2 := map[string]float64{"a": 5, "b": 15, "c": 30}

	result := f.Fuse(s1, s2)

	// a: 0.7*10 + 0.3*5 = 8.5
	assert.InDelta(t, 8.5, result["a"], 0.001, "a score")
	// b: 0.7*20 + 0.3*15 = 18.5
	assert.InDelta(t, 18.5, result["b"], 0.001, "b score")
	// c: 0.7*0 + 0.3*30 = 9.0
	assert.InDelta(t, 9.0, result["c"], 0.01, "c score")
}

func TestWeightedFusion_PureVector(t *testing.T) {
	f := NewWeightedFusion(1.0)
	s1 := map[string]float64{"a": 10, "b": 20}
	result := f.Fuse(s1)
	assert.InDelta(t, 10.0, result["a"], 0.001, "pure vector a")
	assert.InDelta(t, 20.0, result["b"], 0.001, "pure vector b")
}

func TestWeightedFusion_PureBM25(t *testing.T) {
	f := NewWeightedFusion(0.0)
	s2 := map[string]float64{"a": 5, "b": 15}
	result := f.Fuse(s2)
	assert.InDelta(t, 5.0, result["a"], 0.001, "pure BM25 a")
	assert.InDelta(t, 15.0, result["b"], 0.001, "pure BM25 b")
}

func TestWeightedFusion_Empty(t *testing.T) {
	f := NewWeightedFusion(0.5)
	result := f.Fuse()
	assert.Empty(t, result, "expected empty result")
}

// --- RRF Fusion tests ---

func TestRRFFusion_Basic(t *testing.T) {
	f := NewRRFFusion(60)

	// s1: a=10 (rank 1), b=5 (rank 2)
	// s2: b=20 (rank 1), c=15 (rank 2)
	s1 := map[string]float64{"a": 10, "b": 5}
	s2 := map[string]float64{"b": 20, "c": 15}

	result := f.Fuse(s1, s2)

	// a: 1/(60+1) = 1/61 ≈ 0.01639
	assert.InDelta(t, 1.0/61.0, result["a"], 0.001, "a RRF score")
	// b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.03279
	assert.InDelta(t, 1.0/62.0+1.0/61.0, result["b"], 0.001, "b RRF score")
	// c: 1/(60+2) = 1/62 ≈ 0.01613
	assert.InDelta(t, 1.0/62.0, result["c"], 0.001, "c RRF score")
}

func TestRRFFusion_SingleMap(t *testing.T) {
	f := NewRRFFusion(60)
	s1 := map[string]float64{"x": 100, "y": 50}
	result := f.Fuse(s1)
	require.Len(t, result, 2, "expected 2 results")
	// x should have higher RRF score (rank 1)
	assert.Greater(t, result["x"], result["y"], "x should rank higher than y")
}

func TestRRFFusion_Empty(t *testing.T) {
	f := NewRRFFusion(60)
	result := f.Fuse()
	assert.Empty(t, result, "expected empty result")
}

func TestRRFFusion_DefaultK(t *testing.T) {
	f := NewRRFFusion(0) // Should default to 60
	assert.Equal(t, 60, f.K, "expected default K=60")
}

// --- Mockery-generated Fusion mock tests ---

func TestMockFusion_Fuse(t *testing.T) {
	m := new(MockFusion)
	expected := map[string]float64{"a": 1.0, "b": 2.0}
	m.On("Fuse", mock.Anything, mock.Anything).Return(expected)

	s1 := map[string]float64{"a": 10}
	s2 := map[string]float64{"b": 20}
	result := m.Fuse(s1, s2)

	assert.Equal(t, expected, result)
	m.AssertExpectations(t)
}

func TestMockFusion_Fuse_Empty(t *testing.T) {
	m := new(MockFusion)
	expected := map[string]float64{}
	m.On("Fuse").Return(expected)

	result := m.Fuse()
	assert.Empty(t, result)
	m.AssertExpectations(t)
}
