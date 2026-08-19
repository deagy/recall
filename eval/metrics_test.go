package eval

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrecisionAtK(t *testing.T) {
	retrieved := []string{"a", "b", "c", "d"}
	relevant := map[string]bool{"a": true, "c": true, "e": true}

	assert.InDelta(t, 2.0/3.0, PrecisionAtK(retrieved, relevant, 3), 1e-9)
	assert.InDelta(t, 1.0, PrecisionAtK(retrieved, relevant, 1), 1e-9) // a is first
	assert.Equal(t, 0.0, PrecisionAtK(retrieved, relevant, 0))
	assert.InDelta(t, 2.0/10.0, PrecisionAtK(retrieved, relevant, 10), 1e-9) // divides by k
}

func TestRecallAtK(t *testing.T) {
	retrieved := []string{"a", "b", "c", "d"}
	relevant := map[string]bool{"a": true, "c": true, "e": true}

	assert.InDelta(t, 2.0/3.0, RecallAtK(retrieved, relevant, 4), 1e-9)
	assert.InDelta(t, 1.0/3.0, RecallAtK(retrieved, relevant, 1), 1e-9)
	assert.Equal(t, 0.0, RecallAtK(retrieved, map[string]bool{}, 4), "no relevant -> 0")
	// duplicates in retrieved must not double-count
	dup := []string{"a", "a", "c", "c"}
	assert.InDelta(t, 2.0/3.0, RecallAtK(dup, relevant, 4), 1e-9)
}

func TestMRR(t *testing.T) {
	relevant := map[string]bool{"a": true}
	assert.InDelta(t, 1.0, MRR([]string{"a", "b"}, relevant), 1e-9)
	assert.InDelta(t, 0.5, MRR([]string{"x", "a"}, relevant), 1e-9)
	assert.InDelta(t, 1.0/3.0, MRR([]string{"x", "y", "a"}, relevant), 1e-9)
	assert.Equal(t, 0.0, MRR([]string{"x", "y"}, relevant))
	assert.Equal(t, 0.0, MRR(nil, relevant))
}

func TestNDCGAtK_PerfectRanking(t *testing.T) {
	retrieved := []string{"a", "b", "c"}
	relevance := map[string]int{"a": 3, "b": 2, "c": 1}
	assert.InDelta(t, 1.0, NDCGAtK(retrieved, relevance, 3), 1e-9)
}

func TestNDCGAtK_ReverseRanking(t *testing.T) {
	relevance := map[string]int{"a": 3, "b": 2, "c": 1}
	reverse := NDCGAtK([]string{"c", "b", "a"}, relevance, 3)
	assert.InDelta(t, 0.6806, reverse, 1e-3)
	assert.Less(t, reverse, 1.0)
}

func TestNDCGAtK_EdgeCases(t *testing.T) {
	assert.Equal(t, 0.0, NDCGAtK([]string{"a"}, nil, 3))
	assert.Equal(t, 0.0, NDCGAtK([]string{"a"}, map[string]int{"a": 1}, 0))
	// no relevant grades present in retrieved
	assert.Equal(t, 0.0, NDCGAtK([]string{"x"}, map[string]int{"a": 2}, 1))
}

func TestNDCGAtK_KCutoff(t *testing.T) {
	retrieved := []string{"a", "b"}
	relevance := map[string]int{"a": 3, "b": 2, "c": 1}
	// DCG@2 over a,b = (2^3-1)/log2(2) + (2^2-1)/log2(3)
	// IDCG@2 over top-2 grades [3,2] = same -> NDCG 1.0
	assert.InDelta(t, 1.0, NDCGAtK(retrieved, relevance, 2), 1e-9)
}

func TestComputeRetrievalMetrics_Binary(t *testing.T) {
	retrieved := []string{"a", "b", "c"}
	m := ComputeRetrievalMetrics("q", retrieved, []string{"a", "c"}, nil, 3)
	assert.Equal(t, "q", m.Query)
	assert.Equal(t, 3, m.K)
	assert.InDelta(t, 2.0/3.0, m.Precision, 1e-9)
	assert.InDelta(t, 1.0, m.Recall, 1e-9)
	assert.InDelta(t, 1.0, m.MRR, 1e-9)
	assert.InDelta(t, 0.9197, m.NDCG, 1e-3)
	assert.Equal(t, 2, m.NumRelevant)
	assert.Equal(t, 3, m.NumRetrieved)
}

func TestComputeRetrievalMetrics_Graded(t *testing.T) {
	retrieved := []string{"a", "b", "c"}
	graded := map[string]int{"a": 3, "c": 2}
	m := ComputeRetrievalMetrics("q", retrieved, nil, graded, 3)
	// a is relevant and first -> MRR 1.0; relevant set = {a,c}
	assert.InDelta(t, 1.0, m.MRR, 1e-9)
	assert.Equal(t, 2, m.NumRelevant)
	assert.Greater(t, m.NDCG, 0.0)
}

func TestComputeRetrievalMetrics_NoRelevant(t *testing.T) {
	m := ComputeRetrievalMetrics("q", []string{"a", "b"}, nil, nil, 5)
	require.Equal(t, 0, m.NumRelevant)
	assert.Equal(t, 0.0, m.Precision)
	assert.Equal(t, 0.0, m.Recall)
	assert.Equal(t, 0.0, m.MRR)
	assert.Equal(t, 0.0, m.NDCG)
}
