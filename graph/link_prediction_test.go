package graph

import (
	"testing"
)

func TestEvaluateMetrics_AddResult(t *testing.T) {
	metrics := NewEvaluateMetrics()

	// Add some results
	metrics.AddResult(1, []int{1, 3, 5, 10})
	metrics.AddResult(2, []int{1, 3, 5, 10})
	metrics.AddResult(3, []int{1, 3, 5, 10})

	if metrics.TotalTests != 3 {
		t.Errorf("Expected 3 tests, got %d", metrics.TotalTests)
	}
}

func TestEvaluateMetrics_Finalize(t *testing.T) {
	metrics := NewEvaluateMetrics()

	// Add results
	metrics.AddResult(1, []int{1, 3, 5, 10})
	metrics.AddResult(2, []int{1, 3, 5, 10})
	metrics.AddResult(5, []int{1, 3, 5, 10})

	metrics.Finalize()

	// MRR = (1/1 + 1/2 + 1/5) / 3 = (1 + 0.5 + 0.2) / 3 = 0.5667
	expectedMRR := (1.0 + 0.5 + 0.2) / 3.0
	if metrics.MeanReciprocalRank < expectedMRR-0.001 || metrics.MeanReciprocalRank > expectedMRR+0.001 {
		t.Errorf("MRR = %v, want %v", metrics.MeanReciprocalRank, expectedMRR)
	}

	// Mean Rank = (1 + 2 + 5) / 3 = 2.6667
	expectedMeanRank := 8.0 / 3.0
	if metrics.MeanRank < expectedMeanRank-0.001 || metrics.MeanRank > expectedMeanRank+0.001 {
		t.Errorf("MeanRank = %v, want %v", metrics.MeanRank, expectedMeanRank)
	}

	// Hits@1 = 1/3 = 0.3333
	expectedHits1 := 1.0 / 3.0
	if metrics.HitsAtK[1] < expectedHits1-0.001 || metrics.HitsAtK[1] > expectedHits1+0.001 {
		t.Errorf("Hits@1 = %v, want %v", metrics.HitsAtK[1], expectedHits1)
	}

	// Hits@3 = 2/3 = 0.6667
	expectedHits3 := 2.0 / 3.0
	if metrics.HitsAtK[3] < expectedHits3-0.001 || metrics.HitsAtK[3] > expectedHits3+0.001 {
		t.Errorf("Hits@3 = %v, want %v", metrics.HitsAtK[3], expectedHits3)
	}

	// Hits@5 = 3/3 = 1.0
	expectedHits5 := 1.0
	if metrics.HitsAtK[5] < expectedHits5-0.001 || metrics.HitsAtK[5] > expectedHits5+0.001 {
		t.Errorf("Hits@5 = %v, want %v", metrics.HitsAtK[5], expectedHits5)
	}
}

func TestEvaluateMetrics_FinalizeEmpty(t *testing.T) {
	metrics := NewEvaluateMetrics()
	metrics.Finalize()

	if metrics.MeanReciprocalRank != 0 {
		t.Errorf("Expected MRR 0 for empty metrics, got %v", metrics.MeanReciprocalRank)
	}
	if metrics.MeanRank != 0 {
		t.Errorf("Expected MeanRank 0 for empty metrics, got %v", metrics.MeanRank)
	}
}

func TestComputeMRR(t *testing.T) {
	tests := []struct {
		name     string
		ranks    []int
		expected float64
	}{
		{
			name:     "all first rank",
			ranks:    []int{1, 1, 1},
			expected: 1.0,
		},
		{
			name:     "mixed ranks",
			ranks:    []int{1, 2, 3},
			expected: (1.0 + 0.5 + 0.3333) / 3.0,
		},
		{
			name:     "empty",
			ranks:    []int{},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ComputeMRR(tt.ranks)
			if result < tt.expected-0.001 || result > tt.expected+0.001 {
				t.Errorf("ComputeMRR() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestComputeHitsAtK(t *testing.T) {
	tests := []struct {
		name     string
		ranks    []int
		k        int
		expected float64
	}{
		{
			name:     "all within k",
			ranks:    []int{1, 2, 3},
			k:        5,
			expected: 1.0,
		},
		{
			name:     "some within k",
			ranks:    []int{1, 3, 5},
			k:        3,
			expected: 2.0 / 3.0,
		},
		{
			name:     "none within k",
			ranks:    []int{4, 5, 6},
			k:        3,
			expected: 0.0,
		},
		{
			name:     "empty",
			ranks:    []int{},
			k:        1,
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ComputeHitsAtK(tt.ranks, tt.k)
			if result < tt.expected-0.001 || result > tt.expected+0.001 {
				t.Errorf("ComputeHitsAtK() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestComputeMeanRank(t *testing.T) {
	tests := []struct {
		name     string
		ranks    []int
		expected float64
	}{
		{
			name:     "all first rank",
			ranks:    []int{1, 1, 1},
			expected: 1.0,
		},
		{
			name:     "mixed ranks",
			ranks:    []int{1, 2, 3},
			expected: 2.0,
		},
		{
			name:     "empty",
			ranks:    []int{},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ComputeMeanRank(tt.ranks)
			if result < tt.expected-0.001 || result > tt.expected+0.001 {
				t.Errorf("ComputeMeanRank() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestLinkPrediction_PredictTail(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "alice", Relation: "knows", Tail: "charlie"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 2,
	}

	model.Train(triples, opts)

	// Create link prediction instance
	lp := NewLinkPrediction(model)

	// Predict tail for (alice, knows)
	results, err := lp.PredictTail("alice", "knows", 5)
	if err != nil {
		t.Fatalf("PredictTail() error = %v", err)
	}

	// Should return empty results for now (placeholder implementation)
	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}

func TestLinkPrediction_PredictHead(t *testing.T) {
	store := NewEmbeddingStore(32)
	model := NewTransE(store)

	// Train on some triples
	triples := []*Triple{
		{Head: "alice", Relation: "knows", Tail: "bob"},
		{Head: "charlie", Relation: "knows", Tail: "bob"},
	}

	opts := TrainOptions{
		Dimension: 32,
		Epochs:    1,
		BatchSize: 2,
	}

	model.Train(triples, opts)

	// Create link prediction instance
	lp := NewLinkPrediction(model)

	// Predict head for (knows, bob)
	results, err := lp.PredictHead("knows", "bob", 5)
	if err != nil {
		t.Fatalf("PredictHead() error = %v", err)
	}

	// Should return empty results for now (placeholder implementation)
	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}
