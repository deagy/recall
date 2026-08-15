package fuse

import (
	"testing"
)

func TestWeightedFusion_Basic(t *testing.T) {
	f := NewWeightedFusion(0.7)

	s1 := map[string]float64{"a": 10, "b": 20}
	s2 := map[string]float64{"a": 5, "b": 15, "c": 30}

	result := f.Fuse(s1, s2)

	// a: 0.7*10 + 0.3*5 = 8.5
	if result["a"] != 8.5 {
		t.Errorf("expected 8.5 for 'a', got %f", result["a"])
	}
	// b: 0.7*20 + 0.3*15 = 18.5
	if result["b"] != 18.5 {
		t.Errorf("expected 18.5 for 'b', got %f", result["b"])
	}
	// c: 0.7*0 + 0.3*30 = 9.0
	if result["c"] < 8.99 || result["c"] > 9.01 {
		t.Errorf("expected ~9.0 for 'c', got %f", result["c"])
	}
}

func TestWeightedFusion_PureVector(t *testing.T) {
	f := NewWeightedFusion(1.0)
	s1 := map[string]float64{"a": 10, "b": 20}
	result := f.Fuse(s1)
	if result["a"] != 10 || result["b"] != 20 {
		t.Error("expected pure vector scores")
	}
}

func TestWeightedFusion_PureBM25(t *testing.T) {
	f := NewWeightedFusion(0.0)
	s2 := map[string]float64{"a": 5, "b": 15}
	result := f.Fuse(s2)
	if result["a"] != 5 || result["b"] != 15 {
		t.Error("expected pure BM25 scores")
	}
}

func TestWeightedFusion_Empty(t *testing.T) {
	f := NewWeightedFusion(0.5)
	result := f.Fuse()
	if len(result) != 0 {
		t.Error("expected empty result")
	}
}

func TestRRFFusion_Basic(t *testing.T) {
	f := NewRRFFusion(60)

	// s1: a=10 (rank 1), b=5 (rank 2)
	// s2: b=20 (rank 1), c=15 (rank 2)
	s1 := map[string]float64{"a": 10, "b": 5}
	s2 := map[string]float64{"b": 20, "c": 15}

	result := f.Fuse(s1, s2)

	// a: 1/(60+1) = 1/61 ≈ 0.01639
	if result["a"] < 0.016 || result["a"] > 0.017 {
		t.Errorf("expected ~0.0164 for 'a', got %f", result["a"])
	}
	// b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.03279
	if result["b"] < 0.032 || result["b"] > 0.033 {
		t.Errorf("expected ~0.0328 for 'b', got %f", result["b"])
	}
	// c: 1/(60+2) = 1/62 ≈ 0.01613
	if result["c"] < 0.016 || result["c"] > 0.017 {
		t.Errorf("expected ~0.0161 for 'c', got %f", result["c"])
	}
}

func TestRRFFusion_SingleMap(t *testing.T) {
	f := NewRRFFusion(60)
	s1 := map[string]float64{"x": 100, "y": 50}
	result := f.Fuse(s1)
	if len(result) != 2 {
		t.Fatalf("expected 2 results, got %d", len(result))
	}
	// x should have higher RRF score (rank 1)
	if result["x"] <= result["y"] {
		t.Error("expected x to rank higher than y")
	}
}

func TestRRFFusion_Empty(t *testing.T) {
	f := NewRRFFusion(60)
	result := f.Fuse()
	if len(result) != 0 {
		t.Error("expected empty result")
	}
}

func TestRRFFusion_DefaultK(t *testing.T) {
	f := NewRRFFusion(0) // Should default to 60
	if f.K != 60 {
		t.Errorf("expected default K=60, got %d", f.K)
	}
}