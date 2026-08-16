package fuse

import (
	"testing"
)

// --- WeightedFusion benchmarks ---

func BenchmarkWeightedFusion_Fuse_TwoMaps_Equal(b *testing.B) {
	f := NewWeightedFusion(0.5)
	s1 := map[string]float64{"a": 10, "b": 20, "c": 30}
	s2 := map[string]float64{"a": 5, "b": 15, "c": 25}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1, s2)
	}
}

func BenchmarkWeightedFusion_Fuse_TwoMaps_VectorDominant(b *testing.B) {
	f := NewWeightedFusion(0.9)
	s1 := map[string]float64{"a": 10, "b": 20, "c": 30}
	s2 := map[string]float64{"a": 5, "b": 15, "c": 25}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1, s2)
	}
}

func BenchmarkWeightedFusion_Fuse_TwoMaps_BM25Dominant(b *testing.B) {
	f := NewWeightedFusion(0.1)
	s1 := map[string]float64{"a": 10, "b": 20, "c": 30}
	s2 := map[string]float64{"a": 5, "b": 15, "c": 25}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1, s2)
	}
}

func BenchmarkWeightedFusion_Fuse_ManyMaps(b *testing.B) {
	f := NewWeightedFusion(0.5)
	scores := []map[string]float64{
		{"a": 10, "b": 20, "c": 30, "d": 40, "e": 50},
		{"a": 5, "b": 15, "c": 25, "d": 35, "e": 45},
		{"a": 8, "b": 18, "c": 28, "d": 38, "e": 48},
		{"a": 3, "b": 13, "c": 23, "d": 33, "e": 43},
		{"a": 7, "b": 17, "c": 27, "d": 37, "e": 47},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(scores...)
	}
}

func BenchmarkWeightedFusion_Fuse_Empty(b *testing.B) {
	f := NewWeightedFusion(0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse()
	}
}

func BenchmarkWeightedFusion_Fuse_SingleMap(b *testing.B) {
	f := NewWeightedFusion(0.5)
	s1 := map[string]float64{"a": 10, "b": 20, "c": 30}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1)
	}
}

// --- RRFFusion benchmarks ---

func BenchmarkRRFFusion_Fuse_TwoMaps_DefaultK(b *testing.B) {
	f := NewRRFFusion(60)
	s1 := map[string]float64{"a": 10, "b": 5, "c": 3}
	s2 := map[string]float64{"b": 20, "c": 15, "d": 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1, s2)
	}
}

func BenchmarkRRFFusion_Fuse_TwoMaps_SmallK(b *testing.B) {
	f := NewRRFFusion(10)
	s1 := map[string]float64{"a": 10, "b": 5, "c": 3}
	s2 := map[string]float64{"b": 20, "c": 15, "d": 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1, s2)
	}
}

func BenchmarkRRFFusion_Fuse_TwoMaps_LargeK(b *testing.B) {
	f := NewRRFFusion(100)
	s1 := map[string]float64{"a": 10, "b": 5, "c": 3}
	s2 := map[string]float64{"b": 20, "c": 15, "d": 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1, s2)
	}
}

func BenchmarkRRFFusion_Fuse_ManyMaps(b *testing.B) {
	f := NewRRFFusion(60)
	scores := []map[string]float64{
		{"a": 100, "b": 90, "c": 80, "d": 70, "e": 60},
		{"b": 100, "c": 90, "d": 80, "e": 70, "f": 60},
		{"c": 100, "d": 90, "e": 80, "f": 70, "g": 60},
		{"d": 100, "e": 90, "f": 80, "g": 70, "h": 60},
		{"e": 100, "f": 90, "g": 80, "h": 70, "a": 60},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(scores...)
	}
}

func BenchmarkRRFFusion_Fuse_Empty(b *testing.B) {
	f := NewRRFFusion(60)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse()
	}
}

func BenchmarkRRFFusion_Fuse_SingleMap(b *testing.B) {
	f := NewRRFFusion(60)
	s1 := map[string]float64{"a": 100, "b": 50, "c": 25}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Fuse(s1)
	}
}
