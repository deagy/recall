package index

import (
	"math/rand"
	"testing"
)

func BenchmarkHNSW_Add(b *testing.B) {
	h := NewHNSW(128, DefaultHNSWConfig())
	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(rng.Float64())
		}
		h.Add(string(rune('a'+i%26)), vec)
	}
}

func BenchmarkHNSW_Search(b *testing.B) {
	h := NewHNSW(128, DefaultHNSWConfig())
	rng := rand.New(rand.NewSource(42))

	// Insert 1000 vectors
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(rng.Float64())
		}
		h.Add(string(rune('a'+i%26)), vec)
	}

	queryVec := make([]float32, 128)
	for j := range queryVec {
		queryVec[j] = float32(rng.Float64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(queryVec, 10)
	}
}

func BenchmarkHNSW_SearchLarge(b *testing.B) {
	h := NewHNSW(128, DefaultHNSWConfig())
	rng := rand.New(rand.NewSource(42))

	// Insert 10000 vectors
	for i := 0; i < 10000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(rng.Float64())
		}
		h.Add(string(rune('a'+i%26)), vec)
	}

	queryVec := make([]float32, 128)
	for j := range queryVec {
		queryVec[j] = float32(rng.Float64())
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(queryVec, 10)
	}
}