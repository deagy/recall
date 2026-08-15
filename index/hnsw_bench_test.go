package index

import (
	"testing"
)

func BenchmarkHNSW_Add(b *testing.B) {
	hnsw := NewHNSW(32, DefaultHNSWConfig())
	embedding := make([]float32, 32)
	for i := range embedding {
		embedding[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnsw.Add(string(rune(i)), embedding)
	}
}

func BenchmarkHNSW_Search(b *testing.B) {
	hnsw := NewHNSW(32, DefaultHNSWConfig())
	for i := 0; i < 1000; i++ {
		embedding := make([]float32, 32)
		for j := range embedding {
			embedding[j] = float32(i + j)
		}
		hnsw.Add(string(rune(i)), embedding)
	}

	query := make([]float32, 32)
	for i := range query {
		query[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnsw.Search(query, 10)
	}
}

func BenchmarkHNSW_SearchLarge(b *testing.B) {
	hnsw := NewHNSW(32, DefaultHNSWConfig())
	for i := 0; i < 10000; i++ {
		embedding := make([]float32, 32)
		for j := range embedding {
			embedding[j] = float32(i + j)
		}
		hnsw.Add(string(rune(i)), embedding)
	}

	query := make([]float32, 32)
	for i := range query {
		query[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnsw.Search(query, 10)
	}
}
