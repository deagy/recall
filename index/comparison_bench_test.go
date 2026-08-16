package index

import (
	"context"
	"math/rand"
	"testing"

	"github.com/deagy/recall/core"
)

func makeRandomEmbed(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	// Normalize
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	if norm > 0 {
		norm = float32(1.0 / float64(norm))
		for i := range v {
			v[i] *= norm
		}
	}
	return v
}

func BenchmarkSearch_Comparison_100Docs(b *testing.B) {
	dim := 32
	query := makeRandomEmbed(dim)

	// Brute-force index
	bfIdx := NewMemoryIndex("bf", dim)
	for i := 0; i < 100; i++ {
		_ = bfIdx.Add(context.Background(), &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeRandomEmbed(dim),
		})
	}

	// HNSW index (manually built since 100 < HNSWThreshold)
	hnsw := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < 100; i++ {
		hnsw.Add(string(rune('a'+i%26)), makeRandomEmbed(dim))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bfIdx.Search(context.Background(), query, DefaultSearchOptions(10))
	}
}

func BenchmarkSearch_Comparison_HNSW_100Docs(b *testing.B) {
	dim := 32
	query := makeRandomEmbed(dim)

	hnsw := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < 100; i++ {
		hnsw.Add(string(rune('a'+i%26)), makeRandomEmbed(dim))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnsw.Search(query, 50)
	}
}

func BenchmarkSearch_Comparison_1000Docs(b *testing.B) {
	dim := 32
	query := makeRandomEmbed(dim)

	// Brute-force index (1000 == HNSWThreshold, still uses brute force)
	bfIdx := NewMemoryIndex("bf", dim)
	for i := 0; i < 1000; i++ {
		_ = bfIdx.Add(context.Background(), &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeRandomEmbed(dim),
		})
	}

	// HNSW index
	hnsw := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < 1000; i++ {
		hnsw.Add(string(rune('a'+i%26)), makeRandomEmbed(dim))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bfIdx.Search(context.Background(), query, DefaultSearchOptions(10))
	}
}

func BenchmarkSearch_Comparison_HNSW_1000Docs(b *testing.B) {
	dim := 32
	query := makeRandomEmbed(dim)

	hnsw := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < 1000; i++ {
		hnsw.Add(string(rune('a'+i%26)), makeRandomEmbed(dim))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnsw.Search(query, 50)
	}
}

func BenchmarkSearch_Comparison_10000Docs(b *testing.B) {
	dim := 32
	query := makeRandomEmbed(dim)

	// Brute-force index
	bfIdx := NewMemoryIndex("bf", dim)
	for i := 0; i < 10000; i++ {
		_ = bfIdx.Add(context.Background(), &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeRandomEmbed(dim),
		})
	}

	// HNSW index
	hnsw := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < 10000; i++ {
		hnsw.Add(string(rune('a'+i%26)), makeRandomEmbed(dim))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bfIdx.Search(context.Background(), query, DefaultSearchOptions(10))
	}
}

func BenchmarkSearch_Comparison_HNSW_10000Docs(b *testing.B) {
	dim := 32
	query := makeRandomEmbed(dim)

	hnsw := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < 10000; i++ {
		hnsw.Add(string(rune('a'+i%26)), makeRandomEmbed(dim))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hnsw.Search(query, 50)
	}
}
