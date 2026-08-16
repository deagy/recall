package index

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
)

func makeBenchEmbed(dim int, x, y, z float32) []float32 {
	v := make([]float32, dim)
	if dim >= 1 {
		v[0] = x
	}
	if dim >= 2 {
		v[1] = y
	}
	if dim >= 3 {
		v[2] = z
	}
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

func BenchmarkMemoryIndex_Search_Empty(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	query := makeBenchEmbed(3, 1, 0, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, DefaultSearchOptions(10))
	}
}

func BenchmarkMemoryIndex_Search_SingleElement(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	_ = idx.Add(ctx, &core.Chunk{ID: "c1", Content: "test", Embedding: makeBenchEmbed(3, 1, 0, 0)})
	query := makeBenchEmbed(3, 1, 0, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, DefaultSearchOptions(1))
	}
}

func BenchmarkMemoryIndex_AddBatch(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	chunks := make([]*core.Chunk, b.N)
	for i := range chunks {
		chunks[i] = &core.Chunk{
			ID:        "c" + string(rune('a'+i%26)),
			Content:   "test content",
			Embedding: makeBenchEmbed(3, float32(i%10)*0.1, float32(i%5)*0.2, float32(i%3)*0.3),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.AddBatch(ctx, chunks[:1])
	}
}

func BenchmarkMemoryIndex_Search_WithTermFilter(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		_ = idx.Add(ctx, &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeBenchEmbed(3, float32(i%10)*0.1, float32(i%5)*0.2, float32(i%3)*0.3),
			Metadata:  map[string]core.Value{"source": core.String{Value: "test.txt"}},
		})
	}
	query := makeBenchEmbed(3, 1, 0, 0)
	filter := &TermFilter{Key: "source", Value: "test.txt"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, SearchOptions{TopK: 10, Filters: []Filter{filter}})
	}
}

func BenchmarkMemoryIndex_Search_WithRangeFilter(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		score := float64(i) * 0.1
		_ = idx.Add(ctx, &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeBenchEmbed(3, float32(i%10)*0.1, float32(i%5)*0.2, float32(i%3)*0.3),
			Metadata:  map[string]core.Value{"score": core.Number{Value: score}},
		})
	}
	query := makeBenchEmbed(3, 1, 0, 0)
	min := 5.0
	max := 50.0
	filter := &RangeFilter{Key: "score", Min: &min, Max: &max, MinIncl: true, MaxIncl: true}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, SearchOptions{TopK: 10, Filters: []Filter{filter}})
	}
}

func BenchmarkMemoryIndex_Search_WithMultipleFilters(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		_ = idx.Add(ctx, &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeBenchEmbed(3, float32(i%10)*0.1, float32(i%5)*0.2, float32(i%3)*0.3),
			Metadata: map[string]core.Value{
				"source": core.String{Value: "test.txt"},
				"tag":    core.String{Value: "go"},
			},
		})
	}
	query := makeBenchEmbed(3, 1, 0, 0)
	filters := []Filter{
		&TermFilter{Key: "source", Value: "test.txt"},
		&TermInFilter{Key: "tag", Values: []string{"go", "rust"}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, SearchOptions{TopK: 10, Filters: filters})
	}
}

func BenchmarkMemoryIndex_Add_Single(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	chunk := &core.Chunk{
		ID:        "c1",
		Content:   "test",
		Embedding: makeBenchEmbed(3, 1, 0, 0),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Add(ctx, chunk)
	}
}

func BenchmarkMemoryIndex_Search_100Docs(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		_ = idx.Add(ctx, &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeBenchEmbed(3, float32(i%10)*0.1, float32(i%5)*0.2, float32(i%3)*0.3),
		})
	}
	query := makeBenchEmbed(3, 1, 0, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, DefaultSearchOptions(10))
	}
}

func BenchmarkMemoryIndex_Search_500Docs(b *testing.B) {
	idx := NewMemoryIndex("test", 3)
	ctx := context.Background()
	for i := 0; i < 500; i++ {
		_ = idx.Add(ctx, &core.Chunk{
			ID:        string(rune('a' + i%26)),
			Content:   "test content",
			Embedding: makeBenchEmbed(3, float32(i%10)*0.1, float32(i%5)*0.2, float32(i%3)*0.3),
		})
	}
	query := makeBenchEmbed(3, 1, 0, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, DefaultSearchOptions(10))
	}
}
