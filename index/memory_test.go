package index

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
)

func makeTestChunks(dim int) []*core.Chunk {
	return []*core.Chunk{
		{
			ID: "c1", Content: "Go programming language", DocumentRef: "doc1", ChunkIndex: 0,
			Embedding: makeEmbed(dim, 1, 0, 0),
			Metadata:  map[string]core.Value{"source": core.String{Value: "golang.org"}},
		},
		{
			ID: "c2", Content: "Python programming language", DocumentRef: "doc1", ChunkIndex: 1,
			Embedding: makeEmbed(dim, 0, 1, 0),
			Metadata:  map[string]core.Value{"source": core.String{Value: "python.org"}},
		},
		{
			ID: "c3", Content: "Rust systems language", DocumentRef: "doc2", ChunkIndex: 0,
			Embedding: makeEmbed(dim, 0, 0, 1),
			Metadata:  map[string]core.Value{"source": core.String{Value: "rust-lang.org"}},
		},
	}
}

func makeEmbed(dim int, x, y, z float32) []float32 {
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

func TestMemoryIndex_AddAndCount(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunks := makeTestChunks(3)
	for _, c := range chunks {
		if err := idx.Add(context.Background(), c); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}
	if idx.Count() != 3 {
		t.Errorf("expected count 3, got %d", idx.Count())
	}
}

func TestMemoryIndex_AddInvalidEmbedding(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: nil}
	err := idx.Add(context.Background(), chunk)
	if err != core.ErrInvalidEmbedding {
		t.Errorf("expected ErrInvalidEmbedding, got %v", err)
	}
}

func TestMemoryIndex_AddEmbeddingMismatch(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	chunk := &core.Chunk{ID: "c1", Content: "test", Embedding: []float32{1, 2}}
	err := idx.Add(context.Background(), chunk)
	if err != core.ErrEmbeddingMismatch {
		t.Errorf("expected ErrEmbeddingMismatch, got %v", err)
	}
}

func TestMemoryIndex_Search(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{TopK: 3})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0].Chunk.ID != "c1" {
		t.Errorf("expected first result c1, got %s", results[0].Chunk.ID)
	}
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted descending at index %d", i)
		}
	}
}

func TestMemoryIndex_SearchWithFilters(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	filter := &TermFilter{Key: "source", Value: "golang.org"}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK: 10, Filters: []Filter{filter},
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Chunk.ID != "c1" {
		t.Errorf("expected c1, got %s", results[0].Chunk.ID)
	}
}

func TestMemoryIndex_SearchTermInFilter(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	filter := &TermInFilter{Key: "source", Values: []string{"golang.org", "rust-lang.org"}}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK: 10, Filters: []Filter{filter},
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestMemoryIndex_Delete(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	if err := idx.Delete(context.Background(), "c2"); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if idx.Count() != 2 {
		t.Errorf("expected count 2, got %d", idx.Count())
	}
	results, _ := idx.Search(context.Background(), []float32{0, 1, 0}, SearchOptions{TopK: 10})
	for _, r := range results {
		if r.Chunk.ID == "c2" {
			t.Error("deleted chunk should not appear")
		}
	}
}

func TestMemoryIndex_AddBatch(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	if err := idx.AddBatch(context.Background(), makeTestChunks(3)); err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}
	if idx.Count() != 3 {
		t.Errorf("expected count 3, got %d", idx.Count())
	}
}

func TestMemoryIndex_GetChunk(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	c, ok := idx.GetChunk("c1")
	if !ok {
		t.Fatal("expected to find c1")
	}
	if c.Content != "Go programming language" {
		t.Errorf("unexpected content: %s", c.Content)
	}
	_, ok = idx.GetChunk("nonexistent")
	if ok {
		t.Error("expected not to find nonexistent")
	}
}

func TestMemoryIndex_NamespaceAndDimension(t *testing.T) {
	idx := NewMemoryIndex("myns", 1536)
	if idx.Namespace() != "myns" {
		t.Errorf("expected namespace 'myns', got %q", idx.Namespace())
	}
	if idx.Dimension() != 1536 {
		t.Errorf("expected dimension 1536, got %d", idx.Dimension())
	}
}

func TestMemoryIndex_MinScoreFilter(t *testing.T) {
	idx := NewMemoryIndex("test", 3)
	for _, c := range makeTestChunks(3) {
		_ = idx.Add(context.Background(), c)
	}
	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, SearchOptions{
		TopK: 10, MinScore: 0.9,
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result with MinScore 0.9, got %d", len(results))
	}
	if results[0].Chunk.ID != "c1" {
		t.Errorf("expected c1, got %s", results[0].Chunk.ID)
	}
}
