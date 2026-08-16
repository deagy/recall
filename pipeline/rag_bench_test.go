package pipeline

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// benchMockStore is a simple mock store for benchmarks.
type benchMockStore struct {
	chunks   []*core.Chunk
	embedder embedder.Embedder
}

func newBenchMockStore() *benchMockStore {
	e := embedder.NewMockEmbedder(4)
	s := &benchMockStore{embedder: e}
	for i := 0; i < 5; i++ {
		content := "This is test chunk " + string(rune('A'+i)) + " about Go programming"
		emb, _ := e.Embed(context.Background(), content)
		s.chunks = append(s.chunks, &core.Chunk{
			ID:        "chunk-" + string(rune('A'+i)),
			Content:   content,
			Embedding: emb,
		})
	}
	return s
}

func (s *benchMockStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	return nil
}

func (s *benchMockStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	results := make([]index.SearchResult, 0)
	for _, c := range s.chunks {
		if len(results) >= opts.TopK {
			break
		}
		results = append(results, index.SearchResult{Chunk: c, Score: 0.9})
	}
	return results, nil
}

func (s *benchMockStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return s.Search(ctx, query, opts)
}

func (s *benchMockStore) GetChunk(id string) (*core.Chunk, bool) {
	for _, c := range s.chunks {
		if c.ID == id {
			return c, true
		}
	}
	return nil, false
}

func (s *benchMockStore) DeleteChunk(id string) error {
	return nil
}

func (s *benchMockStore) DeleteDocument(docID string) error {
	return nil
}

func (s *benchMockStore) Count() int {
	return len(s.chunks)
}

func (s *benchMockStore) Namespaces() []string {
	return []string{"default"}
}

func (s *benchMockStore) Close() error {
	return nil
}

// --- ContextWindow benchmarks ---

func BenchmarkContextWindow_AddChunk(b *testing.B) {
	cw := NewContextWindow(4096)
	emb, _ := embedder.NewMockEmbedder(4).Embed(context.Background(), "test")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cw.AddChunk(core.Chunk{
			ID:        "chunk",
			Content:   strings.Repeat("word ", 30),
			Embedding: emb,
		})
	}
}

func BenchmarkContextWindow_String(b *testing.B) {
	cw := NewContextWindow(4096)
	cw.AddChunk(core.Chunk{ID: "c1", Content: "First chunk of context content here."})
	cw.AddChunk(core.Chunk{ID: "c2", Content: "Second chunk of context content here."})
	cw.AddChunk(core.Chunk{ID: "c3", Content: "Third chunk of context content here."})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cw.String()
	}
}

func BenchmarkContextWindow_String_Large(b *testing.B) {
	cw := NewContextWindow(4096)
	for i := 0; i < 20; i++ {
		cw.AddChunk(core.Chunk{ID: "c" + string(rune('0'+i)), Content: strings.Repeat("word ", 20)})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cw.String()
	}
}

// --- Template benchmarks ---

func BenchmarkTemplate_Render(b *testing.B) {
	tmpl := DefaultTemplate()
	vars := map[string]interface{}{
		"Context":  "This is relevant context for answering the question.\n\n---\n\nMore context here.",
		"Question": "What is Go programming language?",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tmpl.Render(vars)
	}
}

func BenchmarkTemplate_Render_EmptyContext(b *testing.B) {
	tmpl := DefaultTemplate()
	vars := map[string]interface{}{
		"Context":  "",
		"Question": "What is Go?",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tmpl.Render(vars)
	}
}

func BenchmarkTemplate_Render_LargeContext(b *testing.B) {
	tmpl := DefaultTemplate()
	vars := map[string]interface{}{
		"Context":  strings.Repeat("This is a relevant context passage with useful information. ", 100),
		"Question": "What is Go programming language?",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tmpl.Render(vars)
	}
}

// --- RAGPipeline benchmarks ---

func BenchmarkRAGPipeline_Query(b *testing.B) {
	s := newBenchMockStore()
	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Query(context.Background(), "test question about Go programming")
	}
}

func BenchmarkRAGPipeline_Query_EmptyStore(b *testing.B) {
	s := &benchMockStore{chunks: make([]*core.Chunk, 0), embedder: embedder.NewMockEmbedder(4)}
	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Query(context.Background(), "test question")
	}
}

func BenchmarkRAGPipeline_QueryHybrid(b *testing.B) {
	s := newBenchMockStore()
	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.QueryHybrid(context.Background(), "test question about Go programming")
	}
}

func BenchmarkRAGPipeline_WithMaxTokens(b *testing.B) {
	s := newBenchMockStore()
	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5).WithMaxTokens(100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Query(context.Background(), "test question about Go programming")
	}
}
