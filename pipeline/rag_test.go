package pipeline

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

func TestContextWindow_AddAndLimit(t *testing.T) {
	cw := NewContextWindow(50)

	chunk1 := core.Chunk{ID: "c1", Content: strings.Repeat("word ", 25)}  // ~125 chars / ~31 tokens
	chunk2 := core.Chunk{ID: "c2", Content: strings.Repeat("word ", 25)}  // ~125 chars / ~31 tokens

	if !cw.AddChunk(chunk1) {
		t.Fatal("expected chunk1 to fit")
	}
	if cw.Len() != 1 {
		t.Fatalf("expected 1 chunk, got %d", cw.Len())
	}

	// chunk2 should not fit (31 + 31 > 50)
	if cw.AddChunk(chunk2) {
		t.Fatal("expected chunk2 to be rejected")
	}
	if cw.Len() != 1 {
		t.Fatalf("expected 1 chunk, got %d", cw.Len())
	}
}

func TestContextWindow_Empty(t *testing.T) {
	cw := NewContextWindow(100)
	if !cw.IsEmpty() {
		t.Fatal("expected empty context")
	}
	if cw.Tokens() != 0 {
		t.Fatalf("expected 0 tokens, got %d", cw.Tokens())
	}
}

func TestContextWindow_String(t *testing.T) {
	cw := NewContextWindow(1000)
	cw.AddChunk(core.Chunk{ID: "c1", Content: "hello"})
	cw.AddChunk(core.Chunk{ID: "c2", Content: "world"})

	s := cw.String()
	if !strings.Contains(s, "hello") || !strings.Contains(s, "world") {
		t.Fatal("expected both chunks in string")
	}
	if !strings.Contains(s, "[Chunk 1]") || !strings.Contains(s, "[Chunk 2]") {
		t.Fatal("expected chunk labels")
	}
}

func TestTemplate_Render(t *testing.T) {
	tmpl := NewTemplate(
		"System: {{.SystemVar}}",
		"User: {{.UserVar}}",
	)

	vars := map[string]interface{}{
		"SystemVar": "You are helpful",
		"UserVar":   "Answer the question",
	}

	result := tmpl.Render(vars)
	if !strings.Contains(result, "You are helpful") {
		t.Fatal("expected system var substitution")
	}
	if !strings.Contains(result, "Answer the question") {
		t.Fatal("expected user var substitution")
	}
}

func TestTemplate_DefaultTemplate(t *testing.T) {
	tmpl := DefaultTemplate()
	if tmpl.System == "" {
		t.Fatal("expected non-empty system prompt")
	}
	if tmpl.User == "" {
		t.Fatal("expected non-empty user prompt")
	}
	if !strings.Contains(tmpl.User, "{{.Context}}") {
		t.Fatal("expected Context variable in user prompt")
	}
	if !strings.Contains(tmpl.User, "{{.Question}}") {
		t.Fatal("expected Question variable in user prompt")
	}
}

func TestRAGPipeline_Query(t *testing.T) {
	// Create a mock store
	s := newMockStore(t)

	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5)

	resp, err := p.Query(context.Background(), "test question")
	if err != nil {
		t.Fatal(err)
	}

	if resp.Context == "" {
		t.Fatal("expected non-empty context")
	}
	if resp.Answer == "" {
		t.Fatal("expected non-empty answer")
	}
	if len(resp.Sources) == 0 {
		t.Fatal("expected sources")
	}
	if resp.Tokens == 0 {
		t.Fatal("expected non-zero tokens")
	}
}

func TestRAGPipeline_ContextLimit(t *testing.T) {
	s := newMockStore(t)

	p := NewRAGPipeline(s, DefaultTemplate()).WithMaxTokens(50)

	resp, err := p.Query(context.Background(), "test question")
	if err != nil {
		t.Fatal(err)
	}

	// Context should respect the token limit
	if resp.Tokens > 50 {
		t.Fatalf("expected tokens <= 50, got %d", resp.Tokens)
	}
}

func TestRAGPipeline_EmptyResults(t *testing.T) {
	// Store with no data
	s := newEmptyStore()

	p := NewRAGPipeline(s, DefaultTemplate())

	resp, err := p.Query(context.Background(), "test question")
	if err != nil {
		t.Fatal(err)
	}

	if resp.Context != "" {
		t.Fatal("expected empty context")
	}
	if len(resp.Sources) != 0 {
		t.Fatalf("expected 0 sources, got %d", len(resp.Sources))
	}
}

func TestRAGPipeline_Builders(t *testing.T) {
	s := newMockStore(t)
	tmpl := DefaultTemplate()

	p := NewRAGPipeline(s, tmpl)
	p = p.WithTopK(20)
	p = p.WithMinScore(0.5)
	p = p.WithMaxTokens(2048)

	if p.topK != 20 {
		t.Fatalf("expected topK=20, got %d", p.topK)
	}
	if p.minScore != 0.5 {
		t.Fatalf("expected minScore=0.5, got %f", p.minScore)
	}
	if p.maxContextTokens != 2048 {
		t.Fatalf("expected maxContextTokens=2048, got %d", p.maxContextTokens)
	}
}

// --- Mock Store for Testing ---

type mockStore struct {
	chunks []*core.Chunk
	embedder embedder.Embedder
}

func newMockStore(t *testing.T) *mockStore {
	t.Helper()
	e := embedder.NewMockEmbedder(4)
	s := &mockStore{
		embedder: e,
	}
	// Add some test chunks
	for i := 0; i < 5; i++ {
		content := "This is test chunk " + string(rune('A'+i)) + " about Go programming"
		emb, err := e.Embed(context.Background(), content)
		if err != nil {
			t.Fatal(err)
		}
		s.chunks = append(s.chunks, &core.Chunk{
			ID:        "chunk-" + string(rune('A'+i)),
			Content:   content,
			Embedding: emb,
		})
	}
	return s
}

func newEmptyStore() *mockStore {
	return &mockStore{
		chunks:   make([]*core.Chunk, 0),
		embedder: embedder.NewMockEmbedder(4),
	}
}

func (s *mockStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	return nil
}

func (s *mockStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	results := make([]index.SearchResult, 0)
	for _, c := range s.chunks {
		if len(results) >= opts.TopK {
			break
		}
		results = append(results, index.SearchResult{Chunk: c, Score: 0.9})
	}
	return results, nil
}

func (s *mockStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return s.Search(ctx, query, opts)
}

func (s *mockStore) GetChunk(id string) (*core.Chunk, bool) {
	for _, c := range s.chunks {
		if c.ID == id {
			return c, true
		}
	}
	return nil, false
}

func (s *mockStore) DeleteChunk(id string) error {
	return nil
}

func (s *mockStore) DeleteDocument(docID string) error {
	return nil
}

func (s *mockStore) Count() int {
	return len(s.chunks)
}

func (s *mockStore) Namespaces() []string {
	return []string{"default"}
}

func (s *mockStore) Close() error {
	return nil
}