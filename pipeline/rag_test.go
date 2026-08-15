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

	chunk1 := core.Chunk{ID: "c1", Content: strings.Repeat("word ", 25)} // ~125 chars / ~31 tokens
	chunk2 := core.Chunk{ID: "c2", Content: strings.Repeat("word ", 25)} // ~125 chars / ~31 tokens

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
	chunks   []*core.Chunk
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

func TestTemplate_RenderSystem(t *testing.T) {
	tmpl := NewTemplate("System: {{.SystemVar}}", "User: {{.UserVar}}")

	vars := map[string]interface{}{
		"SystemVar": "You are helpful",
		"UserVar":   "Answer the question",
	}

	result := tmpl.RenderSystem(vars)
	if !strings.Contains(result, "You are helpful") {
		t.Fatal("expected system var substitution")
	}
	if strings.Contains(result, "{{.UserVar}}") {
		t.Fatal("user var should not be in system render")
	}
}

func TestTemplate_RenderUser(t *testing.T) {
	tmpl := NewTemplate("System: {{.SystemVar}}", "User: {{.UserVar}}")

	vars := map[string]interface{}{
		"SystemVar": "You are helpful",
		"UserVar":   "Answer the question",
	}

	result := tmpl.RenderUser(vars)
	if !strings.Contains(result, "Answer the question") {
		t.Fatal("expected user var substitution")
	}
	if strings.Contains(result, "{{.SystemVar}}") {
		t.Fatal("system var should not be in user render")
	}
}

func TestTemplate_Render_NoVariables(t *testing.T) {
	tmpl := NewTemplate("Simple system", "Simple user")

	vars := map[string]interface{}{}

	result := tmpl.Render(vars)
	if !strings.Contains(result, "Simple system") {
		t.Fatal("expected system text")
	}
	if !strings.Contains(result, "Simple user") {
		t.Fatal("expected user text")
	}
}

func TestTemplate_Render_EmptyVars(t *testing.T) {
	tmpl := NewTemplate("System: {{.Missing}}", "User: {{.Missing}}")

	vars := map[string]interface{}{}

	result := tmpl.Render(vars)
	// Missing variables should remain as placeholders
	if !strings.Contains(result, "{{.Missing}}") {
		t.Fatal("expected missing variable to remain as placeholder")
	}
}

func TestTemplate_RenderSpecialChars(t *testing.T) {
	tmpl := NewTemplate("System: {{.Val}}", "User: {{.Val}}")

	vars := map[string]interface{}{
		"Val": "Hello <world> & 'friends'",
	}

	result := tmpl.Render(vars)
	if !strings.Contains(result, "Hello <world> & 'friends'") {
		t.Fatal("expected special characters to be preserved")
	}
}

func TestRAGPipeline_NullTemplate(t *testing.T) {
	s := newMockStore(t)

	p := NewRAGPipeline(s, nil)
	if p.template == nil {
		t.Fatal("expected default template")
	}

	resp, err := p.Query(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Answer == "" {
		t.Fatal("expected non-empty answer")
	}
}

func TestRAGPipeline_QueryHybrid(t *testing.T) {
	s := newMockStore(t)

	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5)

	resp, err := p.QueryHybrid(context.Background(), "test question")
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
}

func TestRAGPipeline_WithTopK(t *testing.T) {
	s := newMockStore(t)
	p := NewRAGPipeline(s, DefaultTemplate())
	p = p.WithTopK(0) // Should not change
	if p.topK != 10 {
		t.Fatalf("expected topK=10, got %d", p.topK)
	}

	p = p.WithTopK(-1) // Should not change
	if p.topK != 10 {
		t.Fatalf("expected topK=10, got %d", p.topK)
	}

	p = p.WithTopK(15)
	if p.topK != 15 {
		t.Fatalf("expected topK=15, got %d", p.topK)
	}
}

func TestRAGPipeline_WithMinScore(t *testing.T) {
	s := newMockStore(t)
	p := NewRAGPipeline(s, DefaultTemplate())
	p = p.WithMinScore(-1) // Should not change
	if p.minScore != 0 {
		t.Fatalf("expected minScore=0, got %f", p.minScore)
	}

	p = p.WithMinScore(0.5)
	if p.minScore != 0.5 {
		t.Fatalf("expected minScore=0.5, got %f", p.minScore)
	}
}

func TestRAGPipeline_WithMaxTokens(t *testing.T) {
	s := newMockStore(t)
	p := NewRAGPipeline(s, DefaultTemplate())
	p = p.WithMaxTokens(0) // Should not change
	if p.maxContextTokens != 4096 {
		t.Fatalf("expected maxContextTokens=4096, got %d", p.maxContextTokens)
	}

	p = p.WithMaxTokens(-1) // Should not change
	if p.maxContextTokens != 4096 {
		t.Fatalf("expected maxContextTokens=4096, got %d", p.maxContextTokens)
	}

	p = p.WithMaxTokens(1024)
	if p.maxContextTokens != 1024 {
		t.Fatalf("expected maxContextTokens=1024, got %d", p.maxContextTokens)
	}
}

func TestContextWindow_AddChunk_ZeroMaxTokens(t *testing.T) {
	cw := NewContextWindow(0)
	if cw.MaxTokens != 4096 {
		t.Fatalf("expected MaxTokens=4096, got %d", cw.MaxTokens)
	}
}

func TestContextWindow_AddChunk_NegativeMaxTokens(t *testing.T) {
	cw := NewContextWindow(-100)
	if cw.MaxTokens != 4096 {
		t.Fatalf("expected MaxTokens=4096, got %d", cw.MaxTokens)
	}
}

func TestContextWindow_AddChunk_EmptyContent(t *testing.T) {
	cw := NewContextWindow(100)
	chunk := core.Chunk{ID: "c1", Content: ""}
	if !cw.AddChunk(chunk) {
		t.Fatal("expected empty content chunk to fit")
	}
	if cw.Tokens() != 0 {
		t.Fatalf("expected 0 tokens, got %d", cw.Tokens())
	}
}

func TestContextWindow_String_Empty(t *testing.T) {
	cw := NewContextWindow(100)
	s := cw.String()
	if s != "" {
		t.Errorf("expected empty string, got %q", s)
	}
}

func TestContextWindow_Len(t *testing.T) {
	cw := NewContextWindow(1000)
	if cw.Len() != 0 {
		t.Fatalf("expected 0, got %d", cw.Len())
	}

	cw.AddChunk(core.Chunk{ID: "c1", Content: "hello"})
	if cw.Len() != 1 {
		t.Fatalf("expected 1, got %d", cw.Len())
	}

	cw.AddChunk(core.Chunk{ID: "c2", Content: "world"})
	if cw.Len() != 2 {
		t.Fatalf("expected 2, got %d", cw.Len())
	}
}

func TestContextWindow_EstimateTokens(t *testing.T) {
	cw := NewContextWindow(100)

	if cw.estimateTokens("") != 0 {
		t.Error("expected 0 tokens for empty string")
	}

	// Rough estimate: chars / 4
	longStr := strings.Repeat("word ", 100)
	tokens := cw.estimateTokens(longStr)
	if tokens <= 0 {
		t.Error("expected positive tokens for non-empty string")
	}
}

func TestContextWindow_String_MultipleChunks(t *testing.T) {
	cw := NewContextWindow(10000)
	cw.AddChunk(core.Chunk{ID: "c1", Content: "First"})
	cw.AddChunk(core.Chunk{ID: "c2", Content: "Second"})
	cw.AddChunk(core.Chunk{ID: "c3", Content: "Third"})

	s := cw.String()
	if !strings.Contains(s, "[Chunk 1]") {
		t.Error("expected [Chunk 1]")
	}
	if !strings.Contains(s, "[Chunk 2]") {
		t.Error("expected [Chunk 2]")
	}
	if !strings.Contains(s, "[Chunk 3]") {
		t.Error("expected [Chunk 3]")
	}
	if !strings.Contains(s, "First") {
		t.Error("expected First")
	}
	if !strings.Contains(s, "Second") {
		t.Error("expected Second")
	}
	if !strings.Contains(s, "Third") {
		t.Error("expected Third")
	}
	if !strings.Contains(s, "\n\n---\n\n") {
		t.Error("expected separator between chunks")
	}
}

func TestRAGResponse_Struct(t *testing.T) {
	resp := &RAGResponse{
		Answer:  "test answer",
		Context: "test context",
		Tokens:  100,
	}

	if resp.Answer != "test answer" {
		t.Errorf("expected 'test answer', got %q", resp.Answer)
	}
	if resp.Context != "test context" {
		t.Errorf("expected 'test context', got %q", resp.Context)
	}
	if resp.Tokens != 100 {
		t.Errorf("expected 100, got %d", resp.Tokens)
	}
	if resp.Sources != nil {
		t.Error("expected nil sources")
	}
}
