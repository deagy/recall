package pipeline

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

// contentOf returns a string of exactly n runes (for token-budget math).
func contentOf(n int) string { return strings.Repeat("x", n) }

func TestSmartContextWindow_SelectsByPriority(t *testing.T) {
	w := NewSmartContextWindow(150) // tokens
	candidates := []ScoredChunk{
		{Chunk: core.Chunk{ID: "A", Content: contentOf(400)}, Score: 0.9}, // 100 tokens
		{Chunk: core.Chunk{ID: "B", Content: contentOf(200)}, Score: 0.8}, // 50 tokens
		{Chunk: core.Chunk{ID: "C", Content: contentOf(200)}, Score: 0.7}, // 50 tokens
	}
	got := w.Select(candidates)
	// A (100) + B (50) = 150 fits; C would exceed.
	if len(got) != 2 || got[0].ID != "A" || got[1].ID != "B" {
		t.Fatalf("expected [A B], got %v", ids(got))
	}
}

func TestSmartContextWindow_SkipsOversized(t *testing.T) {
	w := NewSmartContextWindow(100)
	candidates := []ScoredChunk{
		{Chunk: core.Chunk{ID: "big", Content: contentOf(800)}, Score: 0.9},   // 200 tokens, doesn't fit
		{Chunk: core.Chunk{ID: "small", Content: contentOf(400)}, Score: 0.5}, // 100 tokens, fits
	}
	got := w.Select(candidates)
	if len(got) != 1 || got[0].ID != "small" {
		t.Fatalf("expected [small], got %v", ids(got))
	}
}

func TestSmartContextWindow_DeterministicTieBreak(t *testing.T) {
	w := NewSmartContextWindow(100)
	candidates := []ScoredChunk{
		{Chunk: core.Chunk{ID: "y", Content: contentOf(400)}, Score: 0.5},
		{Chunk: core.Chunk{ID: "x", Content: contentOf(400)}, Score: 0.5},
	}
	got := w.Select(candidates)
	// Same score -> lower ID considered first, so "x" wins the single slot.
	if len(got) != 1 || got[0].ID != "x" {
		t.Fatalf("expected [x], got %v", ids(got))
	}
}

func TestExtractiveSummarizer_Shortens(t *testing.T) {
	text := "The Go programming language is fast. Go is a compiled language. " +
		"The weather outside is quite pleasant today. I enjoy a cup of coffee in the morning. " +
		"Go programs run with great performance. The garden has many colorful flowers."
	sum := ExtractiveSummarizer(2)
	out, err := sum(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	got := len(splitSentences(out))
	if got != 2 {
		t.Fatalf("expected 2 sentences, got %d (%q)", got, out)
	}
	if len(out) >= len(text) {
		t.Fatalf("expected a shorter summary, got same/longer length")
	}
}

func TestExtractiveSummarizer_NoOpWhenShort(t *testing.T) {
	text := "One short sentence. Another one."
	out, err := ExtractiveSummarizer(5)(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	if out != text {
		t.Fatalf("expected text unchanged, got %q", out)
	}
}

func TestContextCompressor_ShortensOversized(t *testing.T) {
	c := NewContextCompressor(nil) // default extractive
	c.SetMaxChunkTokens(10)
	long := "Go is fast. Go is concurrent. Go has garbage collection. " +
		"The sky is blue. Birds fly high. The river runs to the sea. " +
		"Mountains are tall. Trees grow green. The wind blows softly."
	chunks := []core.Chunk{
		{ID: "long", Content: long},         // > 10 tokens -> compressed
		{ID: "short", Content: "tiny text"}, // <= 10 tokens -> unchanged
	}
	out, err := c.Compress(context.Background(), chunks)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(out))
	}
	if out[0].ID != "long" || out[0].Content == long {
		t.Fatalf("expected long chunk to be summarized, got %q", out[0].Content)
	}
	if len(out[0].Content) >= len(long) {
		t.Fatalf("expected summarized content to be shorter")
	}
	if out[1].ID != "short" || out[1].Content != "tiny text" {
		t.Fatalf("expected short chunk unchanged, got %q", out[1].Content)
	}
}

// ids returns the chunk IDs for readable assertions.
func ids(chunks []core.Chunk) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, c.ID)
	}
	return out
}

func TestTrackCitations_NumberingAndOrder(t *testing.T) {
	results := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", DocumentRef: "doc1.txt", Content: "First point. More detail."}, Score: 0.9},
		{Chunk: &core.Chunk{ID: "c2", DocumentRef: "doc2.txt", Content: "Second point."}, Score: 0.7},
	}
	cits := TrackCitations(results)
	if len(cits) != 2 {
		t.Fatalf("expected 2 citations, got %d", len(cits))
	}
	if cits[0].Number != 1 || cits[0].ChunkID != "c1" || cits[0].DocumentRef != "doc1.txt" {
		t.Fatalf("unexpected first citation: %+v", cits[0])
	}
	if cits[1].Number != 2 || cits[1].ChunkID != "c2" {
		t.Fatalf("unexpected second citation: %+v", cits[1])
	}
	if cits[0].Snippet != "First point." {
		t.Fatalf("expected snippet 'First point.', got %q", cits[0].Snippet)
	}
}

func TestTrackCitations_SkipsNilChunks(t *testing.T) {
	results := []index.SearchResult{
		{Chunk: nil, Score: 0.9},
		{Chunk: &core.Chunk{ID: "c2", DocumentRef: "doc2.txt", Content: "Only one."}, Score: 0.7},
	}
	cits := TrackCitations(results)
	if len(cits) != 1 || cits[0].Number != 1 || cits[0].ChunkID != "c2" {
		t.Fatalf("expected single citation for c2, got %+v", cits)
	}
}

func TestRenderCitations(t *testing.T) {
	cits := TrackCitations([]index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", DocumentRef: "d.txt", Content: "Hello there."}, Score: 0.5},
	})
	out := RenderCitations(cits)
	if !strings.Contains(out, "[1]") || !strings.Contains(out, "c1") || !strings.Contains(out, "Hello there") {
		t.Fatalf("unexpected render: %q", out)
	}
	if RenderCitations(nil) != "" {
		t.Fatal("expected empty string for no citations")
	}
}

func TestHallucinationDetector_Supported(t *testing.T) {
	sources := []core.Chunk{{ID: "s1", Content: "The capital of France is Paris. Paris is located in Europe."}}
	d := NewHallucinationDetector(0.5)

	grounded := "The capital of France is Paris."
	checks := d.Check(grounded, sources)
	if len(checks) != 1 || !checks[0].Supported {
		t.Fatalf("expected grounded claim supported, got %+v", checks)
	}
	if rate := d.HallucinationRate(grounded, sources); rate != 0 {
		t.Fatalf("expected 0 hallucination rate, got %v", rate)
	}
}

func TestHallucinationDetector_Unsupported(t *testing.T) {
	sources := []core.Chunk{{ID: "s1", Content: "The capital of France is Paris."}}
	d := NewHallucinationDetector(0.5)

	fabricated := "Quantum flux capacitors enable time travel."
	checks := d.Check(fabricated, sources)
	if len(checks) != 1 || checks[0].Supported {
		t.Fatalf("expected fabricated claim unsupported, got %+v", checks)
	}
	if rate := d.HallucinationRate(fabricated, sources); rate != 1 {
		t.Fatalf("expected 1.0 hallucination rate, got %v", rate)
	}
}

func TestRAGPipeline_WithCitations(t *testing.T) {
	s := newMockStore(t)
	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5).WithCitations()
	resp, err := p.Query(context.Background(), "what is go")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Citations) == 0 {
		t.Fatal("expected citations to be populated")
	}
	if resp.Citations[0].Number != 1 {
		t.Fatalf("expected first citation numbered 1, got %d", resp.Citations[0].Number)
	}
	// Without WithCitations the field stays empty.
	p2 := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5)
	resp2, err := p2.Query(context.Background(), "what is go")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp2.Citations) != 0 {
		t.Fatalf("expected no citations by default, got %d", len(resp2.Citations))
	}
}

func TestRAGPipeline_WithSmartContext(t *testing.T) {
	s := newMockStore(t)
	// Tiny budget: only the highest-priority (first by ID tie-break) chunks fit.
	p := NewRAGPipeline(s, DefaultTemplate()).WithTopK(5).WithMaxTokens(20).WithSmartContext()
	resp, err := p.Query(context.Background(), "what is go")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Tokens > 20 {
		t.Fatalf("expected context within 20 tokens, got %d", resp.Tokens)
	}
	if len(resp.Sources) == 0 {
		t.Fatal("expected sources in response")
	}
}
