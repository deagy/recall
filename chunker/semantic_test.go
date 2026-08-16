package chunker

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
)

// mockEmbedder is a simple embedder for testing.
type mockEmbedder struct {
	dim int
}

func newMockEmbedder(dim int) *mockEmbedder {
	return &mockEmbedder{dim: dim}
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vec := make([]float32, m.dim)
	// Simple deterministic embedding based on text length
	for i := range vec {
		vec[i] = float32(len(text)+i) / float32(m.dim)
	}
	return vec, nil
}

func TestSemanticChunker_EmptyContent(t *testing.T) {
	embedder := newMockEmbedder(10)
	cfg := DefaultSemanticConfig()
	cfg.Threshold = 0.5
	chunker := NewSemantic(embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	chunks, err := chunker.Chunk(doc, "")
	if err != nil {
		t.Fatal(err)
	}
	if chunks != nil {
		t.Errorf("expected nil chunks for empty content, got %d", len(chunks))
	}
}

func TestSemanticChunker_SingleSentence(t *testing.T) {
	embedder := newMockEmbedder(10)
	cfg := DefaultSemanticConfig()
	cfg.Threshold = 0.5
	cfg.MinChunkSize = 10
	chunker := NewSemantic(embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	chunks, err := chunker.Chunk(doc, "This is a single sentence.")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestSemanticChunker_MultipleSentences(t *testing.T) {
	embedder := newMockEmbedder(10)
	cfg := DefaultSemanticConfig()
	cfg.Threshold = 0.5
	cfg.MinChunkSize = 10
	chunker := NewSemantic(embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	content := "This is the first sentence. This is the second sentence. This is the third sentence."
	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestSemanticChunker_SplitPoints(t *testing.T) {
	embedder := newMockEmbedder(10)
	cfg := DefaultSemanticConfig()
	cfg.Threshold = 0.5
	cfg.MinChunkSize = 10
	chunker := NewSemantic(embedder, cfg)

	// Test with content that should create multiple chunks
	doc := &core.Document{ID: "doc-1"}
	content := "Go is a programming language. Python is also popular. Rust is gaining traction. JavaScript is ubiquitous."
	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 1 {
		t.Error("expected at least one chunk")
	}
}

func TestSemanticChunker_Overlap(t *testing.T) {
	embedder := newMockEmbedder(10)
	cfg := DefaultSemanticConfig()
	cfg.Threshold = 0.5
	cfg.MinChunkSize = 10
	cfg.PreserveOverlap = true
	cfg.OverlapSize = 1
	chunker := NewSemantic(embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	content := "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestSplitSentences(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int
	}{
		{"empty", "", 0},
		{"single sentence", "Hello world.", 1},
		{"multiple sentences", "Hello world. How are you? I am fine!", 3},
		{"no punctuation", "Hello world", 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sentences := splitSentences(tt.input)
			if len(sentences) != tt.expected {
				t.Errorf("expected %d sentences, got %d: %v", tt.expected, len(sentences), sentences)
			}
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float64
	}{
		{"identical vectors", []float32{1, 0, 0}, []float32{1, 0, 0}, 1.0},
		{"orthogonal vectors", []float32{1, 0, 0}, []float32{0, 1, 0}, 0.0},
		{"opposite vectors", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1.0},
		{"different lengths", []float32{1, 0}, []float32{1, 0, 0}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sim := cosineSimilarity(tt.a, tt.b)
			if sim < tt.expected-0.01 || sim > tt.expected+0.01 {
				t.Errorf("expected similarity %.2f, got %.2f", tt.expected, sim)
			}
		})
	}
}

func TestChunkCoherence(t *testing.T) {
	// Test with identical embeddings (perfect coherence)
	embeddings := [][]float32{
		{1, 0, 0},
		{1, 0, 0},
		{1, 0, 0},
	}
	coherence := ChunkCoherence(embeddings)
	if coherence != 1.0 {
		t.Errorf("expected coherence 1.0, got %f", coherence)
	}

	// Test with single sentence
	coherence = ChunkCoherence([][]float32{{1, 0, 0}})
	if coherence != 1.0 {
		t.Errorf("expected coherence 1.0 for single sentence, got %f", coherence)
	}

	// Test with empty
	coherence = ChunkCoherence(nil)
	if coherence != 1.0 {
		t.Errorf("expected coherence 1.0 for empty, got %f", coherence)
	}
}

func TestNormalizeCoherence(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{1.0, 1.0},
		{-1.0, 0.0},
		{2.0, 1.0},  // Clamped
		{-2.0, 0.0}, // Clamped
	}

	for _, tt := range tests {
		result := NormalizeCoherence(tt.input)
		if result < tt.expected-0.01 || result > tt.expected+0.01 {
			t.Errorf("NormalizeCoherence(%f) = %f, want %f", tt.input, result, tt.expected)
		}
	}
}

func TestMean(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		expected float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 5},
		{"multiple", []float64{1, 2, 3, 4, 5}, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Mean(tt.data)
			if result < tt.expected-0.01 || result > tt.expected+0.01 {
				t.Errorf("Mean(%v) = %f, want %f", tt.data, result, tt.expected)
			}
		})
	}
}

func TestVariance(t *testing.T) {
	data := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	expected := 4.0
	result := Variance(data)
	if result < expected-0.01 || result > expected+0.01 {
		t.Errorf("Variance(%v) = %f, want %f", data, result, expected)
	}
}

func TestStdDev(t *testing.T) {
	data := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	expected := 2.0
	result := StdDev(data)
	if result < expected-0.01 || result > expected+0.01 {
		t.Errorf("StdDev(%v) = %f, want %f", data, result, expected)
	}
}

func TestChunkMetrics(t *testing.T) {
	metrics := NewChunkMetrics()

	// Record some chunks
	metrics.RecordChunk(100, 1000000, 0.8)
	metrics.RecordChunk(200, 2000000, 0.9)
	metrics.RecordChunk(150, 1500000, 0.85)

	summary := metrics.Summary()
	if summary.TotalChunks != 3 {
		t.Errorf("expected 3 chunks, got %d", summary.TotalChunks)
	}
	if summary.AvgChunkSize < 149 || summary.AvgChunkSize > 151 {
		t.Errorf("expected avg chunk size ~150, got %f", summary.AvgChunkSize)
	}
	if summary.MinChunkSize != 100 {
		t.Errorf("expected min chunk size 100, got %d", summary.MinChunkSize)
	}
	if summary.MaxChunkSize != 200 {
		t.Errorf("expected max chunk size 200, got %d", summary.MaxChunkSize)
	}
}

func TestChunkMetrics_Empty(t *testing.T) {
	metrics := NewChunkMetrics()
	summary := metrics.Summary()
	if summary.TotalChunks != 0 {
		t.Errorf("expected 0 chunks, got %d", summary.TotalChunks)
	}
}

func TestAnalyzeChunk(t *testing.T) {
	info := AnalyzeChunk("This is a test sentence with some content.", 0.8)
	if info.Size != len("This is a test sentence with some content.") {
		t.Errorf("expected size %d, got %d", len("This is a test sentence with some content."), info.Size)
	}
	if info.Coherence != 0.8 {
		t.Errorf("expected coherence 0.8, got %f", info.Coherence)
	}
	if !info.IsCoherent {
		t.Error("expected chunk to be coherent")
	}
}
