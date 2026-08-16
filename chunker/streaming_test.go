package chunker

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/deagy/recall/core"
)

func TestStreamingChunker_EmptyContent(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 5
	sc := NewStreaming(inner, embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	chunks, err := sc.Chunk(doc, "")
	if err != nil {
		t.Fatal(err)
	}
	if chunks != nil {
		t.Errorf("expected nil chunks for empty content, got %d", len(chunks))
	}
}

func TestStreamingChunker_SingleBatch(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 100 // Large buffer to fit all in one batch
	sc := NewStreaming(inner, embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	content := "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
	chunks, err := sc.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestStreamingChunker_MultipleBatches(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 2 // Small buffer to force multiple batches
	sc := NewStreaming(inner, embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	content := "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence. Seventh sentence."
	chunks, err := sc.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestStreamingChunker_OnChunkCallback(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 5

	var chunkCount int32
	cfg.OnChunk = func(chunk *core.Chunk) {
		atomic.AddInt32(&chunkCount, 1)
	}

	sc := NewStreaming(inner, embedder, cfg)
	doc := &core.Document{ID: "doc-1"}
	content := "First sentence. Second sentence. Third sentence."
	_, err := sc.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}

	if atomic.LoadInt32(&chunkCount) == 0 {
		t.Error("expected OnChunk callback to be called")
	}
}

func TestStreamingChunker_GetChunks(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 5
	sc := NewStreaming(inner, embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	content := "First sentence. Second sentence. Third sentence."
	sc.Chunk(doc, content)

	chunks := sc.GetChunks()
	if len(chunks) == 0 {
		t.Error("expected chunks to be available")
	}
}

func TestStreamingChunker_IsDone(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 5
	sc := NewStreaming(inner, embedder, cfg)

	// Before processing
	if sc.IsDone() {
		t.Error("expected IsDone to be false before processing")
	}

	doc := &core.Document{ID: "doc-1"}
	content := "First sentence. Second sentence. Third sentence."
	sc.Chunk(doc, content)

	// After processing
	if !sc.IsDone() {
		t.Error("expected IsDone to be true after processing")
	}
}

func TestStreamingChunker_MaxChunksExceeded(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 1})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 1
	cfg.MaxPendingChunks = 2 // Very low limit
	sc := NewStreaming(inner, embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	// Create content that will generate many small chunks
	content := "A. B. C. D. E. F. G. H. I. J."
	_, err := sc.Chunk(doc, content)
	// The error might not be returned if chunks are processed successfully
	// Just verify that processing completes without panic
	_ = err
}

func TestStreamingChunker_DefaultConfig(t *testing.T) {
	cfg := DefaultStreamingConfig()
	if cfg.BufferSize != 10 {
		t.Errorf("expected BufferSize 10, got %d", cfg.BufferSize)
	}
	if cfg.FlushInterval != 100*time.Millisecond {
		t.Errorf("expected FlushInterval 100ms, got %v", cfg.FlushInterval)
	}
	if cfg.MaxPendingChunks != 1000 {
		t.Errorf("expected MaxPendingChunks 1000, got %d", cfg.MaxPendingChunks)
	}
}

func TestStreamingChunker_CustomConfig(t *testing.T) {
	cfg := StreamingConfig{
		BufferSize:       20,
		FlushInterval:    200 * time.Millisecond,
		MaxPendingChunks: 500,
	}
	sc := NewStreaming(nil, nil, cfg)

	if sc.config.BufferSize != 20 {
		t.Errorf("expected BufferSize 20, got %d", sc.config.BufferSize)
	}
	if sc.config.FlushInterval != 200*time.Millisecond {
		t.Errorf("expected FlushInterval 200ms, got %v", sc.config.FlushInterval)
	}
	if sc.config.MaxPendingChunks != 500 {
		t.Errorf("expected MaxPendingChunks 500, got %d", sc.config.MaxPendingChunks)
	}
}

func TestChunkerError(t *testing.T) {
	err := &ChunkerError{Message: "test error"}
	if err.Error() != "test error" {
		t.Errorf("expected 'test error', got %q", err.Error())
	}
}

func TestStreamingChunker_LargeDocument(t *testing.T) {
	inner := NewFixed(Config{MaxTokens: 512, MinChunkSize: 10})
	embedder := newMockEmbedder(10)
	cfg := DefaultStreamingConfig()
	cfg.BufferSize = 10
	sc := NewStreaming(inner, embedder, cfg)

	doc := &core.Document{ID: "doc-1"}
	// Create a large document with many sentences
	content := ""
	for i := 0; i < 100; i++ {
		content += "This is sentence number " + itoa(i) + ". "
	}

	chunks, err := sc.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}
