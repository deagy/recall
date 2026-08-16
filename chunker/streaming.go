package chunker

import (
	"strings"
	"sync"
	"time"

	"github.com/deagy/recall/core"
)

// StreamingChunker processes documents incrementally, allowing for
// low-latency processing of large documents by emitting chunks as
// they become available rather than waiting for the entire document.
type StreamingChunker struct {
	inner       Chunker
	embedder    Embedder
	config      StreamingConfig
	mu          sync.Mutex
	chunks      []*core.Chunk
	done        bool
	embeddingCh chan []float32
}

// StreamingConfig holds configuration for streaming chunking.
type StreamingConfig struct {
	// BufferSize is the number of sentences to buffer before processing.
	// Default: 10
	BufferSize int

	// FlushInterval is the maximum time to wait before flushing buffered content.
	// Default: 100ms
	FlushInterval time.Duration

	// MaxPendingChunks is the maximum number of chunks to keep in memory.
	// Default: 1000
	MaxPendingChunks int

	// OnChunk is called for each chunk as it's processed.
	// Default: nil
	OnChunk func(chunk *core.Chunk)
}

// DefaultStreamingConfig returns a StreamingConfig with sensible defaults.
func DefaultStreamingConfig() StreamingConfig {
	return StreamingConfig{
		BufferSize:       10,
		FlushInterval:    100 * time.Millisecond,
		MaxPendingChunks: 1000,
	}
}

// NewStreaming creates a new StreamingChunker.
func NewStreaming(inner Chunker, embedder Embedder, cfg StreamingConfig) *StreamingChunker {
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = 10
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 100 * time.Millisecond
	}
	if cfg.MaxPendingChunks <= 0 {
		cfg.MaxPendingChunks = 1000
	}

	sc := &StreamingChunker{
		inner:       inner,
		embedder:    embedder,
		config:      cfg,
		embeddingCh: make(chan []float32, cfg.BufferSize),
	}

	return sc
}

// Chunk processes a document in streaming fashion, emitting chunks as they
// become available. This method blocks until all chunks are processed.
func (s *StreamingChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}

	s.mu.Lock()
	s.chunks = nil
	s.done = false
	s.mu.Unlock()

	// Split into sentences for streaming
	sentences := splitSentences(content)
	if len(sentences) == 0 {
		return nil, nil
	}

	// Process sentences in batches
	var allChunks []*core.Chunk
	batchSize := s.config.BufferSize

	for i := 0; i < len(sentences); i += batchSize {
		end := i + batchSize
		if end > len(sentences) {
			end = len(sentences)
		}

		batch := sentences[i:end]
		batchChunks, err := s.processBatch(doc, batch)
		if err != nil {
			return allChunks, err
		}

		allChunks = append(allChunks, batchChunks...)

		// Call OnChunk callback for each chunk
		if s.config.OnChunk != nil {
			for _, chunk := range batchChunks {
				s.config.OnChunk(chunk)
			}
		}

		// Check if we've exceeded max pending chunks
		s.mu.Lock()
		if len(allChunks) > s.config.MaxPendingChunks {
			s.mu.Unlock()
			return allChunks, ErrMaxChunksExceeded
		}
		s.mu.Unlock()
	}

	s.mu.Lock()
	s.chunks = allChunks
	s.done = true
	s.mu.Unlock()

	return allChunks, nil
}

// processBatch processes a batch of sentences into chunks.
func (s *StreamingChunker) processBatch(doc *core.Document, sentences []string) ([]*core.Chunk, error) {
	// Create a temporary document for this batch
	batchDoc := &core.Document{
		ID:       doc.ID,
		Title:    doc.Title,
		Author:   doc.Author,
		Source:   doc.Source,
		Tags:     doc.Tags,
		Metadata: doc.Metadata,
	}

	// Join sentences and process with inner chunker
	content := strings.Join(sentences, ". ")
	return s.inner.Chunk(batchDoc, content)
}

// GetChunks returns all chunks that have been processed so far.
func (s *StreamingChunker) GetChunks() []*core.Chunk {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := make([]*core.Chunk, len(s.chunks))
	copy(result, s.chunks)
	return result
}

// IsDone returns true if all chunks have been processed.
func (s *StreamingChunker) IsDone() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.done
}

// ErrMaxChunksExceeded is returned when the maximum number of chunks is exceeded.
var ErrMaxChunksExceeded = &ChunkerError{
	Message: "maximum number of pending chunks exceeded",
}

// ChunkerError represents an error that occurred during chunking.
type ChunkerError struct {
	Message string
}

func (e *ChunkerError) Error() string {
	return e.Message
}
