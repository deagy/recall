package llm

import (
	"context"
	"strings"
	"sync"
)

// MockBackend is a mock LLM backend for testing.
type MockBackend struct {
	// Response is the default response to return.
	Response string

	// ResponseFunc is an optional function to generate responses dynamically.
	ResponseFunc func(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// StreamResponse is the default streaming response.
	StreamResponse string

	// mu protects concurrent access.
	mu sync.Mutex

	// CallCount tracks the number of times Chat was called.
	CallCount int

	// LastRequest stores the last request made.
	LastRequest *ChatRequest
}

// NewMockBackend creates a new MockBackend with default settings.
func NewMockBackend() *MockBackend {
	return &MockBackend{
		Response:       "This is a mock response.",
		StreamResponse: "This is a mock streaming response.",
	}
}

// Chat sends a chat request and returns a mock response.
func (b *MockBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	b.mu.Lock()
	b.CallCount++
	b.LastRequest = req
	b.mu.Unlock()

	if b.ResponseFunc != nil {
		return b.ResponseFunc(ctx, req)
	}

	return &ChatResponse{
		Message: Message{
			Role:    "assistant",
			Content: b.Response,
		},
		Usage: Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
		FinishReason: "stop",
	}, nil
}

// ChatStream sends a chat request with streaming enabled.
func (b *MockBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	b.mu.Lock()
	b.CallCount++
	b.LastRequest = req
	b.mu.Unlock()

	// Split the response into chunks
	words := strings.Fields(b.StreamResponse)
	for i, word := range words {
		chunk := &StreamChunk{
			Delta: Message{
				Role:    "assistant",
				Content: word + " ",
			},
		}

		if err := fn(chunk); err != nil {
			return err
		}

		// Send final chunk with usage
		if i == len(words)-1 {
			finalChunk := &StreamChunk{
				Delta: Message{
					Role:    "assistant",
					Content: "",
				},
				Usage: &Usage{
					PromptTokens:     10,
					CompletionTokens: len(words),
					TotalTokens:      10 + len(words),
				},
				FinishReason: "stop",
			}
			if err := fn(finalChunk); err != nil {
				return err
			}
		}
	}

	return nil
}

// MockExtractor is a mock extractor for testing.
type MockExtractor struct {
	// Entities is the default entities to return.
	Entities []Entity

	// Relations is the default relations to return.
	Relations []Relation

	// ExtractFunc is an optional function to extract entities dynamically.
	ExtractFunc func(ctx context.Context, text string, sourceChunkID string) ([]Entity, error)

	// ExtractRelationsFunc is an optional function to extract relations dynamically.
	ExtractRelationsFunc func(ctx context.Context, text string, sourceChunkID string) ([]Relation, error)
}

// NewMockExtractor creates a new MockExtractor with default settings.
func NewMockExtractor() *MockExtractor {
	return &MockExtractor{
		Entities: []Entity{
			{ID: "mock-entity-1", Label: "Mock Entity", Type: "concept", Confidence: 0.9},
		},
		Relations: []Relation{
			{From: "mock-entity-1", To: "mock-entity-2", Type: "related_to", Confidence: 0.8},
		},
	}
}

// ExtractEntities extracts entities from text.
func (e *MockExtractor) ExtractEntities(ctx context.Context, text string, sourceChunkID string) ([]Entity, error) {
	if e.ExtractFunc != nil {
		return e.ExtractFunc(ctx, text, sourceChunkID)
	}
	return e.Entities, nil
}

// ExtractRelations extracts relations from text.
func (e *MockExtractor) ExtractRelations(ctx context.Context, text string, sourceChunkID string) ([]Relation, error) {
	if e.ExtractRelationsFunc != nil {
		return e.ExtractRelationsFunc(ctx, text, sourceChunkID)
	}
	return e.Relations, nil
}
