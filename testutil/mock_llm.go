package testutil

import (
	"context"
	"strings"
	"sync"

	"github.com/deagy/recall/llm"
)

// MockLLM is a deterministic, scripted LLM backend for tests. Each Chat or
// ChatStream call returns the next scripted response in order; once the script
// is exhausted the last response repeats. With no script it returns a fixed
// default. It satisfies llm.Backend and records call count and last request.
type MockLLM struct {
	mu      sync.Mutex
	script  []string
	pos     int
	calls   int
	lastReq *llm.ChatRequest
}

// NewMockLLM creates a MockLLM that returns the given responses in order.
func NewMockLLM(responses ...string) *MockLLM {
	if len(responses) == 0 {
		responses = []string{"mock answer"}
	}
	return &MockLLM{script: responses}
}

// Chat returns the next scripted response.
func (m *MockLLM) Chat(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	m.lastReq = req
	text := m.script[m.pos]
	if m.pos < len(m.script)-1 {
		m.pos++
	}
	return &llm.ChatResponse{
		Message:      llm.Message{Role: "assistant", Content: text},
		Usage:        llm.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		FinishReason: "stop",
	}, nil
}

// ChatStream streams the next scripted response word by word, followed by a
// final chunk carrying usage and the finish reason.
func (m *MockLLM) ChatStream(ctx context.Context, req *llm.ChatRequest, fn func(chunk *llm.StreamChunk) error) error {
	resp, err := m.Chat(ctx, req)
	if err != nil {
		return err
	}
	words := strings.Fields(resp.Message.Content)
	for _, w := range words {
		if err := fn(&llm.StreamChunk{
			Delta: llm.Message{Role: "assistant", Content: w + " "},
		}); err != nil {
			return err
		}
	}
	final := &llm.StreamChunk{
		Delta:        llm.Message{Role: "assistant", Content: ""},
		Usage:        &llm.Usage{PromptTokens: 1, CompletionTokens: len(words), TotalTokens: 1 + len(words)},
		FinishReason: "stop",
	}
	return fn(final)
}

// Calls returns the number of Chat/ChatStream calls made.
func (m *MockLLM) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

// LastRequest returns the last request received (nil before any call).
func (m *MockLLM) LastRequest() *llm.ChatRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastReq
}

var _ llm.Backend = (*MockLLM)(nil)
