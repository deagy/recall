package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests exercise the real HTTP backends (OpenAI + Ollama) against
// local httptest servers — no network access required.

// --- OpenAI ---

func TestOpenAIClient_DefaultBaseURL(t *testing.T) {
	c := NewOpenAIClient("sk-test", "")
	assert.Equal(t, "https://api.openai.com/v1", c.BaseURL)
	assert.Equal(t, "sk-test", c.APIKey)
	require.NotNil(t, c.HTTPClient)

	c2 := NewOpenAIClient("sk-test", "https://example.com/v1")
	assert.Equal(t, "https://example.com/v1", c2.BaseURL)
}

func TestOpenAIClient_Chat(t *testing.T) {
	var gotAuth, gotPath string
	var gotBody OpenAIChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		fmt.Fprint(w, `{"id":"cmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"hello back"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`)
	}))
	defer srv.Close()

	c := NewOpenAIClient("sk-test", srv.URL)
	resp, err := c.Chat(context.Background(), &ChatRequest{
		Model:       "gpt-4",
		Messages:    []Message{{Role: "user", Content: "hello"}},
		Temperature: 0.2,
		MaxTokens:   100,
		Stop:        []string{"\n"},
	})
	require.NoError(t, err)

	assert.Equal(t, "/chat/completions", gotPath)
	assert.Equal(t, "Bearer sk-test", gotAuth)
	assert.Equal(t, "gpt-4", gotBody.Model)
	require.Len(t, gotBody.Messages, 1)
	assert.Equal(t, "user", gotBody.Messages[0].Role)
	assert.Equal(t, 0.2, gotBody.Temperature)
	assert.Equal(t, 100, gotBody.MaxTokens)

	assert.Equal(t, "assistant", resp.Message.Role)
	assert.Equal(t, "hello back", resp.Message.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, Usage{5, 3, 8}, resp.Usage)
}

func TestOpenAIClient_Chat_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"boom"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := NewOpenAIClient("sk-bad", srv.URL)
	_, err := c.Chat(context.Background(), &ChatRequest{Model: "gpt-4", Messages: []Message{{Role: "user", Content: "hi"}}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "status 401")
	assert.Contains(t, err.Error(), "boom")
}

func TestOpenAIClient_Chat_DecodeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `not json`)
	}))
	defer srv.Close()

	c := NewOpenAIClient("sk-test", srv.URL)
	_, err := c.Chat(context.Background(), &ChatRequest{Model: "gpt-4", Messages: []Message{{Role: "user", Content: "hi"}}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to decode response")
}

func TestOpenAIClient_ChatStream(t *testing.T) {
	var gotAccept string
	chunks := []string{
		`{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"he"}}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{"content":"llo"}}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6}}`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		// The client reads concatenated JSON objects (not SSE framing).
		var req OpenAIChatRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.True(t, req.Stream, "streaming requests must set stream=true")
		fmt.Fprint(w, strings.Join(chunks, ""))
	}))
	defer srv.Close()

	c := NewOpenAIClient("sk-test", srv.URL)
	var got []StreamChunk
	err := c.ChatStream(context.Background(),
		&ChatRequest{Model: "gpt-4", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(chunk *StreamChunk) error {
			got = append(got, *chunk)
			return nil
		})
	require.NoError(t, err)
	assert.Equal(t, "text/event-stream", gotAccept)
	require.Len(t, got, 3)
	assert.Equal(t, "he", got[0].Delta.Content)
	assert.Equal(t, "llo", got[1].Delta.Content)
	assert.Equal(t, "stop", got[2].FinishReason)
	require.NotNil(t, got[2].Usage)
	assert.Equal(t, 6, got[2].Usage.TotalTokens)
}

// An OpenAI server may interleave usage-only chunks with empty choices;
// this must not panic and must not end the stream early.
func TestOpenAIClient_ChatStream_EmptyChoicesChunk(t *testing.T) {
	chunks := []string{
		`{"id":"1","choices":[{"index":0,"delta":{"content":"a"}}]}`,
		`{"id":"1","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, strings.Join(chunks, ""))
	}))
	defer srv.Close()

	c := NewOpenAIClient("sk-test", srv.URL)
	var got []StreamChunk
	err := c.ChatStream(context.Background(),
		&ChatRequest{Model: "gpt-4", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(chunk *StreamChunk) error {
			got = append(got, *chunk)
			return nil
		})
	require.NoError(t, err)
	require.Len(t, got, 3)
	assert.NotNil(t, got[1].Usage, "usage-only chunk must carry usage")
	assert.Empty(t, got[1].Delta.Content)
	assert.Equal(t, "stop", got[2].FinishReason)
}

func TestOpenAIClient_ChatStream_CallbackError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"id":"1","choices":[{"index":0,"delta":{"content":"a"}}]}`)
	}))
	defer srv.Close()

	c := NewOpenAIClient("sk-test", srv.URL)
	wantErr := fmt.Errorf("consumer gave up")
	err := c.ChatStream(context.Background(),
		&ChatRequest{Model: "gpt-4", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(chunk *StreamChunk) error { return wantErr })
	require.ErrorIs(t, err, wantErr)
}

// --- Ollama ---

func TestOllamaClient_DefaultBaseURL(t *testing.T) {
	c := NewOllamaClient("")
	assert.Equal(t, "http://localhost:11434", c.BaseURL)
	require.NotNil(t, c.HTTPClient)

	c2 := NewOllamaClient("http://gpu:11434")
	assert.Equal(t, "http://gpu:11434", c2.BaseURL)
}

func TestOllamaClient_Chat(t *testing.T) {
	var gotBody OllamaChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		fmt.Fprint(w, `{"model":"llama3","created_at":"now","message":{"role":"assistant","content":"ciao"},"done":true,"prompt_eval_count":7,"eval_count":4}`)
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL)
	resp, err := c.Chat(context.Background(), &ChatRequest{
		Model:       "llama3",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: 0.5,
	})
	require.NoError(t, err)
	assert.Equal(t, 0.5, gotBody.Options.Temperature)

	assert.Equal(t, "assistant", resp.Message.Role)
	assert.Equal(t, "ciao", resp.Message.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, Usage{7, 4, 11}, resp.Usage)
}

func TestOllamaClient_Chat_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL)
	_, err := c.Chat(context.Background(), &ChatRequest{Model: "nope", Messages: []Message{{Role: "user", Content: "hi"}}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "status 404")
	assert.Contains(t, err.Error(), "model not found")
}

func TestOllamaClient_ChatStream(t *testing.T) {
	chunks := []string{
		`{"model":"m","message":{"role":"assistant","content":"un"},"done":false}`,
		`{"model":"m","message":{"role":"assistant","content":"due"},"done":false}`,
		`{"model":"m","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":3,"eval_count":2}`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaChatRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.True(t, req.Stream)
		fmt.Fprint(w, strings.Join(chunks, ""))
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL)
	var got []StreamChunk
	err := c.ChatStream(context.Background(),
		&ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(chunk *StreamChunk) error {
			got = append(got, *chunk)
			return nil
		})
	require.NoError(t, err)
	require.Len(t, got, 3)
	assert.Equal(t, "un", got[0].Delta.Content)
	assert.Nil(t, got[0].Usage, "non-final chunks must not carry usage")
	assert.Equal(t, "stop", got[2].FinishReason)
	require.NotNil(t, got[2].Usage)
	assert.Equal(t, 5, got[2].Usage.TotalTokens)
}

func TestOllamaClient_ChatStream_CallbackError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"model":"m","message":{"role":"assistant","content":"x"},"done":false}`)
	}))
	defer srv.Close()

	c := NewOllamaClient(srv.URL)
	wantErr := fmt.Errorf("consumer gave up")
	err := c.ChatStream(context.Background(),
		&ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(chunk *StreamChunk) error { return wantErr })
	require.ErrorIs(t, err, wantErr)
}

// --- LLMExtractor error paths ---

type cannedBackend struct{ reply, errReply string }

func (b *cannedBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	if b.errReply != "" {
		return nil, fmt.Errorf("%s", b.errReply)
	}
	return &ChatResponse{Message: Message{Role: "assistant", Content: b.reply}}, nil
}

func (b *cannedBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	return nil // never called by these tests
}

func TestLLMExtractor_ExtractRelations_ParseError(t *testing.T) {
	e := NewLLMExtractor(&cannedBackend{reply: "this is not json"}, "model")
	_, err := e.ExtractRelations(context.Background(), "text", "chunk-1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse relations")
}

func TestLLMExtractor_ExtractEntities_ParseError(t *testing.T) {
	e := NewLLMExtractor(&cannedBackend{reply: "{nope"}, "model")
	_, err := e.ExtractEntities(context.Background(), "text", "chunk-1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse entities")
}

func TestLLMExtractor_ExtractRelations_BackendError(t *testing.T) {
	e := NewLLMExtractor(&cannedBackend{errReply: "upstream down"}, "model")
	_, err := e.ExtractRelations(context.Background(), "text", "chunk-1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "LLM extraction failed")
	assert.Contains(t, err.Error(), "upstream down")
}
