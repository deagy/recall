package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

func TestMockBackend_Chat(t *testing.T) {
	backend := NewMockBackend()
	backend.Response = "Test response"

	resp, err := backend.Chat(context.Background(), &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Model: "test-model",
	})

	if err != nil {
		t.Fatal(err)
	}

	if resp.Message.Content != "Test response" {
		t.Errorf("expected 'Test response', got %q", resp.Message.Content)
	}

	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
	}

	if backend.CallCount != 1 {
		t.Errorf("expected 1 call, got %d", backend.CallCount)
	}
}

func TestMockBackend_Chat_WithFunc(t *testing.T) {
	backend := NewMockBackend()
	backend.ResponseFunc = func(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
		return &ChatResponse{
			Message: Message{Role: "assistant", Content: "Dynamic response"},
			Usage:   Usage{TotalTokens: 20},
		}, nil
	}

	resp, err := backend.Chat(context.Background(), &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err != nil {
		t.Fatal(err)
	}

	if resp.Message.Content != "Dynamic response" {
		t.Errorf("expected 'Dynamic response', got %q", resp.Message.Content)
	}
}

func TestMockBackend_ChatStream(t *testing.T) {
	backend := NewMockBackend()
	backend.StreamResponse = "Hello world test"

	var chunks []StreamChunk
	err := backend.ChatStream(context.Background(), &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}, func(chunk *StreamChunk) error {
		chunks = append(chunks, *chunk)
		return nil
	})

	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}

	// Check that we got the final chunk with usage
	finalChunk := chunks[len(chunks)-1]
	if finalChunk.Usage == nil {
		t.Error("expected final chunk to have usage")
	}

	if finalChunk.FinishReason != "stop" {
		t.Errorf("expected finish reason 'stop', got %q", finalChunk.FinishReason)
	}
}

func TestMockBackend_Concurrent(t *testing.T) {
	backend := NewMockBackend()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			backend.Chat(context.Background(), &ChatRequest{
				Messages: []Message{
					{Role: "user", Content: "Hello"},
				},
			})
		}()
	}
	wg.Wait()

	if backend.CallCount != 10 {
		t.Errorf("expected 10 calls, got %d", backend.CallCount)
	}
}

func TestMockExtractor_ExtractEntities(t *testing.T) {
	extractor := NewMockExtractor()

	entities, err := extractor.ExtractEntities(context.Background(), "Test text", "chunk-1")
	if err != nil {
		t.Fatal(err)
	}

	if len(entities) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(entities))
	}

	if entities[0].Label != "Mock Entity" {
		t.Errorf("expected 'Mock Entity', got %q", entities[0].Label)
	}
}

func TestMockExtractor_ExtractRelations(t *testing.T) {
	extractor := NewMockExtractor()

	relations, err := extractor.ExtractRelations(context.Background(), "Test text", "chunk-1")
	if err != nil {
		t.Fatal(err)
	}

	if len(relations) != 1 {
		t.Fatalf("expected 1 relation, got %d", len(relations))
	}

	if relations[0].Type != "related_to" {
		t.Errorf("expected 'related_to', got %q", relations[0].Type)
	}
}

func TestMockExtractor_WithFunc(t *testing.T) {
	extractor := NewMockExtractor()
	extractor.ExtractFunc = func(ctx context.Context, text string, sourceChunkID string) ([]Entity, error) {
		return []Entity{
			{ID: "custom-1", Label: "Custom Entity", Type: "person", Confidence: 0.95},
		}, nil
	}

	entities, err := extractor.ExtractEntities(context.Background(), "Test text", "chunk-1")
	if err != nil {
		t.Fatal(err)
	}

	if len(entities) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(entities))
	}

	if entities[0].Label != "Custom Entity" {
		t.Errorf("expected 'Custom Entity', got %q", entities[0].Label)
	}
}

func TestLLMExtractor_ExtractEntities(t *testing.T) {
	mockBackend := NewMockBackend()
	mockBackend.Response = `[{"id":"entity-1","label":"Go","type":"language","confidence":0.95}]`

	extractor := NewLLMExtractor(mockBackend, "gpt-4")

	entities, err := extractor.ExtractEntities(context.Background(), "Go is a programming language", "chunk-1")
	if err != nil {
		t.Fatal(err)
	}

	if len(entities) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(entities))
	}

	if entities[0].Label != "Go" {
		t.Errorf("expected 'Go', got %q", entities[0].Label)
	}

	if entities[0].Properties["source_chunk_id"] != "chunk-1" {
		t.Errorf("expected source_chunk_id 'chunk-1', got %q", entities[0].Properties["source_chunk_id"])
	}
}

func TestLLMExtractor_ExtractRelations(t *testing.T) {
	mockBackend := NewMockBackend()
	mockBackend.Response = `[{"from":"go","to":"python","type":"compare","confidence":0.8}]`

	extractor := NewLLMExtractor(mockBackend, "gpt-4")

	relations, err := extractor.ExtractRelations(context.Background(), "Go and Python are programming languages", "chunk-1")
	if err != nil {
		t.Fatal(err)
	}

	if len(relations) != 1 {
		t.Fatalf("expected 1 relation, got %d", len(relations))
	}

	if relations[0].Type != "compare" {
		t.Errorf("expected 'compare', got %q", relations[0].Type)
	}
}

func TestLLMExtractor_InvalidJSON(t *testing.T) {
	mockBackend := NewMockBackend()
	mockBackend.Response = "invalid json"

	extractor := NewLLMExtractor(mockBackend, "gpt-4")

	_, err := extractor.ExtractEntities(context.Background(), "Test", "chunk-1")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLLMExtractor_BackendError(t *testing.T) {
	mockBackend := NewMockBackend()
	mockBackend.ResponseFunc = func(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
		return nil, errors.New("backend error")
	}

	extractor := NewLLMExtractor(mockBackend, "gpt-4")

	_, err := extractor.ExtractEntities(context.Background(), "Test", "chunk-1")
	if err == nil {
		t.Error("expected error from backend")
	}

	if !strings.Contains(err.Error(), "backend error") {
		t.Errorf("expected 'backend error' in error message, got %q", err.Error())
	}
}

func TestOpenAIClient_ConvertRequest(t *testing.T) {
	client := NewOpenAIClient("test-key", "")

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Model:       "gpt-4",
		Temperature: 0.7,
		MaxTokens:   100,
		Stop:        []string{"\n"},
	}

	openaiReq := client.convertRequest(req)

	if openaiReq.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %q", openaiReq.Model)
	}

	if len(openaiReq.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(openaiReq.Messages))
	}

	if openaiReq.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", openaiReq.Temperature)
	}

	if openaiReq.MaxTokens != 100 {
		t.Errorf("expected max tokens 100, got %d", openaiReq.MaxTokens)
	}
}

func TestOpenAIClient_ConvertResponse(t *testing.T) {
	client := NewOpenAIClient("test-key", "")

	openaiResp := &OpenAIChatResponse{
		Choices: []OpenAIChoice{
			{
				Message: OpenAIMessage{
					Role:    "assistant",
					Content: "Hello!",
				},
				FinishReason: "stop",
			},
		},
		Usage: OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	resp := client.convertResponse(openaiResp)

	if resp.Message.Content != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Message.Content)
	}

	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
	}

	if resp.FinishReason != "stop" {
		t.Errorf("expected finish reason 'stop', got %q", resp.FinishReason)
	}
}

func TestOllamaClient_ConvertRequest(t *testing.T) {
	client := NewOllamaClient("")

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Model:       "llama2",
		Temperature: 0.8,
	}

	ollamaReq := client.convertRequest(req)

	if ollamaReq.Model != "llama2" {
		t.Errorf("expected model 'llama2', got %q", ollamaReq.Model)
	}

	if ollamaReq.Options.Temperature != 0.8 {
		t.Errorf("expected temperature 0.8, got %f", ollamaReq.Options.Temperature)
	}
}

func TestOllamaClient_ConvertResponse(t *testing.T) {
	client := NewOllamaClient("")

	ollamaResp := &OllamaChatResponse{
		Message: OllamaMessage{
			Role:    "assistant",
			Content: "Hello!",
		},
		PromptEvalCount: 10,
		EvalCount:       5,
	}

	resp := client.convertResponse(ollamaResp)

	if resp.Message.Content != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Message.Content)
	}

	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatRequest_Defaults(t *testing.T) {
	req := &ChatRequest{}

	if req.Model != "" {
		t.Errorf("expected empty model, got %q", req.Model)
	}

	if req.Temperature != 0 {
		t.Errorf("expected 0 temperature, got %f", req.Temperature)
	}

	if req.MaxTokens != 0 {
		t.Errorf("expected 0 max tokens, got %d", req.MaxTokens)
	}
}

func TestStreamChunk_Defaults(t *testing.T) {
	chunk := &StreamChunk{}

	if chunk.Delta.Role != "" {
		t.Errorf("expected empty role, got %q", chunk.Delta.Role)
	}

	if chunk.Usage != nil {
		t.Error("expected nil usage")
	}

	if chunk.FinishReason != "" {
		t.Errorf("expected empty finish reason, got %q", chunk.FinishReason)
	}
}

func TestEntity_Defaults(t *testing.T) {
	entity := &Entity{}

	if entity.ID != "" {
		t.Errorf("expected empty ID, got %q", entity.ID)
	}

	if entity.Properties != nil {
		t.Error("expected nil properties")
	}
}

func TestRelation_Defaults(t *testing.T) {
	relation := &Relation{}

	if relation.From != "" {
		t.Errorf("expected empty from, got %q", relation.From)
	}

	if relation.Properties != nil {
		t.Error("expected nil properties")
	}
}

func TestResponseFormat_JSON(t *testing.T) {
	format := &ResponseFormat{
		Type: "json_object",
	}

	if format.Type != "json_object" {
		t.Errorf("expected 'json_object', got %q", format.Type)
	}
}

func TestUsage_Defaults(t *testing.T) {
	usage := &Usage{}

	if usage.TotalTokens != 0 {
		t.Errorf("expected 0 total tokens, got %d", usage.TotalTokens)
	}
}

func TestMessage_Defaults(t *testing.T) {
	msg := &Message{}

	if msg.Role != "" {
		t.Errorf("expected empty role, got %q", msg.Role)
	}

	if msg.Content != "" {
		t.Errorf("expected empty content, got %q", msg.Content)
	}
}

func TestJSONSerialization(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Model:       "gpt-4",
		Temperature: 0.7,
		MaxTokens:   100,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	var decoded ChatRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %q", decoded.Model)
	}

	if decoded.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", decoded.Temperature)
	}
}
