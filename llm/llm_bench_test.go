package llm

import (
	"context"
	"testing"
)

func BenchmarkMockBackend_Chat(b *testing.B) {
	backend := NewMockBackend()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backend.Chat(ctx, &ChatRequest{
			Messages: []Message{
				{Role: "user", Content: "Hello"},
			},
			Model: "test-model",
		})
	}
}

func BenchmarkMockBackend_ChatStream(b *testing.B) {
	backend := NewMockBackend()
	backend.StreamResponse = "This is a test response for benchmarking"
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var chunks int
		backend.ChatStream(ctx, &ChatRequest{
			Messages: []Message{
				{Role: "user", Content: "Hello"},
			},
		}, func(chunk *StreamChunk) error {
			chunks++
			return nil
		})
	}
}

func BenchmarkMockExtractor_ExtractEntities(b *testing.B) {
	extractor := NewMockExtractor()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractEntities(ctx, "Test text", "chunk-1")
	}
}

func BenchmarkMockExtractor_ExtractRelations(b *testing.B) {
	extractor := NewMockExtractor()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractRelations(ctx, "Test text", "chunk-1")
	}
}

func BenchmarkLLMExtractor_ExtractEntities(b *testing.B) {
	mockBackend := NewMockBackend()
	mockBackend.Response = `[{"id":"entity-1","label":"Go","type":"language","confidence":0.95}]`
	extractor := NewLLMExtractor(mockBackend, "gpt-4")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractEntities(ctx, "Go is a programming language", "chunk-1")
	}
}

func BenchmarkLLMExtractor_ExtractRelations(b *testing.B) {
	mockBackend := NewMockBackend()
	mockBackend.Response = `[{"from":"go","to":"python","type":"compare","confidence":0.8}]`
	extractor := NewLLMExtractor(mockBackend, "gpt-4")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor.ExtractRelations(ctx, "Go and Python are programming languages", "chunk-1")
	}
}

func BenchmarkOpenAIClient_ConvertRequest(b *testing.B) {
	client := NewOpenAIClient("test-key", "")
	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Model:       "gpt-4",
		Temperature: 0.7,
		MaxTokens:   100,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.convertRequest(req)
	}
}

func BenchmarkOllamaClient_ConvertRequest(b *testing.B) {
	client := NewOllamaClient("")
	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Model:       "llama2",
		Temperature: 0.8,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.convertRequest(req)
	}
}
