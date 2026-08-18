package query

import (
	"context"
	"errors"
	"testing"

	"github.com/deagy/recall/llm"
)

func mockBackend(resp string) *llm.MockBackend {
	b := llm.NewMockBackend()
	b.Response = resp
	return b
}

func TestRewrite(t *testing.T) {
	r := NewRewriter(mockBackend("kubernetes pod restart loop  reasons"))
	got, err := r.Rewrite(context.Background(), "why does my k8s pod keep restarting?")
	if err != nil {
		t.Fatal(err)
	}
	if got != "kubernetes pod restart loop  reasons" {
		t.Fatalf("rewrite = %q", got)
	}

	// Multi-line reply -> first line wins.
	r2 := NewRewriter(mockBackend("first line query\nsome commentary"))
	got2, err := r2.Rewrite(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if got2 != "first line query" {
		t.Fatalf("first-line trim = %q", got2)
	}

	// Empty LLM output -> fall back to original.
	r3 := NewRewriter(mockBackend("   \n  "))
	got3, err := r3.Rewrite(context.Background(), "original question")
	if err != nil {
		t.Fatal(err)
	}
	if got3 != "original question" {
		t.Fatalf("fallback = %q", got3)
	}

	// Nil backend errors.
	if _, err := NewRewriter(nil).Rewrite(context.Background(), "q"); err == nil {
		t.Fatal("nil backend should error")
	}

	// Custom system prompt is used.
	seen := ""
	b := llm.NewMockBackend()
	b.ResponseFunc = func(_ context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
		seen = req.Messages[0].Content
		return &llm.ChatResponse{Message: llm.Message{Role: "assistant", Content: "x"}}, nil
	}
	rc := NewRewriter(b)
	rc.SystemPrompt = "custom prompt"
	if _, err := rc.Rewrite(context.Background(), "q"); err != nil {
		t.Fatal(err)
	}
	if seen != "custom prompt" {
		t.Fatalf("custom prompt not used: %q", seen)
	}
}

func TestHyDE(t *testing.T) {
	h := NewHyDE(mockBackend("Hypothetical answer paragraph about embeddings."))
	got, err := h.Generate(context.Background(), "what are embeddings?")
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hypothetical answer paragraph about embeddings." {
		t.Fatalf("hyde = %q", got)
	}
	if _, err := NewHyDE(nil).Generate(context.Background(), "q"); err == nil {
		t.Fatal("nil backend should error")
	}

	// Custom prompt used.
	b := llm.NewMockBackend()
	b.ResponseFunc = func(_ context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
		if req.Messages[0].Content != "my hyde prompt" {
			t.Fatalf("prompt = %q", req.Messages[0].Content)
		}
		return &llm.ChatResponse{Message: llm.Message{Content: "h"}}, nil
	}
	h2 := NewHyDE(b)
	h2.SystemPrompt = "my hyde prompt"
	if _, err := h2.Generate(context.Background(), "q"); err != nil {
		t.Fatal(err)
	}
}

func TestStepBack(t *testing.T) {
	s := NewStepBack(mockBackend("What is a REST API?\nmore text"))
	got, err := s.Generate(context.Background(), "how do I handle 401s in my REST service?")
	if err != nil {
		t.Fatal(err)
	}
	if got != "What is a REST API?" {
		t.Fatalf("stepback = %q", got)
	}
	if _, err := NewStepBack(nil).Generate(context.Background(), "q"); err == nil {
		t.Fatal("nil backend should error")
	}
}

func TestLLMBackendError(t *testing.T) {
	// A backend that fails propagates.
	b := llm.NewMockBackend()
	b.ResponseFunc = func(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
		return nil, errors.New("boom")
	}
	if _, err := NewRewriter(b).Rewrite(context.Background(), "q"); err == nil {
		t.Fatal("backend error should propagate")
	}
}
