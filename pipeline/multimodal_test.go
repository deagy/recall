package pipeline

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/llm"
	"github.com/deagy/recall/store"
)

func TestMultiModalPipeline_Answer(t *testing.T) {
	s, err := store.NewMultiModalStore(embedder.NewMockMultiModal(32))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	_ = s.AddText(ctx, "t1", "solar system has eight planets")
	_ = s.AddText(ctx, "t2", "go channels synchronize goroutines")
	_ = s.AddImage(ctx, "i1", []byte("planet-img"), "image/png", "planets orbiting the sun")
	_ = s.AddImage(ctx, "i2", []byte("code-img"), "image/png", "")

	// No LLM: Answer is the rendered prompt, Context lists the sources.
	p := NewMultiModalPipeline(s, nil).WithTopK(3)
	resp, err := p.Answer(ctx, "tell me about planets in the solar system")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resp.Context, "[text] solar system has eight planets") {
		t.Fatalf("context missing text source: %q", resp.Context)
	}
	if !strings.Contains(resp.Context, "[image image/png]") {
		t.Fatalf("context missing image source: %q", resp.Context)
	}
	if len(resp.Sources) == 0 || len(resp.Sources) > 3 {
		t.Fatalf("sources = %d", len(resp.Sources))
	}
	if !strings.Contains(resp.Answer, "tell me about planets in the solar system") {
		t.Fatalf("rendered answer missing question: %q", resp.Answer)
	}

	// Empty question.
	if _, err := p.Answer(ctx, "   "); err == nil {
		t.Fatal("empty question should error")
	}
	// Nil context accepted (replaced with Background).
	if _, err := p.Answer(nil, "planets"); err != nil {
		t.Fatal(err)
	}

	// Uncaptioned image gets a placeholder.
	p2 := NewMultiModalPipeline(s, nil).WithTopK(1)
	resp2, _ := p2.Answer(ctx, "a diagram of goroutines")
	if !strings.Contains(resp2.Context, "uncaptioned image") {
		t.Fatalf("expected placeholder caption: %q", resp2.Context)
	}

	// Empty store -> no relevant content.
	empty, _ := store.NewMultiModalStore(embedder.NewMockMultiModal(32))
	p3 := NewMultiModalPipeline(empty, nil)
	resp3, err := p3.Answer(ctx, "anything")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resp3.Context, "no relevant content found") {
		t.Fatalf("empty store context: %q", resp3.Context)
	}

	// With LLM backend: answer comes from the backend.
	b := llm.NewMockBackend()
	b.Response = "The solar system has eight planets."
	p4 := NewMultiModalPipeline(s, b).WithTemplate(NewTemplate(
		"system says hi {{.Context}}",
		"User asks: {{.Question}} / Context: {{.Context}}",
	))
	resp4, err := p4.Answer(ctx, "how many planets?")
	if err != nil {
		t.Fatal(err)
	}
	if resp4.Answer != "The solar system has eight planets." {
		t.Fatalf("llm answer = %q", resp4.Answer)
	}

	// Backend error propagates.
	bFail := llm.NewMockBackend()
	bFail.ResponseFunc = func(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
		return nil, context.DeadlineExceeded
	}
	p5 := NewMultiModalPipeline(s, bFail)
	if _, err := p5.Answer(ctx, "how many planets?"); err == nil {
		t.Fatal("backend error should propagate")
	}

	// WithTopK(0) keeps previous value.
	p6 := NewMultiModalPipeline(s, nil).WithTopK(2).WithTopK(0)
	resp6, _ := p6.Answer(ctx, "planets")
	if len(resp6.Sources) > 2 {
		t.Fatalf("topK not honored: %d", len(resp6.Sources))
	}
	// WithTemplate(nil) keeps previous.
	p7 := NewMultiModalPipeline(s, nil)
	p7.WithTemplate(nil)
	if _, err := p7.Answer(ctx, "planets"); err != nil {
		t.Fatal(err)
	}
}
