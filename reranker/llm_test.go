package reranker

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/deagy/recall/index"
	"github.com/deagy/recall/llm"
)

// judgeBackend returns a MockBackend whose reply is the number encoded by a
// marker word present in the user message (the passage).
func judgeBackend(t *testing.T) *llm.MockBackend {
	markers := map[string]int{"aaa": 9, "bbb": 2, "ccc": 6}
	return &llm.MockBackend{
		ResponseFunc: func(_ context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
			user := req.Messages[len(req.Messages)-1].Content
			for w, n := range markers {
				if strings.Contains(user, w) {
					return &llm.ChatResponse{Message: llm.Message{Role: "assistant", Content: "Score: " + strconv.Itoa(n) + "."}}, nil
				}
			}
			return &llm.ChatResponse{Message: llm.Message{Role: "assistant", Content: "0"}}, nil
		},
	}
}

func TestLLMReranker_RanksByJudgeScore(t *testing.T) {
	rr, err := NewLLMReranker(LLMRerankerConfig{Backend: judgeBackend(t)})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	results := []index.SearchResult{
		mkResult("low", "passage with aaa token", 0.3),
		mkResult("high", "passage with bbb token", 0.95), // high coarse, low judge
		mkResult("mid", "passage with ccc token", 0.6),
	}
	out, err := rr.Rerank(context.Background(), "what is it", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	// Judge scores: aaa=9, bbb=2, ccc=6 -> order low, mid, high.
	want := []string{"low", "mid", "high"}
	for i, id := range want {
		if out[i].Chunk.ID != id {
			t.Fatalf("position %d = %s, want %s", i, out[i].Chunk.ID, id)
		}
	}
	if out[0].RerankScore != 0.9 || out[1].RerankScore != 0.6 || out[2].RerankScore != 0.2 {
		t.Errorf("rerank scores = %f,%f,%f, want .9,.6,.2", out[0].RerankScore, out[1].RerankScore, out[2].RerankScore)
	}
	if out[0].Reranker != "llm-judge" {
		t.Errorf("reranker = %q", out[0].Reranker)
	}
}

func TestLLMReranker_MaxCandidates(t *testing.T) {
	backend := judgeBackend(t)
	rr, err := NewLLMReranker(LLMRerankerConfig{Backend: backend, MaxCandidates: 2})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	results := []index.SearchResult{
		mkResult("a", "passage with aaa", 0.9),
		mkResult("b", "passage with bbb", 0.8),
		mkResult("c", "no marker here", 0.7),
	}
	out, err := rr.Rerank(context.Background(), "q", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if backend.CallCount != 2 {
		t.Errorf("expected 2 llm calls, got %d", backend.CallCount)
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 results, got %d", len(out))
	}
	// "c" was not judged; its RerankScore falls back to coarse (0.7).
	var cScore float64
	for _, r := range out {
		if r.Chunk.ID == "c" {
			cScore = r.RerankScore
		}
	}
	if cScore != 0.7 {
		t.Errorf("unjudged score = %f, want coarse 0.7", cScore)
	}
}

func TestLLMReranker_RequiresBackend(t *testing.T) {
	if _, err := NewLLMReranker(LLMRerankerConfig{}); err == nil {
		t.Fatal("expected error for nil backend")
	}
}

func TestLLMReranker_Empty(t *testing.T) {
	rr, _ := NewLLMReranker(LLMRerankerConfig{Backend: llm.NewMockBackend()})
	out, err := rr.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty, got %d", len(out))
	}
}
