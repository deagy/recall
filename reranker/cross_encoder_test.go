package reranker

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/embedder/onnx"
	"github.com/deagy/recall/index"
)

// buildCEModel encodes a tiny cross-encoder: score = sigmoid(input_ids * 2).
// The Mul node plus the reranker's sigmoid flag exercise the ONNX runtime.
func buildCEModel(t *testing.T) *onnx.Model {
	t.Helper()
	w, err := onnx.NewTensor([]int64{1}, onnx.Float32, []float32{2})
	if err != nil {
		t.Fatalf("tensor: %v", err)
	}
	nodes := []onnx.NodeSpec{
		{Op: "Mul", Inputs: []string{"input_ids", "w"}, Outputs: []string{"logits"}},
	}
	inputs := []onnx.NamedType{{Name: "input_ids", Dtype: onnx.Float32}}
	outputs := []onnx.NamedType{{Name: "logits", Dtype: onnx.Float32}}
	data := onnx.Encode(nodes, map[string]*onnx.Tensor{"w": w}, inputs, outputs)
	m, err := onnx.Load(data)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return m
}

// overlapTokenize maps (query, passage) to a single "input_ids" float in [0,1]
// representing the fraction of query terms present in the passage.
func overlapTokenize(query, passage string) (map[string]*onnx.Tensor, error) {
	q := strings.Fields(strings.ToLower(query))
	p := strings.ToLower(passage)
	if len(q) == 0 {
		tensor, _ := onnx.NewTensor([]int64{1}, onnx.Float32, []float32{0})
		return map[string]*onnx.Tensor{"input_ids": tensor}, nil
	}
	hits := 0
	for _, w := range q {
		if strings.Contains(p, w) {
			hits++
		}
	}
	v, _ := onnx.NewTensor([]int64{1}, onnx.Float32, []float32{float32(hits) / float32(len(q))})
	return map[string]*onnx.Tensor{"input_ids": v}, nil
}

func TestCrossEncoderReranker_RanksByModelScore(t *testing.T) {
	model := buildCEModel(t)
	rr, err := NewCrossEncoderReranker(CrossEncoderConfig{
		Model:    model,
		Tokenize: overlapTokenize,
		Sigmoid:  true,
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	results := []index.SearchResult{
		// Coarse scores are intentionally misleading; the cross-encoder score
		// (based on term overlap) should reorder them.
		mkResult("none", "the quick brown fox jumps over the lazy dog", 0.99),
		mkResult("some", "retrieval of many documents", 0.5),
		mkResult("all", "retrieval and generation of documents", 0.1),
	}
	out, err := rr.Rerank(context.Background(), "retrieval generation", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	// overlap("retrieval generation", ...): none=0, some=1/2, all=2/2
	// sigmoid(2*overlap): none=.5, some=.731, all=.881
	if out[0].Chunk.ID != "all" || out[1].Chunk.ID != "some" || out[2].Chunk.ID != "none" {
		t.Fatalf("order = %s %s %s, want all some none", out[0].Chunk.ID, out[1].Chunk.ID, out[2].Chunk.ID)
	}
	if out[0].RerankScore <= out[1].RerankScore || out[1].RerankScore <= out[2].RerankScore {
		t.Errorf("scores not strictly decreasing: %f %f %f", out[0].RerankScore, out[1].RerankScore, out[2].RerankScore)
	}
	if out[0].Reranker != "cross-encoder" {
		t.Errorf("reranker = %q", out[0].Reranker)
	}
	// Sigmoid output must be in (0,1).
	for _, r := range out {
		if r.RerankScore <= 0 || r.RerankScore >= 1 {
			t.Errorf("sigmoid score %f out of (0,1)", r.RerankScore)
		}
	}
}

func TestCrossEncoderReranker_NoSigmoid(t *testing.T) {
	model := buildCEModel(t)
	rr, err := NewCrossEncoderReranker(CrossEncoderConfig{Model: model, Tokenize: overlapTokenize})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	results := []index.SearchResult{mkResult("all", "retrieval generation pipeline", 0.5)}
	out, err := rr.Rerank(context.Background(), "retrieval generation", results)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	// overlap=1.0, model output = 1.0*2 = 2.0, no sigmoid.
	if out[0].RerankScore != 2.0 {
		t.Errorf("no-sigmoid score = %f, want 2.0", out[0].RerankScore)
	}
}

func TestCrossEncoderReranker_RequiresConfig(t *testing.T) {
	model := buildCEModel(t)
	if _, err := NewCrossEncoderReranker(CrossEncoderConfig{Model: model}); err == nil {
		t.Error("expected error for missing tokenizer")
	}
	if _, err := NewCrossEncoderReranker(CrossEncoderConfig{Tokenize: overlapTokenize}); err == nil {
		t.Error("expected error for missing model")
	}
}

func TestCrossEncoderReranker_Empty(t *testing.T) {
	model := buildCEModel(t)
	rr, _ := NewCrossEncoderReranker(CrossEncoderConfig{Model: model, Tokenize: overlapTokenize})
	out, err := rr.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty, got %d", len(out))
	}
}
