package reranker

import (
	"context"
	"fmt"
	"math"

	"github.com/deagy/recall/embedder/onnx"
	"github.com/deagy/recall/index"
)

// CrossEncoderTokenizerFunc converts a (query, passage) pair into the named
// input tensors a cross-encoder ONNX model expects (e.g. "input_ids",
// "attention_mask", "token_type_ids"). Like the embedder's tokenizer it is
// dependency-injected so the reranker stays model-agnostic.
type CrossEncoderTokenizerFunc func(query, passage string) (map[string]*onnx.Tensor, error)

// CrossEncoderConfig configures a CrossEncoderReranker.
type CrossEncoderConfig struct {
	// Model is a loaded ONNX cross-encoder (required).
	Model *onnx.Model

	// Tokenize converts a (query, passage) pair into the model's inputs
	// (required).
	Tokenize CrossEncoderTokenizerFunc

	// Output names the model output to read as the relevance score. When
	// empty, the model's last declared output is used.
	Output string

	// Sigmoid applies the logistic function to the raw model output. Cross
	// -encoders trained for classification typically emit logits that are
	// mapped through sigmoid to a [0,1] relevance probability.
	Sigmoid bool
}

// CrossEncoderReranker scores each (query, passage) pair with a lightweight
// cross-encoder running on the bundled pure-Go ONNX runtime. Unlike
// bi-encoder similarity, a cross-encoder attends jointly over the query and
// passage, which usually yields a sharper relevance signal for fine ranking.
type CrossEncoderReranker struct {
	model     *onnx.Model
	tokenize  CrossEncoderTokenizerFunc
	output    string
	sigmoid   bool
	outputSet bool
}

// NewCrossEncoderReranker creates a cross-encoder reranker from a loaded model.
func NewCrossEncoderReranker(cfg CrossEncoderConfig) (*CrossEncoderReranker, error) {
	if cfg.Model == nil {
		return nil, fmt.Errorf("reranker: cross-encoder requires a model")
	}
	if cfg.Tokenize == nil {
		return nil, fmt.Errorf("reranker: cross-encoder requires a tokenizer function")
	}
	return &CrossEncoderReranker{
		model:     cfg.Model,
		tokenize:  cfg.Tokenize,
		output:    cfg.Output,
		sigmoid:   cfg.Sigmoid,
		outputSet: cfg.Output != "",
	}, nil
}

// NewCrossEncoderFile loads a cross-encoder ONNX model from disk.
func NewCrossEncoderFile(path string, cfg CrossEncoderConfig) (*CrossEncoderReranker, error) {
	m, err := onnx.LoadFile(path)
	if err != nil {
		return nil, err
	}
	cfg.Model = m
	return NewCrossEncoderReranker(cfg)
}

// Name implements Reranker.
func (r *CrossEncoderReranker) Name() string { return "cross-encoder" }

// outputName resolves the configured output, defaulting to the model's last
// declared output.
func (r *CrossEncoderReranker) outputName() (string, error) {
	if r.output != "" {
		return r.output, nil
	}
	outs := r.model.Outputs()
	if len(outs) == 0 {
		return "", fmt.Errorf("reranker: cross-encoder model declares no outputs; set Output explicitly")
	}
	return outs[len(outs)-1].Name, nil
}

// scorePair runs the model on one (query, passage) pair and returns the
// relevance score (applied through sigmoid when configured).
func (r *CrossEncoderReranker) scorePair(ctx context.Context, query, passage string) (float64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	feeds, err := r.tokenize(query, passage)
	if err != nil {
		return 0, fmt.Errorf("reranker: cross-encoder tokenization failed: %w", err)
	}
	outs, err := r.model.Run(ctx, feeds)
	if err != nil {
		return 0, fmt.Errorf("reranker: cross-encoder inference failed: %w", err)
	}
	name, err := r.outputName()
	if err != nil {
		return 0, err
	}
	t, ok := outs[name]
	if !ok || t == nil {
		return 0, fmt.Errorf("reranker: cross-encoder produced no output named %q", name)
	}
	vals, err := t.AsFloat64()
	if err != nil {
		return 0, fmt.Errorf("reranker: cross-encoder output %q is not numeric: %w", name, err)
	}
	if len(vals) == 0 {
		return 0, fmt.Errorf("reranker: cross-encoder output %q is empty", name)
	}
	// Use the first element (scalar or the CLS-pooling row head).
	score := vals[0]
	if r.sigmoid {
		score = 1.0 / (1.0 + math.Exp(-score))
	}
	return score, nil
}

// Rerank scores every result's chunk against the query with the cross-encoder
// and returns them ordered by the resulting relevance score.
func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}
	out := make([]index.SearchResult, 0, len(results))
	for i, res := range results {
		if res.Chunk == nil {
			return nil, fmt.Errorf("reranker: cross-encoder: result %d has no chunk", i)
		}
		score, err := r.scorePair(ctx, query, res.Chunk.Content)
		if err != nil {
			return nil, err
		}
		r2 := res
		r2.RerankScore = score
		out = append(out, r2)
	}
	return finalize(r.Name(), out), nil
}
