package embedder

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/deagy/recall/embedder/onnx"
)

// TokenizerFunc converts a text into the named input tensors an ONNX
// embedding model expects (e.g. "input_ids", "attention_mask"). Tokenization
// is deliberately dependency-injected: the ONNX runtime executes the model,
// while the caller supplies the model-specific tokenizer.
type TokenizerFunc func(text string) (map[string]*onnx.Tensor, error)

// OnnxEmbedderConfig configures an OnnxEmbedder.
type OnnxEmbedderConfig struct {
	// Model is a loaded ONNX model (required).
	Model *onnx.Model

	// Tokenize converts each text into the model's named inputs (required).
	Tokenize TokenizerFunc

	// Output names the model output to read as the embedding. When empty,
	// the model's last declared output is used.
	Output string

	// Normalize L2-normalizes the resulting vectors. Most
	// sentence-transformer models expect normalized inputs for cosine
	// similarity search.
	Normalize bool

	// Dimension optionally pins the output dimension, skipping the lazy
	// probe run that Dimension() would otherwise perform.
	Dimension int

	// BatchConcurrency caps how many sequences EmbedBatch executes in
	// parallel. Zero (the default) auto-selects a worker count from the
	// available CPUs (capped at 8). Values <= 0 are treated as the
	// default; a value of 1 forces sequential execution. Note that peak
	// memory scales linearly with concurrency, since each in-flight
	// sequence holds its full intermediate tensor state.
	BatchConcurrency int
}

// OnnxEmbedder is a zero-network embedder backed by a pure-Go ONNX
// inference runtime. It runs sentence-transformer (or similar) ONNX
// exports locally.
type OnnxEmbedder struct {
	model            *onnx.Model
	tokenize         TokenizerFunc
	output           string
	normalize        bool
	batchConcurrency int

	once   sync.Once
	dim    int
	dimErr error
	dimSet bool
}

// Compile-time assertion that OnnxEmbedder satisfies the public Embedder
// interface (Embed, EmbedBatch, Dimension).
var _ Embedder = (*OnnxEmbedder)(nil)

// NewOnnxEmbedder creates an embedder from a loaded ONNX model.
func NewOnnxEmbedder(cfg OnnxEmbedderConfig) (*OnnxEmbedder, error) {
	if cfg.Model == nil {
		return nil, fmt.Errorf("embedder: onnx embedder requires a model")
	}
	if cfg.Tokenize == nil {
		return nil, fmt.Errorf("embedder: onnx embedder requires a tokenizer function")
	}
	e := &OnnxEmbedder{
		model:            cfg.Model,
		tokenize:         cfg.Tokenize,
		output:           cfg.Output,
		normalize:        cfg.Normalize,
		batchConcurrency: cfg.BatchConcurrency,
	}
	if cfg.Dimension > 0 {
		e.dim = cfg.Dimension
		e.dimSet = true
	}
	return e, nil
}

// NewOnnxEmbedderFile loads an ONNX model from disk and creates an embedder.
func NewOnnxEmbedderFile(path string, cfg OnnxEmbedderConfig) (*OnnxEmbedder, error) {
	m, err := onnx.LoadFile(path)
	if err != nil {
		return nil, err
	}
	cfg.Model = m
	return NewOnnxEmbedder(cfg)
}

// outputName resolves the configured output, defaulting to the last
// declared model output.
func (e *OnnxEmbedder) outputName() string {
	if e.output != "" {
		return e.output
	}
	outs := e.model.Outputs()
	if len(outs) == 0 {
		return ""
	}
	return outs[len(outs)-1].Name
}

// readVector extracts the embedding vector from a model output map. A 2-D
// output of the form [S, D] uses row 0 (the CLS token convention); a
// rank-1 tensor is already the vector.
func (e *OnnxEmbedder) readVector(outs map[string]*onnx.Tensor) ([]float32, error) {
	name := e.outputName()
	if name == "" {
		return nil, fmt.Errorf("embedder: onnx model declares no outputs; set Output explicitly")
	}
	t, ok := outs[name]
	if !ok || t == nil {
		return nil, fmt.Errorf("embedder: onnx model produced no output named %q", name)
	}
	vec, err := t.AsFloat64()
	if err != nil {
		return nil, fmt.Errorf("embedder: onnx output %q is not numeric: %w", name, err)
	}
	if len(t.Shape) >= 2 {
		d := int(t.Shape[len(t.Shape)-1])
		vec = vec[:d]
	}
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = float32(v)
	}
	if e.normalize {
		l2Normalize(out)
	}
	return out, nil
}

// embedOne tokenizes, runs the model, and reads the embedding vector.
func (e *OnnxEmbedder) embedOne(ctx context.Context, text string) ([]float32, error) {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	feeds, err := e.tokenize(text)
	if err != nil {
		return nil, fmt.Errorf("embedder: onnx tokenization failed: %w", err)
	}
	outs, err := e.model.Run(ctx, feeds)
	if err != nil {
		return nil, fmt.Errorf("embedder: onnx inference failed: %w", err)
	}
	return e.readVector(outs)
}

// Embed converts a single text into an embedding vector.
func (e *OnnxEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return e.embedOne(ctx, text)
}

// EmbedBatch converts multiple texts into embedding vectors, one per text.
// Sequences are tokenized sequentially (cheap) and then executed in
// parallel by the ONNX runtime, with the worker count controlled by
// BatchConcurrency (0 = auto).
func (e *OnnxEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	feeds := make([]map[string]*onnx.Tensor, len(texts))
	for i, text := range texts {
		f, err := e.tokenize(text)
		if err != nil {
			return nil, fmt.Errorf("embedder: onnx embed batch item %d: tokenization failed: %w", i, err)
		}
		feeds[i] = f
	}
	outs, err := e.model.BatchRun(ctx, feeds, e.batchConcurrency)
	if err != nil {
		return nil, fmt.Errorf("embedder: onnx embed batch: %w", err)
	}
	vecs := make([][]float32, len(texts))
	for i, out := range outs {
		vec, err := e.readVector(out)
		if err != nil {
			return nil, fmt.Errorf("embedder: onnx embed batch item %d: %w", i, err)
		}
		vecs[i] = vec
	}
	return vecs, nil
}

// Dimension returns the output dimension, probing the model once with a
// minimal text if it was not pinned in the config.
func (e *OnnxEmbedder) Dimension() int {
	if !e.dimSet {
		e.once.Do(func() {
			vec, err := e.embedOne(context.Background(), "dimension probe")
			if err != nil {
				e.dimErr = err
				return
			}
			e.dim = len(vec)
			e.dimSet = true
		})
	}
	return e.dim
}

// l2Normalize scales v to unit length; a zero vector is left unchanged.
func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	norm := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= norm
	}
}
