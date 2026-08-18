package embedder

import (
	"context"
	"os"
	"testing"

	"github.com/deagy/recall/embedder/onnx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testVocab    = 256 // every byte maps to a distinct token
	testHidden   = 8
	testSeqLen   = 4
	testModelDim = 8
)

// miniStModelBytes encodes the tiny sentence-transformer-style ONNX model:
// byte-token embeddings, layer norm, and masked mean pooling — the operator
// family used by real ST exports, but small enough to keep in-repo.
func miniStModelBytes(t *testing.T) []byte {
	t.Helper()
	emb := make([]float32, testVocab*testHidden)
	for i := 0; i < testVocab*testHidden; i++ {
		emb[i] = float32(i+1) / 1000.0
	}
	embT, err := onnx.NewTensor([]int64{testVocab, testHidden}, onnx.Float32, emb)
	require.NoError(t, err)
	lnScale := make([]float32, testHidden)
	for i := range lnScale {
		lnScale[i] = 1
	}
	lnScaleT, err := onnx.NewTensor([]int64{testHidden}, onnx.Float32, lnScale)
	require.NoError(t, err)
	lnBiasT, err := onnx.NewTensor([]int64{testHidden}, onnx.Float32, make([]float32, testHidden))
	require.NoError(t, err)

	nodes := []onnx.NodeSpec{
		{Op: "Gather", Inputs: []string{"word_emb", "input_ids"}, Outputs: []string{"tokens"}, IntAttrs: map[string]int64{"axis": 0}},
		{Op: "LayerNormalization", Inputs: []string{"tokens", "ln_scale", "ln_bias"}, Outputs: []string{"normed"}},
		{Op: "Cast", Inputs: []string{"attention_mask"}, Outputs: []string{"mask_f"}, IntAttrs: map[string]int64{"to": int64(onnx.Float32)}},
		{Op: "Unsqueeze", Inputs: []string{"mask_f", "ax2"}, Outputs: []string{"mask_3d"}},
		{Op: "Mul", Inputs: []string{"normed", "mask_3d"}, Outputs: []string{"masked"}},
		{Op: "ReduceSum", Inputs: []string{"masked", "axes"}, Outputs: []string{"summed"}, IntAttrs: map[string]int64{"keepdims": 0}},
		{Op: "ReduceSum", Inputs: []string{"mask_f", "axes"}, Outputs: []string{"counts"}, IntAttrs: map[string]int64{"keepdims": 0}},
		{Op: "Reshape", Inputs: []string{"counts", "counts_shape"}, Outputs: []string{"counts_2d"}},
		{Op: "Div", Inputs: []string{"summed", "counts_2d"}, Outputs: []string{"pooled"}},
	}
	b := onnx.Encode(nodes, map[string]*onnx.Tensor{
		"word_emb":     embT,
		"ln_scale":     lnScaleT,
		"ln_bias":      lnBiasT,
		"axes":         mustInt64Tensor(t, 1),
		"ax2":          mustInt64Tensor(t, 2),
		"counts_shape": mustInt64Tensor(t, 1, 1),
	},
		[]onnx.NamedType{{Name: "input_ids", Dtype: onnx.Int64}, {Name: "attention_mask", Dtype: onnx.Int64}},
		[]onnx.NamedType{{Name: "pooled", Dtype: onnx.Float32}})
	return b
}

func buildMiniStModel(t *testing.T) *onnx.Model {
	t.Helper()
	m, err := onnx.Load(miniStModelBytes(t))
	require.NoError(t, err)
	return m
}

func mustInt64Tensor(t *testing.T, vs ...int64) *onnx.Tensor {
	t.Helper()
	tn, err := onnx.NewTensor([]int64{int64(len(vs))}, onnx.Int64, vs)
	require.NoError(t, err)
	return tn
}

// byteTokenizer maps each text byte to a token id, right-padding to a fixed
// sequence length — mimicking a real tokenizer's contract of fixed-shape
// model inputs.
func byteTokenizer(text string) (map[string]*onnx.Tensor, error) {
	ids := make([]int64, testSeqLen)
	mask := make([]int64, testSeqLen)
	for i := 0; i < testSeqLen; i++ {
		if i < len(text) {
			ids[i] = int64(text[i])
			mask[i] = 1
		}
	}
	idT, err := onnx.NewTensor([]int64{1, testSeqLen}, onnx.Int64, ids)
	if err != nil {
		return nil, err
	}
	maskT, err := onnx.NewTensor([]int64{1, testSeqLen}, onnx.Int64, mask)
	if err != nil {
		return nil, err
	}
	return map[string]*onnx.Tensor{"input_ids": idT, "attention_mask": maskT}, nil
}

func newTestOnnxEmbedder(t *testing.T, normalize bool) *OnnxEmbedder {
	t.Helper()
	e, err := NewOnnxEmbedder(OnnxEmbedderConfig{
		Model:     buildMiniStModel(t),
		Tokenize:  byteTokenizer,
		Output:    "pooled",
		Normalize: normalize,
	})
	require.NoError(t, err)
	return e
}

func TestOnnxEmbedder(t *testing.T) {
	e := newTestOnnxEmbedder(t, false)
	require.Equal(t, testModelDim, e.Dimension())

	ctx := context.Background()
	vec, err := e.Embed(ctx, "hi")
	require.NoError(t, err)
	require.Len(t, vec, testModelDim)

	// Deterministic: same text, same vector.
	vec2, err := e.Embed(ctx, "hi")
	require.NoError(t, err)
	assert.Equal(t, vec, vec2)

	// Different text yields a different vector.
	vec3, err := e.Embed(ctx, "yo")
	require.NoError(t, err)
	assert.NotEqual(t, vec, vec3)

	// Padded short text differs from the full-length embedding.
	vec4, err := e.Embed(ctx, "h")
	require.NoError(t, err)
	assert.NotEqual(t, vec, vec4)

	// Batch matches per-item results.
	batch, err := e.EmbedBatch(ctx, []string{"hi", "yo"})
	require.NoError(t, err)
	require.Len(t, batch, 2)
	assert.Equal(t, vec, batch[0])
	assert.Equal(t, vec3, batch[1])

	// Cancellation is honored before the run.
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = e.Embed(canceled, "hi")
	require.Error(t, err)
}

func TestOnnxEmbedderNormalized(t *testing.T) {
	e := newTestOnnxEmbedder(t, true)
	vec, err := e.Embed(context.Background(), "hello world")
	require.NoError(t, err)
	var sum float64
	for _, x := range vec {
		sum += float64(x) * float64(x)
	}
	assert.InDelta(t, 1.0, sum, 1e-5)
}

func TestOnnxEmbedderConfigValidation(t *testing.T) {
	_, err := NewOnnxEmbedder(OnnxEmbedderConfig{Tokenize: byteTokenizer})
	require.Error(t, err)
	_, err = NewOnnxEmbedder(OnnxEmbedderConfig{Model: buildMiniStModel(t)})
	require.Error(t, err)
}

func TestOnnxEmbedderFile(t *testing.T) {
	path := t.TempDir() + "/m.onnx"
	require.NoError(t, os.WriteFile(path, miniStModelBytes(t), 0o644))
	_, err := NewOnnxEmbedderFile(path+"/nope.onnx", OnnxEmbedderConfig{Tokenize: byteTokenizer})
	require.Error(t, err)
	e, err := NewOnnxEmbedderFile(path, OnnxEmbedderConfig{Tokenize: byteTokenizer, Output: "pooled"})
	require.NoError(t, err)
	vec, err := e.Embed(context.Background(), "hi")
	require.NoError(t, err)
	require.Len(t, vec, testModelDim)
	// The file-loaded model behaves identically to an in-memory one.
	direct := newTestOnnxEmbedder(t, false)
	want, err := direct.Embed(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, want, vec)
}

func TestPipelineWithOnnxEmbedder(t *testing.T) {
	e := newTestOnnxEmbedder(t, false)
	p, err := NewPipeline(e, NewMockEmbedder(testModelDim))
	require.NoError(t, err)
	assert.Equal(t, testModelDim, p.Dimension())

	vec, err := p.Embed(context.Background(), "hi")
	require.NoError(t, err)
	// The local ONNX embedder succeeds, so the mock is never consulted.
	direct, err := e.Embed(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, direct, vec)
}
