package onnx

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- helpers ---------------------------------------------------------------

func ft(vals ...float32) *Tensor {
	t, err := NewTensor([]int64{int64(len(vals))}, Float32, vals)
	if err != nil {
		panic(err)
	}
	return t
}

func ft2(r, c int, vals ...float32) *Tensor {
	t, err := NewTensor([]int64{int64(r), int64(c)}, Float32, vals)
	if err != nil {
		panic(err)
	}
	return t
}

func it64(vals ...int64) *Tensor {
	t, err := NewTensor([]int64{int64(len(vals))}, Int64, vals)
	if err != nil {
		panic(err)
	}
	return t
}

// runGraph encodes a model from the given nodes/initializers, loads it, and
// runs it with the provided feed tensors.
func runGraph(t *testing.T, nodes []NodeSpec, inits map[string]*Tensor, feeds map[string]*Tensor) map[string]*Tensor {
	t.Helper()
	var inputs, outputs []NamedType
	for name, tn := range feeds {
		inputs = append(inputs, NamedType{Name: name, Dtype: tn.Dtype})
	}
	for _, nd := range nodes {
		for _, o := range nd.Outputs {
			outputs = append(outputs, NamedType{Name: o, Dtype: Float32})
		}
	}
	m, err := Load(Encode(nodes, inits, inputs, outputs))
	require.NoError(t, err)
	out, err := m.Run(context.Background(), feeds)
	require.NoError(t, err)
	return out
}

// runNode runs a single node against the named feed tensors and returns its
// first output.
func runNode(t *testing.T, nd NodeSpec, feeds map[string]*Tensor, inits map[string]*Tensor) *Tensor {
	t.Helper()
	out := runGraph(t, []NodeSpec{nd}, inits, feeds)
	return out[nd.Outputs[0]]
}

// --- parser round-trip -----------------------------------------------------

func TestEncodeLoad_RoundTrip(t *testing.T) {
	inits := map[string]*Tensor{
		"w":    ft2(2, 3, 1, 2, 3, 4, 5, 6),
		"idx":  it64(0, 1, 2, 3),
		"flag": mustTensor(NewTensor([]int64{2}, Bool, []bool{true, false})),
	}
	nd := NodeSpec{
		Op:         "MatMul",
		Inputs:     []string{"x", "w"},
		Outputs:    []string{"y"},
		IntAttrs:   map[string]int64{"axis": 1},
		FloatAttrs: map[string]float32{"scale": 1.5},
	}
	inputs := []NamedType{{Name: "x", Dtype: Float32}}
	outputs := []NamedType{{Name: "y", Dtype: Float32}}
	data := Encode([]NodeSpec{nd}, inits, inputs, outputs)
	m, err := Load(data)
	require.NoError(t, err)

	assert.Equal(t, uint32(8), m.IRVersion)
	require.Len(t, m.Opsets, 1)
	assert.Equal(t, "ai.onnx", m.Opsets[0].Domain)
	assert.Equal(t, int64(13), m.Opsets[0].Version)
	require.Len(t, m.Graph.Nodes, 1)
	assert.Equal(t, "MatMul", m.Graph.Nodes[0].OpType)
	assert.Equal(t, []string{"x", "w"}, m.Graph.Nodes[0].Inputs)
	require.Len(t, m.Graph.Nodes[0].Attrs, 2)
	gotAxis := int64(-1)
	var gotScale float32
	for _, a := range m.Graph.Nodes[0].Attrs {
		switch a.Name {
		case "axis":
			gotAxis = a.I
		case "scale":
			gotScale = a.F
		}
	}
	assert.Equal(t, int64(1), gotAxis)
	assert.InDelta(t, 1.5, float64(gotScale), 1e-6)

	w := m.Graph.Initializers["w"]
	require.NotNil(t, w)
	assert.Equal(t, []int64{2, 3}, w.Shape)
	assert.Equal(t, Float32, w.Dtype)
	assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, w.Data.([]float32))
	idx := m.Graph.Initializers["idx"]
	require.NotNil(t, idx)
	assert.Equal(t, []int64{0, 1, 2, 3}, idx.Data.([]int64))
	flag := m.Graph.Initializers["flag"]
	require.NotNil(t, flag)
	assert.Equal(t, []bool{true, false}, flag.Data.([]bool))

	// x is a feed input; w (initializer) is not.
	feeds := m.Graph.FeedInputs()
	require.Len(t, feeds, 1)
	assert.Equal(t, "x", feeds[0].Name)
	assert.Equal(t, Float32, feeds[0].Dtype)
}

func mustTensor(t *Tensor, err error) *Tensor {
	if err != nil {
		panic(err)
	}
	return t
}

func TestOpMatMul(t *testing.T) {
	// 2-D x 2-D.
	out := runNode(t, NodeSpec{Op: "MatMul", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 2, 1, 2, 3, 4), "b": ft2(2, 2, 5, 6, 7, 8)}, nil)
	assert.Equal(t, []float32{19, 22, 43, 50}, out.Data.([]float32))

	// Vector x matrix.
	out = runNode(t, NodeSpec{Op: "MatMul", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3), "b": ft2(3, 2, 1, 0, 0, 1, 1, 1)}, nil)
	assert.Equal(t, []int64{2}, out.Shape)
	assert.Equal(t, []float32{4, 5}, out.Data.([]float32))

	// Batched: [2,2,2] x [2,2].
	a := mustTensor(NewTensor([]int64{2, 2, 2}, Float32, []float32{
		1, 0, 0, 1, // identity
		2, 0, 0, 3,
	}))
	out = runNode(t, NodeSpec{Op: "MatMul", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": a, "b": ft2(2, 2, 5, 6, 7, 8)}, nil)
	assert.Equal(t, []int64{2, 2, 2}, out.Shape)
	assert.Equal(t, []float32{5, 6, 7, 8, 10, 12, 21, 24}, out.Data.([]float32))

	// Inner dimension mismatch.
	_, err := runNodeErr(t, NodeSpec{Op: "MatMul", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 3, 1, 2, 3, 4, 5, 6), "b": ft2(2, 2, 5, 6, 7, 8)}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "inner dimensions")
}

// runNodeErr runs a node and returns the error instead of failing the test.
func runNodeErr(t *testing.T, nd NodeSpec, feeds map[string]*Tensor, inits map[string]*Tensor) (*Tensor, error) {
	t.Helper()
	m, err := Load(Encode([]NodeSpec{nd}, inits, namedFeeds(feeds), namedOutputs(nd.Outputs)))
	if err != nil {
		return nil, err
	}
	out, err := m.Run(context.Background(), feeds)
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out[nd.Outputs[0]], nil
}

func namedFeeds(feeds map[string]*Tensor) []NamedType {
	var out []NamedType
	for name, tn := range feeds {
		out = append(out, NamedType{Name: name, Dtype: tn.Dtype})
	}
	return out
}

func namedOutputs(names []string) []NamedType {
	var out []NamedType
	for _, n := range names {
		out = append(out, NamedType{Name: n, Dtype: Float32})
	}
	return out
}

func TestOpGemm(t *testing.T) {
	// Plain: A @ B.
	out := runNode(t, NodeSpec{Op: "Gemm", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 2, 1, 2, 3, 4), "b": ft2(2, 2, 5, 6, 7, 8)}, nil)
	assert.Equal(t, []float32{19, 22, 43, 50}, out.Data.([]float32))

	// alpha/beta with C broadcast: alpha=2, beta=10, C=[1] scalar broadcast.
	out = runNode(t, NodeSpec{
		Op:         "Gemm",
		Inputs:     []string{"a", "b", "c"},
		Outputs:    []string{"y"},
		FloatAttrs: map[string]float32{"alpha": 2, "beta": 10},
	}, map[string]*Tensor{
		"a": ft2(1, 2, 1, 2),
		"b": ft2(2, 1, 3, 4),
		"c": ft(7),
	}, nil)
	// 2*(1*3+2*4) + 10*7 = 2*11 + 70 = 92
	assert.Equal(t, []float32{92}, out.Data.([]float32))

	// transA: A is [k, m], so Y = (A^T) @ B.
	out = runNode(t, NodeSpec{
		Op:     "Gemm",
		Inputs: []string{"a", "b"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"transA": 1},
	}, map[string]*Tensor{
		"a": ft2(2, 2, 1, 2, 3, 4), // A^T = [[1,3],[2,4]]
		"b": ft2(2, 1, 5, 6),
	}, nil)
	// [1*5+3*6, 2*5+4*6] = [23, 34]
	assert.Equal(t, []float32{23, 34}, out.Data.([]float32))

	// transB: B is [n, k], so Y = A @ (B^T).
	out = runNode(t, NodeSpec{
		Op:     "Gemm",
		Inputs: []string{"a", "b"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"transB": 1},
	}, map[string]*Tensor{
		"a": ft2(1, 2, 5, 6),
		"b": ft2(2, 2, 1, 2, 3, 4), // B^T = [[1,3],[2,4]]
	}, nil)
	// [[5,6]] @ [[1,3],[2,4]] = [[17, 39]]
	assert.Equal(t, []float32{17, 39}, out.Data.([]float32))
}

func TestOpElementwise(t *testing.T) {
	// Broadcast [2,1] + [3].
	out := runNode(t, NodeSpec{Op: "Add", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 1, 10, 20), "b": ft(1, 2, 3)}, nil)
	assert.Equal(t, []int64{2, 3}, out.Shape)
	assert.Equal(t, []float32{11, 12, 13, 21, 22, 23}, out.Data.([]float32))

	// Mul with scalar broadcast.
	out = runNode(t, NodeSpec{Op: "Mul", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3), "b": ft(2)}, nil)
	assert.Equal(t, []float32{2, 4, 6}, out.Data.([]float32))

	// Sub, Div.
	out = runNode(t, NodeSpec{Op: "Sub", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(5, 6), "b": ft(1, 2)}, nil)
	assert.Equal(t, []float32{4, 4}, out.Data.([]float32))
	out = runNode(t, NodeSpec{Op: "Div", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(8, 9), "b": ft(2, 3)}, nil)
	assert.Equal(t, []float32{4, 3}, out.Data.([]float32))

	// Comparisons produce bool tensors.
	out = runNode(t, NodeSpec{Op: "Greater", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(3, 1, 2), "b": ft(2, 2, 2)}, nil)
	assert.Equal(t, Bool, out.Dtype)
	assert.Equal(t, []bool{true, false, false}, out.Data.([]bool))

	// Where selects per-element.
	out = runNode(t, NodeSpec{Op: "Where", Inputs: []string{"c", "x", "y"}, Outputs: []string{"o"}},
		map[string]*Tensor{
			"c": mustTensor(NewTensor([]int64{3}, Bool, []bool{true, false, true})),
			"x": ft(1, 2, 3),
			"y": ft(10, 20, 30),
		}, nil)
	assert.Equal(t, []float32{1, 20, 3}, out.Data.([]float32))

	// And/Not on bools.
	bt := func(v ...bool) *Tensor { return mustTensor(NewTensor([]int64{int64(len(v))}, Bool, v)) }
	out = runNode(t, NodeSpec{Op: "And", Inputs: []string{"a", "b"}, Outputs: []string{"o"}},
		map[string]*Tensor{"a": bt(true, false, true), "b": bt(true, true, false)}, nil)
	assert.Equal(t, []bool{true, false, false}, out.Data.([]bool))
	out = runNode(t, NodeSpec{Op: "Not", Inputs: []string{"a"}, Outputs: []string{"o"}},
		map[string]*Tensor{"a": bt(true, false)}, nil)
	assert.Equal(t, []bool{false, true}, out.Data.([]bool))

	// Incompatible broadcast shapes error.
	_, err := runNodeErr(t, NodeSpec{Op: "Add", Inputs: []string{"a", "b"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 2, 1, 2, 3, 4), "b": ft2(3, 1, 5, 6, 7)}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "broadcast")
}

func TestOpUnaryMath(t *testing.T) {
	in := map[string]*Tensor{"a": ft(0, 1, -1)}
	out := runNode(t, NodeSpec{Op: "Neg", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.Equal(t, []float32{0, -1, 1}, out.Data.([]float32))
	out = runNode(t, NodeSpec{Op: "Abs", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.Equal(t, []float32{0, 1, 1}, out.Data.([]float32))
	out = runNode(t, NodeSpec{Op: "Relu", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.Equal(t, []float32{0, 1, 0}, out.Data.([]float32))
	out = runNode(t, NodeSpec{Op: "Sigmoid", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.InDelta(t, 0.5, float64(out.Data.([]float32)[0]), 1e-6)
	assert.InDelta(t, 0.7310586, float64(out.Data.([]float32)[1]), 1e-5)
	assert.InDelta(t, 0.2689414, float64(out.Data.([]float32)[2]), 1e-5)
	out = runNode(t, NodeSpec{Op: "Tanh", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.InDelta(t, 0.7615942, float64(out.Data.([]float32)[1]), 1e-5)
	out = runNode(t, NodeSpec{Op: "Exp", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.InDelta(t, math.E, float64(out.Data.([]float32)[1]), 1e-6)
	out = runNode(t, NodeSpec{Op: "Sqrt", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 4, 9)}, nil)
	assert.Equal(t, []float32{1, 2, 3}, out.Data.([]float32))
	out = runNode(t, NodeSpec{Op: "Reciprocal", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(2, 4)}, nil)
	assert.InDelta(t, 0.25, float64(out.Data.([]float32)[1]), 1e-7)

	// Gelu exact vs approximate.
	gelExact := func(x float64) float64 { return 0.5 * x * (1 + erf(x/math.Sqrt2)) }
	gelApprox := func(x float64) float64 {
		c := 2 / math.Sqrt(math.Pi)
		return 0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x)))
	}
	out = runNode(t, NodeSpec{Op: "Gelu", Inputs: []string{"a"}, Outputs: []string{"y"}}, in, nil)
	assert.InDelta(t, gelExact(1), float64(out.Data.([]float32)[1]), 2e-6)
	out = runNode(t, NodeSpec{Op: "Gelu", Inputs: []string{"a"}, Outputs: []string{"y"}, IntAttrs: map[string]int64{"approximate": 1}}, in, nil)
	assert.InDelta(t, gelApprox(1), float64(out.Data.([]float32)[1]), 2e-6)
}

func TestOpSoftmax(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Softmax", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3)}, nil)
	got := out.Data.([]float32)
	var sum float64
	for _, v := range got {
		sum += float64(v)
	}
	assert.InDelta(t, 1.0, sum, 1e-6)
	assert.InDelta(t, math.Exp(-1)/(math.Exp(-1)+1+math.Exp(1)), float64(got[0]), 1e-6)
	assert.True(t, got[2] > got[1] && got[1] > got[0])

	// Row-wise over axis 0: each column is independent.
	out = runNode(t, NodeSpec{
		Op:       "Softmax",
		Inputs:   []string{"a"},
		Outputs:  []string{"y"},
		IntAttrs: map[string]int64{"axis": 0},
	}, map[string]*Tensor{"a": ft2(2, 2, 1, 2, 3, 4)}, nil)
	got2 := out.Data.([]float32)
	assert.InDelta(t, 1.0, float64(got2[0])+float64(got2[2]), 1e-6)
	assert.InDelta(t, 1.0, float64(got2[1])+float64(got2[3]), 1e-6)
}

func TestOpLayerNormalization(t *testing.T) {
	// Normalize [2,3] over last dim with scale=2, bias=1.
	feeds := map[string]*Tensor{
		"x":     ft2(2, 3, 1, 2, 3, 4, 5, 6),
		"scale": ft(2),
		"bias":  ft(1),
	}
	out := runNode(t, NodeSpec{
		Op:      "LayerNormalization",
		Inputs:  []string{"x", "scale", "bias"},
		Outputs: []string{"y", "mean", "invstd"},
	}, feeds, nil)
	got := out.Data.([]float32)
	// Row 0: mean=2, var=2/3, invstd=1/sqrt(2/3+1e-5)
	want0 := func(x float64) float64 {
		invstd := 1 / math.Sqrt(2.0/3.0+1e-5)
		return (x-2)*invstd*2 + 1
	}
	for i, x := range []float64{1, 2, 3} {
		assert.InDelta(t, want0(x), float64(got[i]), 1e-4)
	}
}

func TestOpBatchNormalization(t *testing.T) {
	feeds := map[string]*Tensor{
		"x":     ft2(2, 3, 1, 2, 3, 4, 5, 6), // [N=2, C=3]
		"scale": ft(1, 1, 1),
		"b":     ft(0, 0, 0),
		"mean":  ft(1, 2, 3),
		"var":   ft(1, 1, 1),
	}
	out := runNode(t, NodeSpec{Op: "BatchNormalization", Inputs: []string{"x", "scale", "b", "mean", "var"}, Outputs: []string{"y"}}, feeds, nil)
	// (x - mean) * scale / sqrt(var + 1e-5) + b
	got := out.Data.([]float32)
	assert.InDelta(t, 0, float64(got[0]), 1e-4)
	assert.InDelta(t, 3, float64(got[3]), 1e-4)
	assert.InDelta(t, 3, float64(got[5]), 1e-4)
}

func TestOpReduce(t *testing.T) {
	a := ft2(2, 3, 1, 2, 3, 4, 5, 6)

	// ReduceMean along axis 1 with keepdims.
	out := runNode(t, NodeSpec{
		Op:           "ReduceMean",
		Inputs:       []string{"a"},
		Outputs:      []string{"y"},
		IntAttrs:     map[string]int64{"keepdims": 1},
		IntListAttrs: map[string][]int64{"axes": {1}},
	}, map[string]*Tensor{"a": a}, nil)
	assert.Equal(t, []int64{2, 1}, out.Shape)
	assert.Equal(t, []float32{2, 5}, out.Data.([]float32))

	// ReduceSum with axes from input tensor, keepdims=0.
	out = runNode(t, NodeSpec{
		Op:       "ReduceSum",
		Inputs:   []string{"a", "axes"},
		Outputs:  []string{"y"},
		IntAttrs: map[string]int64{"keepdims": 0},
	}, map[string]*Tensor{"a": a, "axes": it64(0)}, nil)
	assert.Equal(t, []int64{3}, out.Shape)
	assert.Equal(t, []float32{5, 7, 9}, out.Data.([]float32))

	// ReduceMax over all axes (no axes specified).
	out = runNode(t, NodeSpec{Op: "ReduceMax", Inputs: []string{"a"}, Outputs: []string{"y"}, IntAttrs: map[string]int64{"keepdims": 0}},
		map[string]*Tensor{"a": a}, nil)
	assert.Equal(t, []float32{6}, out.Data.([]float32))

	// ReduceMin along axis 0.
	out = runNode(t, NodeSpec{Op: "ReduceMin", Inputs: []string{"a"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"keepdims": 1}, IntListAttrs: map[string][]int64{"axes": {0}}},
		map[string]*Tensor{"a": a}, nil)
	assert.Equal(t, []float32{1, 2, 3}, out.Data.([]float32))
}

func TestOpShapeOps(t *testing.T) {
	// Reshape with -1 inference.
	out := runNode(t, NodeSpec{Op: "Reshape", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 3, 1, 2, 3, 4, 5, 6), "shape": it64(3, -1)}, nil)
	assert.Equal(t, []int64{3, 2}, out.Shape)
	assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, out.Data.([]float32))

	// Transpose with explicit perm.
	out = runNode(t, NodeSpec{
		Op:           "Transpose",
		Inputs:       []string{"a"},
		Outputs:      []string{"y"},
		IntListAttrs: map[string][]int64{"perm": {1, 0}},
	}, map[string]*Tensor{"a": ft2(2, 3, 1, 2, 3, 4, 5, 6)}, nil)
	assert.Equal(t, []int64{3, 2}, out.Shape)
	assert.Equal(t, []float32{1, 4, 2, 5, 3, 6}, out.Data.([]float32))

	// Flatten.
	out = runNode(t, NodeSpec{Op: "Flatten", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 3, 1, 2, 3, 4, 5, 6)}, nil)
	assert.Equal(t, []int64{2, 3}, out.Shape)

	// Concat along axis 1.
	out = runNode(t, NodeSpec{
		Op:       "Concat",
		Inputs:   []string{"a", "b"},
		Outputs:  []string{"y"},
		IntAttrs: map[string]int64{"axis": 1},
	}, map[string]*Tensor{"a": ft2(2, 2, 1, 2, 3, 4), "b": ft2(2, 1, 5, 6)}, nil)
	assert.Equal(t, []int64{2, 3}, out.Shape)
	assert.Equal(t, []float32{1, 2, 5, 3, 4, 6}, out.Data.([]float32))

	// Split with attribute sizes.
	nd := NodeSpec{
		Op:           "Split",
		Inputs:       []string{"a"},
		Outputs:      []string{"o1", "o2", "o3"},
		IntListAttrs: map[string][]int64{"split": {2, 2, 2}},
	}
	outTensors := runGraph(t, []NodeSpec{nd}, nil, map[string]*Tensor{"a": ft(1, 2, 3, 4, 5, 6)})
	assert.Equal(t, []float32{1, 2}, outTensors["o1"].Data.([]float32))
	assert.Equal(t, []float32{3, 4}, outTensors["o2"].Data.([]float32))
	assert.Equal(t, []float32{5, 6}, outTensors["o3"].Data.([]float32))

	// Slice: rows 1.. (second row onward) of a 2x3.
	out = runNode(t, NodeSpec{Op: "Slice", Inputs: []string{"a", "s", "e", "ax"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 3, 1, 2, 3, 4, 5, 6), "s": it64(1), "e": it64(2), "ax": it64(0)}, nil)
	assert.Equal(t, []int64{1, 3}, out.Shape)
	assert.Equal(t, []float32{4, 5, 6}, out.Data.([]float32))

	// Slice with negative step: indices 4,3,2,1 (end is exclusive).
	out = runNode(t, NodeSpec{Op: "Slice", Inputs: []string{"a", "s", "e", "ax", "st"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3, 4, 5), "s": it64(4), "e": it64(0), "ax": it64(0), "st": it64(-1)}, nil)
	assert.Equal(t, []float32{5, 4, 3, 2}, out.Data.([]float32))

	// Pad: add 1 on each side with value 7.
	out = runNode(t, NodeSpec{
		Op:           "Pad",
		Inputs:       []string{"a"},
		Outputs:      []string{"y"},
		IntListAttrs: map[string][]int64{"pads": {1, 1}},
		FloatAttrs:   map[string]float32{"value": 7},
	}, map[string]*Tensor{"a": ft(1, 2, 3)}, nil)
	assert.Equal(t, []float32{7, 1, 2, 3, 7}, out.Data.([]float32))

	// Tile: repeat [2,1] with repeats [1,2].
	out = runNode(t, NodeSpec{Op: "Tile", Inputs: []string{"a", "r"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 1, 5, 6), "r": it64(1, 2)}, nil)
	assert.Equal(t, []float32{5, 5, 6, 6}, out.Data.([]float32))

	// Tile with rank-1 repeats on rank-2 data (right-aligned).
	out = runNode(t, NodeSpec{Op: "Tile", Inputs: []string{"a", "r"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2), "r": it64(2)}, nil)
	assert.Equal(t, []float32{1, 2, 1, 2}, out.Data.([]float32))

	// Expand: [2,1] -> [2,3].
	out = runNode(t, NodeSpec{Op: "Expand", Inputs: []string{"a", "s"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 1, 10, 20), "s": it64(2, 3)}, nil)
	assert.Equal(t, []int64{2, 3}, out.Shape)
	assert.Equal(t, []float32{10, 10, 10, 20, 20, 20}, out.Data.([]float32))

	// Gather along axis 0.
	out = runNode(t, NodeSpec{Op: "Gather", Inputs: []string{"a", "i"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(3, 2, 1, 2, 3, 4, 5, 6), "i": it64(2, 0)}, nil)
	assert.Equal(t, []int64{2, 2}, out.Shape)
	assert.Equal(t, []float32{5, 6, 1, 2}, out.Data.([]float32))

	// Shape op.
	out = runNode(t, NodeSpec{Op: "Shape", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 3, 1, 2, 3, 4, 5, 6)}, nil)
	assert.Equal(t, []int64{2, 3}, out.Data.([]int64))

	// Squeeze removes size-1 dims.
	sq := mustTensor(NewTensor([]int64{2, 1, 3}, Float32, []float32{1, 2, 3, 4, 5, 6}))
	out = runNode(t, NodeSpec{Op: "Squeeze", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": sq}, nil)
	assert.Equal(t, []int64{2, 3}, out.Shape)

	// Unsqueeze inserts size-1 dims.
	out = runNode(t, NodeSpec{Op: "Unsqueeze", Inputs: []string{"a", "ax"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3), "ax": it64(1)}, nil)
	assert.Equal(t, []int64{3, 1}, out.Shape)

	// Range.
	out = runNode(t, NodeSpec{Op: "Range", Inputs: []string{"s", "l", "d"}, Outputs: []string{"y"}},
		map[string]*Tensor{"s": it64(0), "l": it64(5), "d": it64(2)}, nil)
	assert.Equal(t, []int64{0, 2, 4}, out.Data.([]int64))

	// Cast float -> int64.
	out = runNode(t, NodeSpec{
		Op:       "Cast",
		Inputs:   []string{"a"},
		Outputs:  []string{"y"},
		IntAttrs: map[string]int64{"to": int64(Int64)},
	}, map[string]*Tensor{"a": ft(1.7, 2.3)}, nil)
	assert.Equal(t, []int64{1, 2}, out.Data.([]int64))

	// Cast int64 -> float.
	out = runNode(t, NodeSpec{
		Op:       "Cast",
		Inputs:   []string{"a"},
		Outputs:  []string{"y"},
		IntAttrs: map[string]int64{"to": int64(Float32)},
	}, map[string]*Tensor{"a": it64(3, 4)}, nil)
	assert.Equal(t, []float32{3, 4}, out.Data.([]float32))

	// Identity and Constant.
	out = runNode(t, NodeSpec{Op: "Identity", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2)}, nil)
	assert.Equal(t, []float32{1, 2}, out.Data.([]float32))
	constant := mustTensor(NewTensor([]int64{2}, Float32, []float32{9, 8}))
	cOut := runGraph(t, []NodeSpec{{Op: "Constant", Outputs: []string{"c"}, TensorAttrs: map[string]*Tensor{"value": constant}}}, nil, nil)
	assert.Equal(t, []float32{9, 8}, cOut["c"].Data.([]float32))
}

func TestLoad_Errors(t *testing.T) {
	// Truncated input.
	if _, err := Load([]byte{0x08, 0x80, 0x80, 0x80, 0x80, 0x04}); err == nil {
		t.Fatal("expected error for truncated model")
	}
	// No nodes.
	m, err := Load(Encode(nil, nil, nil, nil))
	if err == nil {
		t.Fatalf("expected error for nodeless model, got %+v", m)
	}
	// Unsupported operator surfaces a named error.
	_, err = runNodeErr(t, NodeSpec{Op: "Loop", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1)}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Loop")
	assert.Contains(t, err.Error(), "not supported")
}

func TestRun_InputValidation(t *testing.T) {
	m, err := Load(Encode([]NodeSpec{{Op: "Identity", Inputs: []string{"x"}, Outputs: []string{"y"}}}, nil,
		[]NamedType{{Name: "x", Dtype: Float32}}, []NamedType{{Name: "y", Dtype: Float32}}))
	require.NoError(t, err)

	// Missing required input.
	_, err = m.Run(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing graph input")

	// Unknown input name.
	_, err = m.Run(context.Background(), map[string]*Tensor{"nope": ft(1)})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unexpected input")

	// Happy path.
	out, err := m.Run(context.Background(), map[string]*Tensor{"x": ft(1, 2)})
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 2}, out["y"].Data.([]float32))
}

// TestMiniTransformerModel exercises a small but realistic BERT-like
// pipeline: embedding lookup, layer norm, and masked mean pooling — the
// same operator family used by sentence-transformer ONNX exports.
func TestMiniTransformerModel(t *testing.T) {
	const (
		vocab  = 8
		hidden = 4
		seq    = 3
		batch  = 2
	)
	// Word embedding table [vocab, hidden]; token i maps to a distinct row.
	emb := make([]float32, vocab*hidden)
	for i := 0; i < vocab*hidden; i++ {
		emb[i] = float32(i+1) / 10.0
	}
	embT := mustTensor(NewTensor([]int64{vocab, hidden}, Float32, emb))
	layernormScale := mustTensor(NewTensor([]int64{hidden}, Float32, []float32{1, 1, 1, 1}))
	layernormBias := mustTensor(NewTensor([]int64{hidden}, Float32, []float32{0, 0, 0, 0}))

	nodes := []NodeSpec{
		// token embeddings: Gather(input_ids [B,S], axis=0) -> [B,S,H]
		{Op: "Gather", Inputs: []string{"word_emb", "input_ids"}, Outputs: []string{"tokens"}, IntAttrs: map[string]int64{"axis": 0}},
		// layer norm over hidden
		{Op: "LayerNormalization", Inputs: []string{"tokens", "ln_scale", "ln_bias"}, Outputs: []string{"normed"}},
		// masked mean pooling: (normed * mask) -> ReduceSum axis=1 -> / mask count
		{Op: "Cast", Inputs: []string{"attention_mask"}, Outputs: []string{"mask_f"}, IntAttrs: map[string]int64{"to": int64(Float32)}},
		{Op: "Unsqueeze", Inputs: []string{"mask_f", "ax1"}, Outputs: []string{"mask_3d"}},
		{Op: "Mul", Inputs: []string{"normed", "mask_3d"}, Outputs: []string{"masked"}},
		{Op: "ReduceSum", Inputs: []string{"masked", "axes1"}, Outputs: []string{"summed"}, IntAttrs: map[string]int64{"keepdims": 0}},
		{Op: "ReduceSum", Inputs: []string{"mask_f", "axes1"}, Outputs: []string{"counts"}, IntAttrs: map[string]int64{"keepdims": 0}},
		{Op: "Reshape", Inputs: []string{"counts", "counts_shape"}, Outputs: []string{"counts_2d"}},
		{Op: "Div", Inputs: []string{"summed", "counts_2d"}, Outputs: []string{"pooled"}},
	}
	// The axes tensor for the reductions (axis 1).
	axes1 := it64(1)

	inits := map[string]*Tensor{
		"word_emb":     embT,
		"ln_scale":     layernormScale,
		"ln_bias":      layernormBias,
		"axes1":        axes1,
		"ax1":          it64(2),
		"counts_shape": it64(2, 1),
	}
	inputs := []NamedType{
		{Name: "input_ids", Dtype: Int64},
		{Name: "attention_mask", Dtype: Int64},
	}
	outputs := []NamedType{{Name: "pooled", Dtype: Float32}}
	m, err := Load(Encode(nodes, inits, inputs, outputs))
	require.NoError(t, err)

	ids := mustTensor(NewTensor([]int64{batch, seq}, Int64, []int64{0, 1, 2, 3, 4, 5}))
	mask := mustTensor(NewTensor([]int64{batch, seq}, Int64, []int64{1, 1, 0, 1, 1, 1}))
	out, err := m.Run(context.Background(), map[string]*Tensor{"input_ids": ids, "attention_mask": mask})
	require.NoError(t, err)
	pooled := out["pooled"]
	require.NotNil(t, pooled)
	assert.Equal(t, []int64{batch, hidden}, pooled.Shape)
	got := pooled.Data.([]float32)
	// Determinism: running twice yields identical results.
	out2, err := m.Run(context.Background(), map[string]*Tensor{"input_ids": ids, "attention_mask": mask})
	require.NoError(t, err)
	assert.Equal(t, got, out2["pooled"].Data.([]float32))
	// Changing only the masked token does not change the pooled row.
	ids2 := mustTensor(NewTensor([]int64{batch, seq}, Int64, []int64{0, 1, 7, 3, 4, 5}))
	out3, err := m.Run(context.Background(), map[string]*Tensor{"input_ids": ids2, "attention_mask": mask})
	require.NoError(t, err)
	got3 := out3["pooled"].Data.([]float32)
	for h := 0; h < hidden; h++ {
		assert.InDelta(t, float64(got[h]), float64(got3[h]), 1e-6)
	}
	// Different real tokens do change the pooled row.
	ids3 := mustTensor(NewTensor([]int64{batch, seq}, Int64, []int64{0, 1, 2, 4, 4, 4}))
	out4, err := m.Run(context.Background(), map[string]*Tensor{"input_ids": ids3, "attention_mask": mask})
	require.NoError(t, err)
	differs := false
	for h := 0; h < hidden; h++ {
		// Batch 0 is unchanged; the token edits are in batch 1's row.
		if out4["pooled"].Data.([]float32)[hidden+h] != got[hidden+h] {
			differs = true
		}
	}
	assert.True(t, differs)
}

// __TEST_MODELS__
