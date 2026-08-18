package onnx

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLoadFileRoundTrip exercises LoadFile, FeedInputs, and Outputs using a
// Double-dtype model written to a temporary path.
func TestLoadFileRoundTrip(t *testing.T) {
	nodes := []NodeSpec{
		{Op: "Identity", Inputs: []string{"x"}, Outputs: []string{"y"}},
		{Op: "Add", Inputs: []string{"y", "offset"}, Outputs: []string{"z"}},
	}
	offset, err := NewTensor([]int64{2}, Double, []float64{10, 20})
	require.NoError(t, err)
	b := Encode(nodes, map[string]*Tensor{"offset": offset},
		[]NamedType{{Name: "x", Dtype: Double}},
		[]NamedType{{Name: "z", Dtype: Double}})

	path := filepath.Join(t.TempDir(), "model.onnx")
	require.NoError(t, os.WriteFile(path, b, 0o644))
	m, err := LoadFile(path)
	require.NoError(t, err)

	fi := m.FeedInputs()
	require.Len(t, fi, 1)
	assert.Equal(t, "x", fi[0].Name)
	assert.Equal(t, Double, fi[0].Dtype)
	outs := m.Outputs()
	require.Len(t, outs, 1)
	assert.Equal(t, "z", outs[0].Name)

	in, err := NewTensor([]int64{2}, Double, []float64{1, 2})
	require.NoError(t, err)
	got, err := m.Run(context.Background(), map[string]*Tensor{"x": in})
	require.NoError(t, err)
	assert.Equal(t, []float64{11, 22}, got["z"].Data.([]float64))

	_, err = LoadFile(filepath.Join(t.TempDir(), "missing.onnx"))
	require.Error(t, err)
}

// TestDropoutInferenceMode verifies Dropout is identity at inference time,
// emits an all-true mask, and rejects training mode.
func TestDropoutInferenceMode(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Dropout", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3)}, nil)
	assert.Equal(t, []float32{1, 2, 3}, out.Data.([]float32))

	// Two-output form: the inference mask is all-true.
	m, err := Load(Encode([]NodeSpec{
		{Op: "Dropout", Inputs: []string{"a", "ratio"}, Outputs: []string{"y", "mask"}},
	}, nil,
		namedFeeds(map[string]*Tensor{"a": ft(1, 2), "ratio": ft(0.5)}),
		[]NamedType{{Name: "y", Dtype: Float32}, {Name: "mask", Dtype: Bool}}))
	require.NoError(t, err)
	got, err := m.Run(context.Background(), map[string]*Tensor{"a": mustTensor(NewTensor([]int64{2}, Float32, []float32{1, 2})), "ratio": ft(0.5)})
	require.NoError(t, err)
	assert.Equal(t, []bool{true, true}, got["mask"].Data.([]bool))

	// training_mode attribute forces the training path error.
	_, err = runNodeErr(t, NodeSpec{
		Op:       "Dropout",
		Inputs:   []string{"a"},
		Outputs:  []string{"y"},
		IntAttrs: map[string]int64{"training_mode": 1},
	}, map[string]*Tensor{"a": ft(1, 2)}, nil)
	require.Error(t, err)

	// Explicit train flag as third input.
	_, err = runNodeErr(t, NodeSpec{Op: "Dropout", Inputs: []string{"a", "ratio", "train"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2), "ratio": ft(0.5), "train": it64(1)}, nil)
	require.Error(t, err)
}

// TestWireReaderUnits covers varint/length/skip edge cases and the
// not-packed 64-bit double path.
func TestWireReaderUnits(t *testing.T) {
	// packedFloats64 with a length-delimited payload.
	dbl := make([]byte, 16)
	binary.LittleEndian.PutUint64(dbl[0:8], float64bits(1.5))
	binary.LittleEndian.PutUint64(dbl[8:16], float64bits(-2.5))
	w := &wireReader{b: append([]byte{0x10}, dbl...)}
	vs, err := packedFloats64(w, 2)
	require.NoError(t, err)
	assert.Equal(t, []float64{1.5, -2.5}, vs)

	// packedFloats64 with a raw 64-bit wire type.
	w = &wireReader{b: dbl[:8]}
	vs, err = packedFloats64(w, 1)
	require.NoError(t, err)
	assert.Equal(t, []float64{1.5}, vs)
	assert.True(t, w.done())

	// mathFloat64FromLE round-trip.
	assert.Equal(t, 3.25, mathFloat64FromLE(dbl8(3.25)))

	// mathFloat32FromLE round-trip.
	assert.Equal(t, float32(-7.5), mathFloat32FromLE([]byte{0x00, 0x00, 0xF0, 0xC0}))

	// Multi-byte varint.
	w = &wireReader{b: []byte{0xAC, 0x02}}
	v, err := w.uvarint()
	require.NoError(t, err)
	assert.Equal(t, uint64(300), v)

	// Truncated varint.
	_, err = (&wireReader{b: []byte{0x80}}).uvarint()
	require.Error(t, err)

	// Truncated length-delimited field.
	_, err = (&wireReader{b: []byte{0xFF}}).lenBytes()
	require.Error(t, err)

	// skip over each wire type.
	w = &wireReader{b: []byte{0xAC, 0x02}}
	require.NoError(t, w.skip(0))
	assert.True(t, w.done())
	w = &wireReader{b: append([]byte{}, dbl...)}
	require.NoError(t, w.skip(1))
	assert.False(t, w.done())
	require.NoError(t, w.skip(1))
	assert.True(t, w.done())
	w = &wireReader{b: dbl[:4]}
	require.NoError(t, w.skip(5))
	assert.True(t, w.done())
	w = &wireReader{b: append([]byte{0x05}, dbl[:5]...)}
	require.NoError(t, w.skip(2))
	assert.True(t, w.done())
	err = (&wireReader{b: []byte{0x01}}).skip(6)
	require.Error(t, err)
	err = (&wireReader{b: []byte{0x00}}).skip(1)
	require.Error(t, err)
	err = (&wireReader{b: []byte{0x00}}).skip(5)
	require.Error(t, err)
}

func dbl8(v float64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, float64bits(v))
	return b
}

// TestPackedIntsNegative verifies two's-complement varint round-trips for
// negative int64 attribute values.
func TestPackedIntsNegative(t *testing.T) {
	g := &wireWriter{}
	g.packedInts(8, []int64{-1, -300, 42})
	w := &wireReader{b: g.b}
	f, wt, err := w.tag()
	require.NoError(t, err)
	assert.Equal(t, 8, f)
	assert.Equal(t, 2, wt)
	vs, err := packedInts(w, wt)
	require.NoError(t, err)
	assert.Equal(t, []int64{-1, -300, 42}, vs)
}

// TestTensorHelpers covers String, NewTensor validation, conversions, and
// l2Normalize.
func TestTensorHelpers(t *testing.T) {
	t1 := mustTensor(NewTensor([]int64{2}, Float32, []float32{1, 2}))
	assert.Equal(t, "float32 tensor [2]", t1.String())
	assert.Equal(t, int64(2), t1.Size())
	assert.Equal(t, 1, t1.Nd())

	_, err := NewTensor([]int64{1}, Float32, "nope")
	require.Error(t, err)
	_, err = NewTensor([]int64{2}, Float32, []float32{1})
	require.Error(t, err)

	// Conversions from every dtype.
	f, err := mustTensor(NewTensor([]int64{2}, Bool, []bool{true, false})).AsFloat64()
	require.NoError(t, err)
	assert.Equal(t, []float64{1, 0}, f)
	i, err := mustTensor(NewTensor([]int64{2}, Uint8, []uint8{250, 1})).AsInt64()
	require.NoError(t, err)
	assert.Equal(t, []int64{250, 1}, i)
	b, err := mustTensor(NewTensor([]int64{2}, Int32, []int32{7, 0})).AsBool()
	require.NoError(t, err)
	assert.Equal(t, []bool{true, false}, b)

	// l2Normalize scales to unit length; zero vector is a no-op.
	v := []float64{3, 4}
	l2Normalize(v)
	assert.InDelta(t, 0.6, v[0], 1e-12)
	assert.InDelta(t, 0.8, v[1], 1e-12)
	z := []float64{0, 0}
	l2Normalize(z)
	assert.Equal(t, []float64{0, 0}, z)
}

// TestAttributeGetters covers the Int/Float/IntList accessors including the
// double and list variants.
func TestAttributeGetters(t *testing.T) {
	a := Attribute{Name: "x", I: 5, G: 2.5, Ints: []int64{1, 2}}
	assert.Equal(t, int64(5), a.Int())
	assert.Equal(t, 2.5, a.Float())
	assert.Equal(t, []int64{1, 2}, a.IntList())
	// Float falls back to f when g is zero.
	b := Attribute{Name: "y", F: 1.5}
	assert.Equal(t, 1.5, b.Float())
}

// TestConstantDtypes exercises Constant with double and int32 tensor
// attributes.
func TestConstantDtypes(t *testing.T) {
	dbl, err := NewTensor([]int64{2}, Double, []float64{1.5, -0.5})
	require.NoError(t, err)
	out := runNode(t, NodeSpec{Op: "Constant", Outputs: []string{"y"}, TensorAttrs: map[string]*Tensor{"value": dbl}}, nil, nil)
	assert.Equal(t, []float64{1.5, -0.5}, out.Data.([]float64))

	i32, err := NewTensor([]int64{2}, Int32, []int32{-7, 9})
	require.NoError(t, err)
	out = runNode(t, NodeSpec{Op: "Constant", Outputs: []string{"y"}, TensorAttrs: map[string]*Tensor{"value": i32}}, nil, nil)
	assert.Equal(t, []int32{-7, 9}, out.Data.([]int32))
}

// TestCastVariants exercises Cast across the supported dtype conversions and
// its error path.
func TestCastVariants(t *testing.T) {
	run := func(to DataType, tn *Tensor) *Tensor {
		return runNode(t, NodeSpec{
			Op:       "Cast",
			Inputs:   []string{"a"},
			Outputs:  []string{"y"},
			IntAttrs: map[string]int64{"to": int64(to)},
		}, map[string]*Tensor{"a": tn}, nil)
	}
	assert.Equal(t, []float64{1.5, -2.5},
		run(Double, mustTensor(NewTensor([]int64{2}, Float32, []float32{1.5, -2.5}))).Data.([]float64))
	assert.Equal(t, []float32{1, 4},
		run(Float32, it64(1, 4)).Data.([]float32))
	assert.Equal(t, []bool{false, true},
		run(Bool, mustTensor(NewTensor([]int64{2}, Float32, []float32{0, 1.5}))).Data.([]bool))
	assert.Equal(t, []int32{1, -4},
		run(Int32, it64(1, -4)).Data.([]int32))
	assert.Equal(t, []uint8{1, 2},
		run(Uint8, mustTensor(NewTensor([]int64{2}, Float32, []float32{1, 2}))).Data.([]uint8))

	// Invalid target dtype.
	_, err := runNodeErr(t, NodeSpec{
		Op: "Cast", Inputs: []string{"a"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"to": 99},
	}, map[string]*Tensor{"a": ft(1)}, nil)
	require.Error(t, err)
}

// TestExpandVariants covers rank growth and incompatible-shape errors.
func TestExpandVariants(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Expand", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft2(2, 1, 7, 8), "shape": it64(2, 3)}, nil)
	assert.Equal(t, []float32{7, 7, 7, 8, 8, 8}, out.Data.([]float32))

	_, err := runNodeErr(t, NodeSpec{Op: "Expand", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2), "shape": it64(3, 4)}, nil)
	require.Error(t, err)
}

// TestConcatDtypes covers the non-float concat branches.
func TestConcatDtypes(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Concat", Inputs: []string{"a", "b"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"axis": 1}},
		map[string]*Tensor{"a": mustTensor(NewTensor([]int64{2, 1}, Int64, []int64{1, 2})),
			"b": mustTensor(NewTensor([]int64{2, 1}, Int64, []int64{3, 4}))}, nil)
	assert.Equal(t, []int64{1, 3, 2, 4}, out.Data.([]int64))

	out = runNode(t, NodeSpec{Op: "Concat", Inputs: []string{"a", "b"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"axis": 0}},
		map[string]*Tensor{
			"a": mustTensor(NewTensor([]int64{2}, Bool, []bool{true, false})),
			"b": mustTensor(NewTensor([]int64{1}, Bool, []bool{true}))}, nil)
	assert.Equal(t, []bool{true, false, true}, out.Data.([]bool))

	// Rank mismatch is rejected.
	_, err := runNodeErr(t, NodeSpec{Op: "Concat", Inputs: []string{"a", "b"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"axis": 0}},
		map[string]*Tensor{"a": ft(1), "b": ft2(1, 1, 2)}, nil)
	require.Error(t, err)
}

// TestSplitVariants covers explicit split sizes and the error path.
func TestSplitVariants(t *testing.T) {
	out := runGraph(t, []NodeSpec{
		{Op: "Split", Inputs: []string{"a", "s"}, Outputs: []string{"x", "y"}},
	}, nil, map[string]*Tensor{"a": it64(1, 2, 3, 4), "s": it64(2, 2)})
	assert.Equal(t, []int64{1, 2}, out["x"].Data.([]int64))
	assert.Equal(t, []int64{3, 4}, out["y"].Data.([]int64))

	// Split sizes that do not sum to the axis length.
	_, err := runNodeErr(t, NodeSpec{Op: "Split", Inputs: []string{"a", "s"}, Outputs: []string{"x", "y"}},
		map[string]*Tensor{"a": it64(1, 2, 3), "s": it64(1, 3)}, nil)
	require.Error(t, err)
}

// TestTileRankError verifies repeats longer than the data rank is rejected.
func TestTileRankError(t *testing.T) {
	_, err := runNodeErr(t, NodeSpec{Op: "Tile", Inputs: []string{"a", "r"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1), "r": it64(2, 3)}, nil)
	require.Error(t, err)
}

// TestReshapeInfer covers the -1 inferred dimension and mismatch errors.
func TestReshapeInfer(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Reshape", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3, 4, 5, 6), "shape": it64(3, -1)}, nil)
	assert.Equal(t, []int64{3, 2}, out.Shape)

	_, err := runNodeErr(t, NodeSpec{Op: "Reshape", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(1, 2, 3, 4), "shape": it64(3, 3)}, nil)
	require.Error(t, err)
}

// TestSqueezeMultiAxis covers multiple axes including a negative one.
func TestSqueezeMultiAxis(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Squeeze", Inputs: []string{"a", "ax"}, Outputs: []string{"y"}},
		map[string]*Tensor{
			"a":  mustTensor(NewTensor([]int64{1, 3, 1, 2}, Float32, make([]float32, 6))),
			"ax": it64(0, -2)}, nil)
	assert.Equal(t, []int64{3, 2}, out.Shape)
}

// TestNonFloatInitializersRoundTrip runs int32, bool, and uint8 initializers
// through Encode/Load to cover parseTensor's remaining raw dtypes.
func TestNonFloatInitializersRoundTrip(t *testing.T) {
	out := runGraph(t, []NodeSpec{
		{Op: "Add", Inputs: []string{"a", "i32"}, Outputs: []string{"y1"}},
		{Op: "Cast", Inputs: []string{"u8"}, Outputs: []string{"y2"}, IntAttrs: map[string]int64{"to": int64(Int64)}},
		{Op: "Cast", Inputs: []string{"bl"}, Outputs: []string{"y3"}, IntAttrs: map[string]int64{"to": int64(Float32)}},
	}, map[string]*Tensor{
		"i32": mustTensor(NewTensor([]int64{2}, Int32, []int32{10, 20})),
		"u8":  mustTensor(NewTensor([]int64{2}, Uint8, []uint8{200, 250})),
		"bl":  mustTensor(NewTensor([]int64{2}, Bool, []bool{true, false})),
	}, map[string]*Tensor{"a": mustTensor(NewTensor([]int64{2}, Int32, []int32{1, 2}))})
	sums, err := out["y1"].AsInt64()
	require.NoError(t, err)
	assert.Equal(t, []int64{11, 22}, sums)
	assert.Equal(t, []int64{200, 250}, out["y2"].Data.([]int64))
	assert.Equal(t, []float32{1, 0}, out["y3"].Data.([]float32))
}

// TestPackedFloats32Raw covers the not-packed 32-bit wire type.
func TestPackedFloats32Raw(t *testing.T) {
	w := &wireReader{b: []byte{0x00, 0x00, 0x00, 0x40}}
	vs, err := packedFloats32(w, 5)
	require.NoError(t, err)
	assert.Equal(t, []float32{2}, vs)
	_, err = packedFloats32(&wireReader{b: []byte{0x01}}, 5)
	require.Error(t, err)
}

// TestConversionErrors covers AsFloat64/AsInt64/AsBool on tensors whose data
// does not match any supported slice type, plus remaining AsInt64 paths.
func TestConversionErrors(t *testing.T) {
	bad := &Tensor{Shape: []int64{1}, Dtype: Float32, Data: 42}
	_, err := bad.AsFloat64()
	require.Error(t, err)
	_, err = bad.AsInt64()
	require.Error(t, err)
	_, err = bad.AsBool()
	require.Error(t, err)
	f, err := mustTensor(NewTensor([]int64{2}, Double, []float64{1.9, 2.1})).AsInt64()
	require.NoError(t, err)
	assert.Equal(t, []int64{1, 2}, f)
	i, err := mustTensor(NewTensor([]int64{1}, Int32, []int32{-5})).AsInt64()
	require.NoError(t, err)
	assert.Equal(t, []int64{-5}, i)
}

// TestDataTypeStrings covers every DataType String() arm.
func TestDataTypeStrings(t *testing.T) {
	cases := map[DataType]string{
		Float32: "float32", Uint8: "uint8", Int8: "int8", Int32: "int32",
		Int64: "int64", Bool: "bool", Double: "float64",
		DataType(99): "onnx-type-99",
	}
	for dt, want := range cases {
		assert.Equal(t, want, dt.String())
	}
}

// TestConstantScalarAttrs covers the value_int / value_float attribute forms
// and their error paths.
func TestConstantScalarAttrs(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Constant", Outputs: []string{"y"},
		IntAttrs: map[string]int64{"value_int": 42}}, nil, nil)
	assert.Equal(t, []int64{42}, out.Data.([]int64))

	out = runNode(t, NodeSpec{Op: "Constant", Outputs: []string{"y"},
		FloatAttrs: map[string]float32{"value_float": 1.25}}, nil, nil)
	assert.Equal(t, []float32{1.25}, out.Data.([]float32))

	_, err := runNodeErr(t, NodeSpec{Op: "Constant", Outputs: []string{"y"}}, nil, nil)
	require.Error(t, err)
	_, err = runNodeErr(t, NodeSpec{Op: "Constant", Outputs: []string{"y"},
		StringAttrs: map[string]string{"value_string": "hi"}}, nil, nil)
	require.Error(t, err)
}

// TestTransposeVariants covers explicit perms, the default reversal, the
// int64 branch, and an invalid perm.
func TestTransposeVariants(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Transpose", Inputs: []string{"a"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": mustTensor(NewTensor([]int64{2, 2}, Int64, []int64{1, 2, 3, 4}))}, nil)
	assert.Equal(t, []int64{1, 3, 2, 4}, out.Data.([]int64))

	out = runNode(t, NodeSpec{Op: "Transpose", Inputs: []string{"a", "perm"}, Outputs: []string{"y"}},
		map[string]*Tensor{
			"a":    mustTensor(NewTensor([]int64{2, 3}, Float32, []float32{1, 2, 3, 4, 5, 6})),
			"perm": it64(1, 0)}, nil)
	assert.Equal(t, []float32{1, 4, 2, 5, 3, 6}, out.Data.([]float32))

	_, err := runNodeErr(t, NodeSpec{Op: "Transpose", Inputs: []string{"a", "perm"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": it64(1, 2), "perm": it64(0, 0)}, nil)
	require.Error(t, err)
}

// TestExpandDtypes covers int64 expansion and left-padding a shorter target
// shape.
func TestExpandDtypes(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Expand", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": it64(5), "shape": it64(3)}, nil)
	assert.Equal(t, []int64{5, 5, 5}, out.Data.([]int64))

	out = runNode(t, NodeSpec{Op: "Expand", Inputs: []string{"a", "shape"}, Outputs: []string{"y"}},
		map[string]*Tensor{"a": ft(9), "shape": it64(2, 1, 1)}, nil)
	assert.Equal(t, []int64{2, 1, 1}, out.Shape)
	assert.Equal(t, []float32{9, 9}, out.Data.([]float32))
}

// TestGatherVariants covers int64 data, scalar indices (rank reduction), and
// negative axis.
func TestGatherVariants(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Gather", Inputs: []string{"a", "idx"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"axis": -1}},
		map[string]*Tensor{
			"a":   it64(10, 20, 30, 40),
			"idx": it64(1, 3, 0)}, nil)
	assert.Equal(t, []int64{20, 40, 10}, out.Data.([]int64))

	out = runNode(t, NodeSpec{Op: "Gather", Inputs: []string{"a", "idx"}, Outputs: []string{"y"},
		IntAttrs: map[string]int64{"axis": 0}},
		map[string]*Tensor{
			"a":   ft2(3, 2, 1, 2, 3, 4, 5, 6),
			"idx": mustTensor(NewTensor([]int64{}, Int64, []int64{1}))}, nil)
	assert.Equal(t, []int64{2}, out.Shape)
	assert.Equal(t, []float32{3, 4}, out.Data.([]float32))
}

// TestRangeFloat covers a non-unit float step.
func TestRangeFloat(t *testing.T) {
	out := runNode(t, NodeSpec{Op: "Range", Inputs: []string{"s", "l", "d"}, Outputs: []string{"y"}},
		map[string]*Tensor{"s": ft(0), "l": ft(1), "d": ft(0.25)}, nil)
	assert.Equal(t, []float32{0, 0.25, 0.5, 0.75}, out.Data.([]float32))
}

// TestStringAttrRoundTrip verifies string attributes survive Encode/Load.
func TestStringAttrRoundTrip(t *testing.T) {
	m, err := Load(Encode([]NodeSpec{
		{Op: "Identity", Inputs: []string{"a"}, Outputs: []string{"y"},
			StringAttrs: map[string]string{"comment": "hello"}},
	}, nil, namedFeeds(map[string]*Tensor{"a": ft(1)}), namedOutputs([]string{"y"})))
	require.NoError(t, err)
	require.Len(t, m.Graph.Nodes, 1)
	attrs := m.Graph.Nodes[0].Attrs
	require.Len(t, attrs, 1)
	assert.Equal(t, "comment", attrs[0].Name)
	assert.Equal(t, "hello", attrs[0].S)
}
