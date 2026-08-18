// Package onnx is a minimal, pure-Go ONNX model loader and interpreter for
// inference-only use. It parses the ONNX protobuf wire format directly (no
// code-generated protobuf dependency) and executes the subset of operators
// commonly found in sentence-transformer (BERT-style) embedding model
// exports: elementwise math, broadcasting, matrix algebra, attention
// building blocks, normalization, pooling, and shape manipulation.
//
// The package is intentionally small and dependency-free (zero CGO).
// Operators outside the supported set return a descriptive error naming the
// missing operator.
package onnx

import (
	"fmt"
	"math"
)

// DataType mirrors the ONNX TensorProto.DataType enum values.
type DataType int32

// Supported ONNX tensor data types (enum values from onnx.proto3).
const (
	Float32 DataType = 1
	Uint8   DataType = 2
	Int8    DataType = 3
	Int32   DataType = 6
	Int64   DataType = 7
	Bool    DataType = 9
	Double  DataType = 11
)

// String returns a human-readable name for the data type.
func (d DataType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Uint8:
		return "uint8"
	case Int8:
		return "int8"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Bool:
		return "bool"
	case Double:
		return "float64"
	default:
		return fmt.Sprintf("onnx-type-%d", int32(d))
	}
}

// Tensor is a dense, row-major (C-order) tensor. Data holds a flat slice in
// one of: []float32, []float64, []int32, []int64, []bool, []uint8,
// matching Dtype.
type Tensor struct {
	Shape []int64
	Dtype DataType
	Data  any
}

// Size returns the total number of elements.
func (t *Tensor) Size() int64 {
	return shapeSize(t.Shape)
}

// Nd returns the number of dimensions.
func (t *Tensor) Nd() int {
	return len(t.Shape)
}

// String returns a short description of the tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("%s tensor %v", t.Dtype, t.Shape)
}

func shapeSize(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		if d < 0 {
			return 0
		}
		n *= d
	}
	return n
}

// NewTensor validates that the data slice matches the declared dtype and
// shape size, then returns the tensor.
func NewTensor(shape []int64, dt DataType, data any) (*Tensor, error) {
	var n int
	switch data.(type) {
	case []float32:
		n = len(data.([]float32))
	case []float64:
		n = len(data.([]float64))
	case []int32:
		n = len(data.([]int32))
	case []int64:
		n = len(data.([]int64))
	case []bool:
		n = len(data.([]bool))
	case []uint8:
		n = len(data.([]uint8))
	default:
		return nil, fmt.Errorf("onnx: unsupported tensor data type %T", data)
	}
	if int64(n) != shapeSize(shape) {
		return nil, fmt.Errorf("onnx: tensor data has %d elements but shape %v requires %d", n, shape, shapeSize(shape))
	}
	return &Tensor{Shape: shape, Dtype: dt, Data: data}, nil
}

// AsFloat64 returns the tensor's elements as []float64, converting numeric
// and boolean tensors as needed. The returned slice is a copy.
func (t *Tensor) AsFloat64() ([]float64, error) {
	switch v := t.Data.(type) {
	case []float32:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out, nil
	case []float64:
		out := make([]float64, len(v))
		copy(out, v)
		return out, nil
	case []int32:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out, nil
	case []int64:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out, nil
	case []bool:
		out := make([]float64, len(v))
		for i, x := range v {
			if x {
				out[i] = 1
			}
		}
		return out, nil
	case []uint8:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("onnx: cannot read %s tensor as float64", t.Dtype)
	}
}

// AsInt64 returns the tensor's elements as []int64.
func (t *Tensor) AsInt64() ([]int64, error) {
	switch v := t.Data.(type) {
	case []int32:
		out := make([]int64, len(v))
		for i, x := range v {
			out[i] = int64(x)
		}
		return out, nil
	case []int64:
		out := make([]int64, len(v))
		copy(out, v)
		return out, nil
	case []float32:
		out := make([]int64, len(v))
		for i, x := range v {
			out[i] = int64(x)
		}
		return out, nil
	case []float64:
		out := make([]int64, len(v))
		for i, x := range v {
			out[i] = int64(x)
		}
		return out, nil
	case []bool:
		out := make([]int64, len(v))
		for i, x := range v {
			if x {
				out[i] = 1
			}
		}
		return out, nil
	case []uint8:
		out := make([]int64, len(v))
		for i, x := range v {
			out[i] = int64(x)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("onnx: cannot read %s tensor as int64", t.Dtype)
	}
}

// AsBool returns the tensor's elements as []bool (non-zero means true).
func (t *Tensor) AsBool() ([]bool, error) {
	switch v := t.Data.(type) {
	case []bool:
		out := make([]bool, len(v))
		copy(out, v)
		return out, nil
	case []int32:
		out := make([]bool, len(v))
		for i, x := range v {
			out[i] = x != 0
		}
		return out, nil
	case []int64:
		out := make([]bool, len(v))
		for i, x := range v {
			out[i] = x != 0
		}
		return out, nil
	case []float32:
		out := make([]bool, len(v))
		for i, x := range v {
			out[i] = x != 0
		}
		return out, nil
	case []float64:
		out := make([]bool, len(v))
		for i, x := range v {
			out[i] = x != 0
		}
		return out, nil
	default:
		return nil, fmt.Errorf("onnx: cannot read %s tensor as bool", t.Dtype)
	}
}

// makeFloat builds a float tensor in the requested dtype from float64
// values.
func makeFloat(shape []int64, dt DataType, vals []float64) (*Tensor, error) {
	switch dt {
	case Float32:
		out := make([]float32, len(vals))
		for i, x := range vals {
			out[i] = float32(x)
		}
		return NewTensor(shape, Float32, out)
	case Double:
		out := make([]float64, len(vals))
		copy(out, vals)
		return NewTensor(shape, Double, out)
	default:
		return nil, fmt.Errorf("onnx: cannot create %s float tensor", dt)
	}
}

// pickFloatDtype chooses the output dtype for float results: float32 when
// every operand is float32, float64 otherwise.
func pickFloatDtype(ts ...*Tensor) DataType {
	for _, t := range ts {
		if t.Dtype != Float32 {
			return Double
		}
	}
	return Float32
}

// broadcastShapes computes the elementwise broadcast result of two shapes
// (right-aligned, dimension 1 stretches).
func broadcastShapes(a, b []int64) ([]int64, error) {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]int64, n)
	for i := 0; i < n; i++ {
		da, db := int64(1), int64(1)
		if i >= n-len(a) {
			da = a[i-(n-len(a))]
		}
		if i >= n-len(b) {
			db = b[i-(n-len(b))]
		}
		if da == db || da == 1 {
			out[i] = db
		} else if db == 1 {
			out[i] = da
		} else {
			return nil, fmt.Errorf("onnx: cannot broadcast shapes %v and %v", a, b)
		}
	}
	return out, nil
}

// broadcastValues stretches a flat value slice of shape `from` to shape
// `to` (which must be a valid broadcast of `from`).
func broadcastValues(vals []float64, from, to []int64) []float64 {
	if int64(len(vals)) == shapeSize(to) && sameShape(from, to) {
		return vals
	}
	out := make([]float64, shapeSize(to))
	fromStride := make([]int64, len(from))
	for i := len(from) - 1; i >= 0; i-- {
		fromStride[i] = 1
		if i < len(from)-1 {
			fromStride[i] = fromStride[i+1] * from[i+1]
		}
	}
	offTo := make([]int64, len(to))
	offset := 0
	for offset < int(shapeSize(to)) {
		fromPos := 0
		for i := len(to) - 1; i >= 0; i-- {
			fi := i - (len(to) - len(from))
			if fi < 0 {
				break
			}
			if from[fi] != 1 {
				fromPos += int(offTo[i]) * int(fromStride[fi])
			}
		}
		out[offset] = vals[fromPos]
		for i := len(to) - 1; i >= 0; i-- {
			offTo[i]++
			if offTo[i] < to[i] {
				break
			}
			offTo[i] = 0
		}
		offset++
	}
	return out
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// l2Normalize normalizes v to unit length in place (no-op when the norm is
// zero).
func l2Normalize(v []float64) {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	norm := 1 / math.Sqrt(sum)
	for i := range v {
		v[i] *= norm
	}
}
