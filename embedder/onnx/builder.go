package onnx

import (
	"encoding/binary"
	"math"
)

// This file provides a minimal ONNX wire-format encoder. It is useful for
// building tiny synthetic models in tests (and anywhere a model needs to be
// produced programmatically); it encodes exactly the fields that Load
// decodes.

// NodeSpec describes one node for Encode.
type NodeSpec struct {
	Op      string
	Inputs  []string
	Outputs []string
	// Scalar and list attributes.
	IntAttrs     map[string]int64
	FloatAttrs   map[string]float32
	StringAttrs  map[string]string
	IntListAttrs map[string][]int64
	// TensorAttrs carries tensor-valued attributes (e.g. Constant "value").
	TensorAttrs map[string]*Tensor
}

// Encode serializes a minimal ModelProto (IR version 8, ai.onnx opset 13)
// with the given nodes, initializers, and graph boundary tensors.
func Encode(nodes []NodeSpec, inits map[string]*Tensor, inputs, outputs []NamedType) []byte {
	var g wireWriter
	g.str(2, "recall-encode")
	for _, spec := range nodes {
		g.msg(1, func(g *wireWriter) { // NodeProto
			for _, in := range spec.Inputs {
				g.str(1, in)
			}
			for _, out := range spec.Outputs {
				g.str(2, out)
			}
			g.str(4, spec.Op)
			for k, v := range spec.IntAttrs {
				g.msg(5, func(g *wireWriter) {
					g.str(1, k)
					g.int64Field(3, v)
				})
			}
			for k, v := range spec.FloatAttrs {
				g.msg(5, func(g *wireWriter) {
					g.str(1, k)
					var fb [4]byte
					binary.LittleEndian.PutUint32(fb[:], float32bits(v))
					g.bytesField(2, fb[:])
				})
			}
			for k, v := range spec.StringAttrs {
				g.msg(5, func(g *wireWriter) {
					g.str(1, k)
					g.bytesField(4, []byte(v))
				})
			}
			for k, v := range spec.IntListAttrs {
				g.msg(5, func(g *wireWriter) {
					g.str(1, k)
					g.packedInts(8, v)
				})
			}
			for k, v := range spec.TensorAttrs {
				g.msg(5, func(g *wireWriter) {
					g.str(1, k)
					g.msg(5, func(g *wireWriter) {
						encodeTensor(g, "", v)
					})
				})
			}
		})
	}
	for name, t := range inits {
		g.msg(5, func(g *wireWriter) { // TensorProto
			encodeTensor(g, name, t)
		})
	}
	for _, in := range inputs {
		g.msg(11, func(g *wireWriter) { // ValueInfoProto
			g.str(1, in.Name)
			g.msg(2, func(g *wireWriter) { // TypeProto
				g.msg(1, func(g *wireWriter) { // Tensor
					g.int64Field(1, int64(in.Dtype))
				})
			})
		})
	}
	for _, out := range outputs {
		g.msg(12, func(g *wireWriter) {
			g.str(1, out.Name)
			g.msg(2, func(g *wireWriter) {
				g.msg(1, func(g *wireWriter) {
					g.int64Field(1, int64(out.Dtype))
				})
			})
		})
	}
	var m wireWriter
	m.int64Field(1, 8)             // ir_version
	m.str(2, "recall")             // producer_name
	m.msg(8, func(m *wireWriter) { // opset_import
		m.str(1, "ai.onnx")
		m.int64Field(2, 13)
	})
	m.bytesField(7, g.b) // graph
	return m.b
}

func encodeTensor(g *wireWriter, name string, t *Tensor) {
	g.packedInts(1, t.Shape)
	g.int64Field(2, int64(t.Dtype))
	g.str(8, name)
	var raw []byte
	switch v := t.Data.(type) {
	case []float32:
		raw = make([]byte, len(v)*4)
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[i*4:], float32bits(x))
		}
	case []float64:
		raw = make([]byte, len(v)*8)
		for i, x := range v {
			binary.LittleEndian.PutUint64(raw[i*8:], float64bits(x))
		}
	case []int32:
		raw = make([]byte, len(v)*4)
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[i*4:], uint32(x))
		}
	case []int64:
		raw = make([]byte, len(v)*8)
		for i, x := range v {
			binary.LittleEndian.PutUint64(raw[i*8:], uint64(x))
		}
	case []bool:
		raw = make([]byte, len(v))
		for i, x := range v {
			if x {
				raw[i] = 1
			}
		}
	case []uint8:
		raw = make([]byte, len(v))
		copy(raw, v)
	default:
		panic("onnx: Encode: unsupported tensor dtype")
	}
	g.bytesField(9, raw)
}

// wireWriter accumulates protobuf wire-format bytes.
type wireWriter struct {
	b []byte
}

func (w *wireWriter) varint(v uint64) {
	for v >= 0x80 {
		w.b = append(w.b, byte(v)|0x80)
		v >>= 7
	}
	w.b = append(w.b, byte(v))
}

func (w *wireWriter) tag(field, wt int) {
	w.varint(uint64(field)<<3 | uint64(wt))
}

func (w *wireWriter) int64Field(field int, v int64) {
	w.tag(field, 0)
	w.varint(uint64(v))
}

func (w *wireWriter) str(field int, s string) {
	w.tag(field, 2)
	w.varint(uint64(len(s)))
	w.b = append(w.b, s...)
}

func (w *wireWriter) bytesField(field int, b []byte) {
	w.tag(field, 2)
	w.varint(uint64(len(b)))
	w.b = append(w.b, b...)
}

func (w *wireWriter) msg(field int, f func(*wireWriter)) {
	var inner wireWriter
	f(&inner)
	w.tag(field, 2)
	w.varint(uint64(len(inner.b)))
	w.b = append(w.b, inner.b...)
}

func (w *wireWriter) packedInts(field int, vs []int64) {
	if len(vs) == 0 {
		return
	}
	var run []byte
	for _, v := range vs {
		run = append(run, appendVarint(uint64(v))...)
	}
	w.bytesField(field, run)
}

func appendVarint(v uint64) []byte {
	var out []byte
	for v >= 0x80 {
		out = append(out, byte(v)|0x80)
		v >>= 7
	}
	return append(out, byte(v))
}

func float32bits(f float32) uint32 { return math.Float32bits(f) }

func float64bits(f float64) uint64 { return math.Float64bits(f) }
