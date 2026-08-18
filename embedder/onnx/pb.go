package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// This file implements a minimal reader for the ONNX protobuf wire format
// (onnx.proto3). Only the fields required for inference are decoded; all
// others are skipped. This keeps the package free of code-generated
// protobuf dependencies.

// wireReader iterates over protobuf-encoded fields in a byte slice.
type wireReader struct {
	b []byte
	i int
}

// done reports whether all bytes have been consumed.
func (w *wireReader) done() bool { return w.i >= len(w.b) }

// uvarint reads a base-128 unsigned varint.
func (w *wireReader) uvarint() (uint64, error) {
	var v uint64
	for shift := uint(0); ; shift += 7 {
		if w.i >= len(w.b) {
			return 0, fmt.Errorf("onnx: truncated varint")
		}
		c := w.b[w.i]
		w.i++
		v |= uint64(c&0x7f) << shift
		if c&0x80 == 0 {
			return v, nil
		}
		if shift >= 63 {
			return 0, fmt.Errorf("onnx: varint too long")
		}
	}
}

// tag reads the next field tag, returning the field number and wire type.
func (w *wireReader) tag() (field int, wt int, err error) {
	v, err := w.uvarint()
	if err != nil {
		return 0, 0, err
	}
	return int(v >> 3), int(v & 7), nil
}

// lenBytes reads a length-delimited field payload.
func (w *wireReader) lenBytes() ([]byte, error) {
	n, err := w.uvarint()
	if err != nil {
		return nil, err
	}
	if n > uint64(len(w.b)-w.i) {
		return nil, fmt.Errorf("onnx: truncated length-delimited field (%d bytes, %d remaining)", n, len(w.b)-w.i)
	}
	out := w.b[w.i : w.i+int(n)]
	w.i += int(n)
	return out, nil
}

// skip advances past the value of the current field.
func (w *wireReader) skip(wt int) error {
	switch wt {
	case 0: // varint
		_, err := w.uvarint()
		return err
	case 1: // 64-bit
		if len(w.b)-w.i < 8 {
			return fmt.Errorf("onnx: truncated 64-bit field")
		}
		w.i += 8
		return nil
	case 2: // length-delimited
		_, err := w.lenBytes()
		return err
	case 5: // 32-bit
		if len(w.b)-w.i < 4 {
			return fmt.Errorf("onnx: truncated 32-bit field")
		}
		w.i += 4
		return nil
	default:
		return fmt.Errorf("onnx: unsupported wire type %d", wt)
	}
}

// eachField invokes fn for every field in b. fn receives a reader positioned
// at the field value (after the tag); the reader must consume the value
// (directly or via skip).
func eachField(b []byte, fn func(field, wt int, w *wireReader) error) error {
	w := &wireReader{b: b}
	for !w.done() {
		f, wt, err := w.tag()
		if err != nil {
			return err
		}
		if err := fn(f, wt, w); err != nil {
			return err
		}
	}
	return nil
}

// packedInts reads a (possibly packed) repeated int64 field: wire type 0
// means a single varint value, wire type 2 a packed byte run.
func packedInts(w *wireReader, wt int) ([]int64, error) {
	if wt == 2 {
		b, err := w.lenBytes()
		if err != nil {
			return nil, err
		}
		pw := &wireReader{b: b}
		var out []int64
		for !pw.done() {
			v, err := pw.uvarint()
			if err != nil {
				return nil, err
			}
			out = append(out, int64(v))
		}
		return out, nil
	}
	v, err := w.uvarint()
	return []int64{int64(v)}, err
}

// packedFloats32 reads a (possibly packed) repeated float field.
func packedFloats32(w *wireReader, wt int) ([]float32, error) {
	if wt == 2 {
		b, err := w.lenBytes()
		if err != nil {
			return nil, err
		}
		if len(b)%4 != 0 {
			return nil, fmt.Errorf("onnx: truncated packed float run")
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = mathFloat32FromLE(b[i*4:])
		}
		return out, nil
	}
	if len(w.b)-w.i < 4 {
		return nil, fmt.Errorf("onnx: truncated float field")
	}
	out := []float32{mathFloat32FromLE(w.b[w.i:])}
	w.i += 4
	return out, nil
}

// packedFloats64 reads a (possibly packed) repeated double field.
func packedFloats64(w *wireReader, wt int) ([]float64, error) {
	if wt == 2 {
		b, err := w.lenBytes()
		if err != nil {
			return nil, err
		}
		if len(b)%8 != 0 {
			return nil, fmt.Errorf("onnx: truncated packed double run")
		}
		out := make([]float64, len(b)/8)
		for i := range out {
			out[i] = mathFloat64FromLE(b[i*8:])
		}
		return out, nil
	}
	if len(w.b)-w.i < 8 {
		return nil, fmt.Errorf("onnx: truncated double field")
	}
	out := []float64{mathFloat64FromLE(w.b[w.i:])}
	w.i += 8
	return out, nil
}

func mathFloat32FromLE(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

func mathFloat64FromLE(b []byte) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(b))
}

// Model is the parsed ONNX model (inference subset).
type Model struct {
	IRVersion uint32
	Opsets    []Opset
	Graph     Graph
}

// Opset records one opset import (domain and version).
type Opset struct {
	Domain  string
	Version int64
}

// Graph is a parsed ONNX GraphProto.
type Graph struct {
	Name         string
	Nodes        []Node
	Initializers map[string]*Tensor
	// Inputs and Outputs are the declared graph boundary tensors. Inputs
	// may include names that are also initializers; FeedInputs returns the
	// subset that must be supplied at run time.
	Inputs  []NamedType
	Outputs []NamedType
}

// FeedInputs returns the graph inputs that are not satisfied by
// initializers (the tensors a caller must provide to Run).
func (g *Graph) FeedInputs() []NamedType {
	var out []NamedType
	for _, in := range g.Inputs {
		if _, ok := g.Initializers[in.Name]; !ok {
			out = append(out, in)
		}
	}
	return out
}

// NamedType names a graph boundary tensor with its declared element type.
type NamedType struct {
	Name  string
	Dtype DataType
}

// Node is a parsed ONNX NodeProto.
type Node struct {
	Inputs  []string
	Outputs []string
	Name    string
	OpType  string
	Domain  string
	Attrs   []Attribute
}

// Attribute is a parsed ONNX AttributeProto (scalar, list, and tensor
// values; graph-valued attributes are not needed for the supported operator
// set).
type Attribute struct {
	Name    string
	I       int64
	F       float32
	G       float64
	S       string
	Ints    []int64
	Floats  []float32
	Strings []string
	// T holds a tensor-valued attribute (AttributeProto field 5), used by
	// Constant.
	T *Tensor
}

// Int returns the attribute's integer value.
func (a *Attribute) Int() int64 { return a.I }

// Float returns the attribute's float value (float32 or double).
func (a *Attribute) Float() float64 {
	if a.G != 0 {
		return a.G
	}
	return float64(a.F)
}

// IntList returns the attribute's repeated integer value.
func (a *Attribute) IntList() []int64 { return a.Ints }

// attrInt looks up an integer attribute by name, returning def when absent.
func attrInt(node *Node, name string, def int64) int64 {
	for i := range node.Attrs {
		if node.Attrs[i].Name == name {
			return node.Attrs[i].I
		}
	}
	return def
}

// attrFloat looks up a float attribute by name, returning def when absent.
func attrFloat(node *Node, name string, def float64) float64 {
	for i := range node.Attrs {
		if node.Attrs[i].Name == name {
			return node.Attrs[i].Float()
		}
	}
	return def
}

// attrInts looks up a repeated-integer attribute by name, returning def
// when absent.
func attrInts(node *Node, name string, def []int64) []int64 {
	for i := range node.Attrs {
		if node.Attrs[i].Name == name && len(node.Attrs[i].Ints) > 0 {
			return node.Attrs[i].Ints
		}
	}
	return def
}

// attrString looks up a string attribute by name, returning def when absent.
func attrString(node *Node, name string, def string) string {
	for i := range node.Attrs {
		if node.Attrs[i].Name == name && node.Attrs[i].S != "" {
			return node.Attrs[i].S
		}
	}
	return def
}

// Load parses a ModelProto byte slice (ONNX wire format, field numbering as
// in onnx.proto3 v1.14 — the numbering used by virtually all released
// models).
func Load(data []byte) (*Model, error) {
	m := &Model{Graph: Graph{Initializers: make(map[string]*Tensor)}}
	err := eachField(data, func(f, wt int, w *wireReader) error {
		switch f {
		case 1: // ir_version
			v, err := w.uvarint()
			if err != nil {
				return err
			}
			m.IRVersion = uint32(v)
		case 7: // graph
			b, err := w.lenBytes()
			if err != nil {
				return err
			}
			g, err := parseGraph(b)
			if err != nil {
				return err
			}
			m.Graph = g
		case 8: // opset_import
			b, err := w.lenBytes()
			if err != nil {
				return err
			}
			var op Opset
			if err := eachField(b, func(f2, wt2 int, w2 *wireReader) error {
				switch f2 {
				case 1:
					s, err := w2.lenBytes()
					if err != nil {
						return err
					}
					op.Domain = string(s)
				case 2:
					v, err := w2.uvarint()
					if err != nil {
						return err
					}
					op.Version = int64(v)
				default:
					return w2.skip(wt2)
				}
				return nil
			}); err != nil {
				return err
			}
			m.Opsets = append(m.Opsets, op)
		default:
			return w.skip(wt)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("onnx: failed to parse model: %w", err)
	}
	if len(m.Graph.Nodes) == 0 {
		return nil, fmt.Errorf("onnx: model has no nodes")
	}
	return m, nil
}

func parseGraph(b []byte) (Graph, error) {
	g := Graph{Initializers: make(map[string]*Tensor)}
	err := eachField(b, func(f, wt int, w *wireReader) error {
		switch f {
		case 1: // node
			nb, err := w.lenBytes()
			if err != nil {
				return err
			}
			n, err := parseNode(nb)
			if err != nil {
				return err
			}
			g.Nodes = append(g.Nodes, n)
		case 2: // name
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			g.Name = string(s)
		case 5: // initializer
			tb, err := w.lenBytes()
			if err != nil {
				return err
			}
			t, name, err := parseTensor(tb)
			if err != nil {
				return err
			}
			g.Initializers[name] = t
		case 11: // input
			vb, err := w.lenBytes()
			if err != nil {
				return err
			}
			nt, err := parseValueInfo(vb)
			if err != nil {
				return err
			}
			g.Inputs = append(g.Inputs, nt)
		case 12: // output
			vb, err := w.lenBytes()
			if err != nil {
				return err
			}
			nt, err := parseValueInfo(vb)
			if err != nil {
				return err
			}
			g.Outputs = append(g.Outputs, nt)
		default: // doc_string, value_info, sparse_initializer, annotations
			return w.skip(wt)
		}
		return nil
	})
	if err != nil {
		return Graph{}, fmt.Errorf("onnx: failed to parse graph: %w", err)
	}
	return g, nil
}

func parseNode(b []byte) (Node, error) {
	var n Node
	err := eachField(b, func(f, wt int, w *wireReader) error {
		switch f {
		case 1: // input
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			n.Inputs = append(n.Inputs, string(s))
		case 2: // output
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			n.Outputs = append(n.Outputs, string(s))
		case 3: // name
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			n.Name = string(s)
		case 4: // op_type
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			n.OpType = string(s)
		case 5: // attribute
			ab, err := w.lenBytes()
			if err != nil {
				return err
			}
			a, err := parseAttribute(ab)
			if err != nil {
				return err
			}
			n.Attrs = append(n.Attrs, a)
		case 7: // domain
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			n.Domain = string(s)
		default:
			return w.skip(wt)
		}
		return nil
	})
	if err != nil {
		return Node{}, fmt.Errorf("onnx: failed to parse node: %w", err)
	}
	return n, nil
}

func parseAttribute(b []byte) (Attribute, error) {
	var a Attribute
	err := eachField(b, func(f, wt int, w *wireReader) error {
		switch f {
		case 1: // name
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			a.Name = string(s)
		case 2: // f
			vs, err := packedFloats32(w, wt)
			if err != nil {
				return err
			}
			if len(vs) > 0 {
				a.F = vs[0]
			}
		case 3: // i
			v, err := w.uvarint()
			if err != nil {
				return err
			}
			a.I = int64(v)
		case 4: // s
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			a.S = string(s)
		case 5: // t (tensor value, e.g. Constant)
			tb, err := w.lenBytes()
			if err != nil {
				return err
			}
			t, _, err := parseTensor(tb)
			if err != nil {
				return err
			}
			a.T = t
		case 7: // floats
			vs, err := packedFloats32(w, wt)
			if err != nil {
				return err
			}
			a.Floats = append(a.Floats, vs...)
		case 8: // ints
			vs, err := packedInts(w, wt)
			if err != nil {
				return err
			}
			a.Ints = append(a.Ints, vs...)
		case 9: // strings
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			a.Strings = append(a.Strings, string(s))
		default:
			return w.skip(wt)
		}
		return nil
	})
	if err != nil {
		return Attribute{}, fmt.Errorf("onnx: failed to parse attribute: %w", err)
	}
	return a, nil
}

func parseValueInfo(b []byte) (NamedType, error) {
	var nt NamedType
	err := eachField(b, func(f, wt int, w *wireReader) error {
		switch f {
		case 1: // name
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			nt.Name = string(s)
		case 2: // type (TypeProto)
			tb, err := w.lenBytes()
			if err != nil {
				return err
			}
			dt, err := parseTypeProto(tb)
			if err != nil {
				return err
			}
			nt.Dtype = dt
		default:
			return w.skip(wt)
		}
		return nil
	})
	if err != nil {
		return NamedType{}, fmt.Errorf("onnx: failed to parse value_info: %w", err)
	}
	return nt, nil
}

// parseTypeProto extracts the tensor element type from a TypeProto.
func parseTypeProto(b []byte) (DataType, error) {
	var dt DataType
	err := eachField(b, func(f, wt int, w *wireReader) error {
		if f != 1 || wt != 2 { // tensor_type
			return w.skip(wt)
		}
		tb, err := w.lenBytes()
		if err != nil {
			return err
		}
		return eachField(tb, func(f2, wt2 int, w2 *wireReader) error {
			if f2 == 1 { // elem_type
				v, err := w2.uvarint()
				if err != nil {
					return err
				}
				dt = DataType(v)
				return nil
			}
			return w2.skip(wt2)
		})
	})
	if err != nil {
		return 0, err
	}
	return dt, nil
}

// parseTensor decodes a TensorProto and returns the tensor together with
// its (possibly empty) name.
func parseTensor(b []byte) (*Tensor, string, error) {
	var (
		dims       []int64
		dt         DataType
		name       string
		raw        []byte
		floatData  []float32
		int32Data  []int64
		int64Data  []int64
		doubleData []float64
		external   bool
	)
	err := eachField(b, func(f, wt int, w *wireReader) error {
		switch f {
		case 1: // dims
			vs, err := packedInts(w, wt)
			if err != nil {
				return err
			}
			dims = append(dims, vs...)
		case 2: // data_type
			v, err := w.uvarint()
			if err != nil {
				return err
			}
			dt = DataType(v)
		case 4: // float_data
			vs, err := packedFloats32(w, wt)
			if err != nil {
				return err
			}
			floatData = append(floatData, vs...)
		case 5: // int32_data
			vs, err := packedInts(w, wt)
			if err != nil {
				return err
			}
			int32Data = append(int32Data, vs...)
		case 7: // int64_data
			vs, err := packedInts(w, wt)
			if err != nil {
				return err
			}
			int64Data = append(int64Data, vs...)
		case 8: // name
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			name = string(s)
		case 9: // raw_data
			s, err := w.lenBytes()
			if err != nil {
				return err
			}
			raw = s
		case 10: // double_data
			vs, err := packedFloats64(w, wt)
			if err != nil {
				return err
			}
			doubleData = append(doubleData, vs...)
		case 14: // data_location
			v, err := w.uvarint()
			if err != nil {
				return err
			}
			external = v == 1
		default:
			return w.skip(wt)
		}
		return nil
	})
	if err != nil {
		return nil, "", fmt.Errorf("onnx: failed to parse tensor %q: %w", name, err)
	}
	if external {
		return nil, "", fmt.Errorf("onnx: tensor %q uses external data, which is not supported", name)
	}
	n := shapeSize(dims)
	switch {
	case raw != nil:
		switch dt {
		case Float32:
			if len(raw) != int(n)*4 {
				return nil, "", fmt.Errorf("onnx: tensor %q raw data size %d does not match shape %v (float32)", name, len(raw), dims)
			}
			vs := make([]float32, n)
			for i := int64(0); i < n; i++ {
				vs[i] = mathFloat32FromLE(raw[i*4:])
			}
			t, err := NewTensor(dims, Float32, vs)
			return t, name, err
		case Double:
			if len(raw) != int(n)*8 {
				return nil, "", fmt.Errorf("onnx: tensor %q raw data size %d does not match shape %v (float64)", name, len(raw), dims)
			}
			vs := make([]float64, n)
			for i := int64(0); i < n; i++ {
				vs[i] = mathFloat64FromLE(raw[i*8:])
			}
			t, err := NewTensor(dims, Double, vs)
			return t, name, err
		case Int32:
			out := make([]int32, n)
			for i := int64(0); i < n; i++ {
				out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
			}
			t, err := NewTensor(dims, Int32, out)
			return t, name, err
		case Int64:
			out := make([]int64, n)
			for i := int64(0); i < n; i++ {
				out[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
			}
			t, err := NewTensor(dims, Int64, out)
			return t, name, err
		case Uint8:
			out := make([]uint8, n)
			copy(out, raw)
			t, err := NewTensor(dims, Uint8, out)
			return t, name, err
		case Bool:
			out := make([]bool, n)
			for i := int64(0); i < n; i++ {
				out[i] = raw[i] != 0
			}
			t, err := NewTensor(dims, Bool, out)
			return t, name, err
		case 10, 16: // FLOAT16, BFLOAT16
			return nil, "", fmt.Errorf("onnx: tensor %q uses %s, which is not supported (convert the model to float32)", name, dt)
		default:
			return nil, "", fmt.Errorf("onnx: tensor %q has unsupported raw data type %s", name, dt)
		}
	case len(floatData) > 0:
		if int64(len(floatData)) != n {
			return nil, "", fmt.Errorf("onnx: tensor %q has %d float values but shape %v requires %d", name, len(floatData), dims, n)
		}
		t, err := NewTensor(dims, dt, floatData)
		return t, name, err
	case len(doubleData) > 0:
		if int64(len(doubleData)) != n {
			return nil, "", fmt.Errorf("onnx: tensor %q has %d double values but shape %v requires %d", name, len(doubleData), dims, n)
		}
		t, err := NewTensor(dims, dt, doubleData)
		return t, name, err
	case len(int32Data) > 0:
		out := make([]int32, len(int32Data))
		for i, v := range int32Data {
			out[i] = int32(v)
		}
		if int64(len(out)) != n {
			return nil, "", fmt.Errorf("onnx: tensor %q has %d int32 values but shape %v requires %d", name, len(out), dims, n)
		}
		t, err := NewTensor(dims, dt, out)
		return t, name, err
	case len(int64Data) > 0:
		if int64(len(int64Data)) != n {
			return nil, "", fmt.Errorf("onnx: tensor %q has %d int64 values but shape %v requires %d", name, len(int64Data), dims, n)
		}
		t, err := NewTensor(dims, dt, int64Data)
		return t, name, err
	default:
		return nil, "", fmt.Errorf("onnx: tensor %q has no data payload", name)
	}
}
