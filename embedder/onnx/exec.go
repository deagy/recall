package onnx

import (
	"context"
	"fmt"
	"math"
)

// Executor runs graph nodes against an environment of named tensors.
type Executor struct {
	graph Graph
	env   map[string]*Tensor
}

func newExecutor(g Graph) *Executor {
	env := make(map[string]*Tensor, len(g.Initializers)+len(g.Inputs))
	for name, t := range g.Initializers {
		env[name] = t
	}
	return &Executor{graph: g, env: env}
}

func (e *Executor) get(name string) (*Tensor, error) {
	t, ok := e.env[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in graph environment", name)
	}
	return t, nil
}

func (e *Executor) set(name string, t *Tensor) { e.env[name] = t }

// run executes the graph with the given feed tensors and returns the graph
// outputs.
func (e *Executor) run(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
	feed := make(map[string]bool, len(e.graph.FeedInputs()))
	for _, in := range e.graph.FeedInputs() {
		feed[in.Name] = true
	}
	for name, t := range inputs {
		if !feed[name] {
			return nil, fmt.Errorf("unexpected input %q: it is not a graph input", name)
		}
		e.env[name] = t
	}
	for _, in := range e.graph.FeedInputs() {
		if _, ok := e.env[in.Name]; !ok {
			return nil, fmt.Errorf("missing graph input %q (expected element type %s)", in.Name, in.Dtype)
		}
	}
	for i := range e.graph.Nodes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := e.execNode(&e.graph.Nodes[i]); err != nil {
			return nil, err
		}
	}
	out := make(map[string]*Tensor, len(e.graph.Outputs))
	for _, o := range e.graph.Outputs {
		t, ok := e.env[o.Name]
		if !ok {
			return nil, fmt.Errorf("graph output %q was not produced by the graph", o.Name)
		}
		out[o.Name] = t
	}
	return out, nil
}

func nodeDesc(n *Node) string {
	name := n.Name
	if name == "" {
		name = n.OpType
	}
	return fmt.Sprintf("%s(%s) [%s]", n.OpType, stringsJoin(n.Inputs, ", "), name)
}

func stringsJoin(ss []string, sep string) string {
	out := ""
	for i, s := range ss {
		if i > 0 {
			out += sep
		}
		out += s
	}
	return out
}

func (e *Executor) execNode(n *Node) error {
	if n.Domain != "" && n.Domain != "ai.onnx" {
		return fmt.Errorf("operator %q in domain %q is not supported (only the default ai.onnx domain is available)", n.OpType, n.Domain)
	}
	in := make([]*Tensor, len(n.Inputs))
	for i, name := range n.Inputs {
		if name == "" { // optional input not provided
			continue
		}
		t, err := e.get(name)
		if err != nil {
			return fmt.Errorf("node %q: %w", nodeDesc(n), err)
		}
		in[i] = t
	}
	outs, err := e.execOp(n, in)
	if err != nil {
		if _, ok := err.(*unsupportedOpError); !ok {
			return fmt.Errorf("node %q: %w", nodeDesc(n), err)
		}
		return err
	}
	if len(outs) != len(n.Outputs) {
		return fmt.Errorf("node %q: operator produced %d outputs, graph declares %d", nodeDesc(n), len(outs), len(n.Outputs))
	}
	for i, name := range n.Outputs {
		e.set(name, outs[i])
	}
	return nil
}

type unsupportedOpError struct {
	op     string
	domain string
}

func (u *unsupportedOpError) Error() string {
	return fmt.Sprintf("operator %q is not supported by this ONNX runtime", u.op)
}

// elemUnary applies fn elementwise to a, keeping a's shape.
func elemUnary(a *Tensor, fn func(float64) float64) (*Tensor, error) {
	v, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = fn(x)
	}
	return makeFloat(a.Shape, pickFloatDtype(a), out)
}

// elemBinary applies fn elementwise to a and b with broadcasting.
func elemBinary(a, b *Tensor, fn func(x, y float64) float64) (*Tensor, error) {
	shape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	av, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	bv, err := b.AsFloat64()
	if err != nil {
		return nil, err
	}
	ab := broadcastValues(av, a.Shape, shape)
	bb := broadcastValues(bv, b.Shape, shape)
	out := make([]float64, len(ab))
	for i := range ab {
		out[i] = fn(ab[i], bb[i])
	}
	return makeFloat(shape, pickFloatDtype(a, b), out)
}

// elemCompare compares a and b elementwise (broadcasting) into a bool tensor.
func elemCompare(a, b *Tensor, fn func(x, y float64) bool) (*Tensor, error) {
	shape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	av, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	bv, err := b.AsFloat64()
	if err != nil {
		return nil, err
	}
	ab := broadcastValues(av, a.Shape, shape)
	bb := broadcastValues(bv, b.Shape, shape)
	out := make([]bool, len(ab))
	for i := range ab {
		out[i] = fn(ab[i], bb[i])
	}
	return NewTensor(shape, Bool, out)
}

// elemBoolBinary applies fn elementwise to two bool tensors (broadcasting).
func elemBoolBinary(a, b *Tensor, fn func(x, y bool) bool) (*Tensor, error) {
	shape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	av, err := a.AsBool()
	if err != nil {
		return nil, err
	}
	bv, err := b.AsBool()
	if err != nil {
		return nil, err
	}
	ab := make([]bool, shapeSize(shape))
	for i, f := range broadcastValues(boolsToFloats(av), a.Shape, shape) {
		ab[i] = f != 0
	}
	bb := make([]bool, shapeSize(shape))
	for i, f := range broadcastValues(boolsToFloats(bv), b.Shape, shape) {
		bb[i] = f != 0
	}
	out := make([]bool, len(ab))
	for i := range ab {
		out[i] = fn(ab[i], bb[i])
	}
	return NewTensor(shape, Bool, out)
}

func boolsToFloats(v []bool) []float64 {
	out := make([]float64, len(v))
	for i, b := range v {
		if b {
			out[i] = 1
		}
	}
	return out
}

// softmax computes softmax along the given axis (negative axes count from
// the end).
func softmax(a *Tensor, axis int64) (*Tensor, error) {
	rank := int64(a.Nd())
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("softmax: axis %d out of range for rank %d", axis, rank)
	}
	v, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	pre := int64(1)
	for i := int64(0); i < axis; i++ {
		pre *= a.Shape[i]
	}
	ax := a.Shape[axis]
	post := int64(1)
	for i := axis + 1; i < rank; i++ {
		post *= a.Shape[i]
	}
	out := make([]float64, len(v))
	for p := int64(0); p < pre; p++ {
		for q := int64(0); q < post; q++ {
			base := p*ax*post + q
			maxV := v[base]
			for i := int64(1); i < ax; i++ {
				if v[base+i*post] > maxV {
					maxV = v[base+i*post]
				}
			}
			sum := 0.0
			for i := int64(0); i < ax; i++ {
				e := math.Exp(v[base+i*post] - maxV)
				out[base+i*post] = e
				sum += e
			}
			if sum != 0 {
				for i := int64(0); i < ax; i++ {
					out[base+i*post] /= sum
				}
			}
		}
	}
	return makeFloat(a.Shape, pickFloatDtype(a), out)
}

// reduceAlong reduces a's elements along the given axes. fn receives the
// elements of each reduced group (row-major order) and returns the combined
// value. When keepDims is set, reduced axes remain in the output shape with
// size 1.
func reduceAlong(a *Tensor, axes []int64, keepDims bool, fn func(vals []float64) float64) (*Tensor, error) {
	rank := a.Nd()
	reduced := make([]bool, rank)
	for _, ax := range axes {
		n := ax
		if n < 0 {
			n += int64(rank)
		}
		if n < 0 || n >= int64(rank) {
			return nil, fmt.Errorf("reduce: axis %d out of range for rank %d", ax, rank)
		}
		reduced[n] = true
	}
	var outShape []int64
	for i, d := range a.Shape {
		switch {
		case keepDims && reduced[i]:
			outShape = append(outShape, 1)
		case !reduced[i]:
			outShape = append(outShape, d)
		}
	}
	// For every input dimension, its stride in the output index space
	// (zero for reduced dimensions — they do not advance the slot).
	outStride := make([]int64, rank)
	for i := rank - 1; i >= 0; i-- {
		if reduced[i] {
			outStride[i] = 0
			continue
		}
		outStride[i] = 1
		for j := i + 1; j < rank; j++ {
			if !reduced[j] {
				outStride[i] *= a.Shape[j]
			}
		}
	}
	v, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	nOut := shapeSize(outShape)
	groups := make([][]float64, nOut)
	idx := make([]int64, rank)
	for i := range v {
		pos := i
		for d := rank - 1; d >= 0; d-- {
			idx[d] = int64(pos % int(a.Shape[d]))
			pos /= int(a.Shape[d])
		}
		slot := int64(0)
		for d := 0; d < rank; d++ {
			slot += idx[d] * outStride[d]
		}
		groups[slot] = append(groups[slot], float64(v[i]))
	}
	out := make([]float64, nOut)
	for slot, g := range groups {
		out[slot] = fn(g)
	}
	return makeFloat(outShape, pickFloatDtype(a), out)
}

// normalizeAxes resolves a possibly-negative axis to an absolute one.
func normalizeAxis(axis int64, rank int) (int64, error) {
	if axis < 0 {
		axis += int64(rank)
	}
	if axis < 0 || axis >= int64(rank) {
		return 0, fmt.Errorf("axis %d out of range for rank %d", axis, rank)
	}
	return axis, nil
}

// erf is the error function (Abramowitz & Stegun 7.1.26 approximation,
// |error| <= 1.5e-7 — well within float32 precision).
func erf(x float64) float64 {
	sign := 1.0
	if x < 0 {
		sign = -1
		x = -x
	}
	t := 1 / (1 + 0.3275911*x)
	y := 1 - ((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*math.Exp(-x*x)
	return sign * y
}

// execOp executes a single node's operator and returns its output tensors
// in graph-declared order.
func (e *Executor) execOp(n *Node, in []*Tensor) ([]*Tensor, error) {
	switch n.OpType {
	case "Constant":
		return opConstant(n)
	case "Identity":
		return one(in[0]), nil
	case "Cast":
		return opCast(in[0], n)
	case "Dropout":
		return opDropout(n, in)
	case "Abs":
		t, err := elemUnary(in[0], math.Abs)
		return wrap(t, err)
	case "Neg":
		t, err := elemUnary(in[0], func(x float64) float64 { return -x })
		return wrap(t, err)
	case "Exp":
		t, err := elemUnary(in[0], math.Exp)
		return wrap(t, err)
	case "Log":
		t, err := elemUnary(in[0], math.Log)
		return wrap(t, err)
	case "Sqrt":
		t, err := elemUnary(in[0], math.Sqrt)
		return wrap(t, err)
	case "Reciprocal":
		t, err := elemUnary(in[0], func(x float64) float64 { return 1 / x })
		return wrap(t, err)
	case "Sigmoid":
		t, err := elemUnary(in[0], func(x float64) float64 { return 1 / (1 + math.Exp(-x)) })
		return wrap(t, err)
	case "Tanh":
		t, err := elemUnary(in[0], math.Tanh)
		return wrap(t, err)
	case "Relu":
		t, err := elemUnary(in[0], func(x float64) float64 {
			if x > 0 {
				return x
			}
			return 0
		})
		return wrap(t, err)
	case "Gelu":
		approx := attrInt(n, "approximate", 0)
		t, err := elemUnary(in[0], func(x float64) float64 {
			if approx == 1 {
				c := 2 / math.Sqrt(math.Pi)
				return 0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x)))
			}
			return 0.5 * x * (1 + erf(x/math.Sqrt2))
		})
		return wrap(t, err)
	case "Erf":
		t, err := elemUnary(in[0], erf)
		return wrap(t, err)
	case "Sign":
		t, err := elemUnary(in[0], func(x float64) float64 {
			if x > 0 {
				return 1
			}
			if x < 0 {
				return -1
			}
			return 0
		})
		return wrap(t, err)
	case "Floor":
		t, err := elemUnary(in[0], math.Floor)
		return wrap(t, err)
	case "Ceil":
		t, err := elemUnary(in[0], math.Ceil)
		return wrap(t, err)
	case "Add":
		t, err := elemBinary(in[0], in[1], func(x, y float64) float64 { return x + y })
		return wrap(t, err)
	case "Sub":
		t, err := elemBinary(in[0], in[1], func(x, y float64) float64 { return x - y })
		return wrap(t, err)
	case "Mul":
		t, err := elemBinary(in[0], in[1], func(x, y float64) float64 { return x * y })
		return wrap(t, err)
	case "Div":
		t, err := elemBinary(in[0], in[1], func(x, y float64) float64 { return x / y })
		return wrap(t, err)
	case "Pow":
		t, err := elemBinary(in[0], in[1], math.Pow)
		return wrap(t, err)
	case "Mod":
		t, err := elemBinary(in[0], in[1], func(x, y float64) float64 { return math.Mod(x, y) })
		return wrap(t, err)
	case "Greater":
		t, err := elemCompare(in[0], in[1], func(x, y float64) bool { return x > y })
		return wrap(t, err)
	case "GreaterOrEqual":
		t, err := elemCompare(in[0], in[1], func(x, y float64) bool { return x >= y })
		return wrap(t, err)
	case "Less":
		t, err := elemCompare(in[0], in[1], func(x, y float64) bool { return x < y })
		return wrap(t, err)
	case "LessOrEqual":
		t, err := elemCompare(in[0], in[1], func(x, y float64) bool { return x <= y })
		return wrap(t, err)
	case "Equal":
		t, err := elemCompare(in[0], in[1], func(x, y float64) bool { return x == y })
		return wrap(t, err)
	case "And":
		t, err := elemBoolBinary(in[0], in[1], func(x, y bool) bool { return x && y })
		return wrap(t, err)
	case "Or":
		t, err := elemBoolBinary(in[0], in[1], func(x, y bool) bool { return x || y })
		return wrap(t, err)
	case "Xor":
		t, err := elemBoolBinary(in[0], in[1], func(x, y bool) bool { return x != y })
		return wrap(t, err)
	case "Not":
		v, err := in[0].AsBool()
		if err != nil {
			return nil, err
		}
		for i := range v {
			v[i] = !v[i]
		}
		t, err := NewTensor(in[0].Shape, Bool, v)
		return wrap(t, err)
	case "Where":
		return opWhere(n, in)
	case "MatMul":
		t, err := opMatMul(in[0], in[1])
		return wrap(t, err)
	case "Gemm":
		t, err := opGemm(n, in)
		return wrap(t, err)
	case "Softmax":
		t, err := softmax(in[0], attrInt(n, "axis", -1))
		return wrap(t, err)
	case "LayerNormalization":
		return opLayerNorm(n, in)
	case "BatchNormalization":
		return opBatchNorm(n, in)
	case "ReduceMean":
		return opReduce(n, in, func(g []float64) float64 {
			s := 0.0
			for _, v := range g {
				s += v
			}
			return s / float64(len(g))
		})
	case "ReduceSum":
		return opReduce(n, in, func(g []float64) float64 {
			s := 0.0
			for _, v := range g {
				s += v
			}
			return s
		})
	case "ReduceMax":
		return opReduce(n, in, func(g []float64) float64 {
			m := g[0]
			for _, v := range g[1:] {
				if v > m {
					m = v
				}
			}
			return m
		})
	case "ReduceMin":
		return opReduce(n, in, func(g []float64) float64 {
			m := g[0]
			for _, v := range g[1:] {
				if v < m {
					m = v
				}
			}
			return m
		})
	case "Reshape":
		return opReshape(n, in)
	case "Transpose":
		t, err := opTranspose(n, in)
		return wrap(t, err)
	case "Flatten":
		return opFlatten(n, in[0])
	case "Concat":
		t, err := opConcat(n, in)
		return wrap(t, err)
	case "Split":
		return opSplit(n, in)
	case "Slice":
		t, err := opSlice(in)
		return wrap(t, err)
	case "Pad":
		t, err := opPad(n, in)
		return wrap(t, err)
	case "Tile":
		t, err := opTile(in[0], in[1])
		return wrap(t, err)
	case "Expand":
		t, err := opExpand(in[0], in[1])
		return wrap(t, err)
	case "Gather":
		t, err := opGather(n, in)
		return wrap(t, err)
	case "Shape":
		v := make([]int64, len(in[0].Shape))
		copy(v, in[0].Shape)
		t, err := NewTensor([]int64{int64(len(v))}, Int64, v)
		return wrap(t, err)
	case "Squeeze":
		t, err := opSqueeze(n, in)
		return wrap(t, err)
	case "Unsqueeze":
		t, err := opUnsqueeze(n, in)
		return wrap(t, err)
	case "Range":
		t, err := opRange(in)
		return wrap(t, err)
	default:
		return nil, &unsupportedOpError{op: n.OpType, domain: n.Domain}
	}
}

// one wraps a single tensor as the kernel return shape.
func one(t *Tensor) []*Tensor { return []*Tensor{t} }

// wrap converts a (tensor, error) pair into the kernel return shape.
func wrap(t *Tensor, err error) ([]*Tensor, error) {
	if err != nil {
		return nil, err
	}
	return []*Tensor{t}, nil
}
