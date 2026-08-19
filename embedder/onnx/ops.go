package onnx

import (
	"fmt"
	"math"
)

// This file contains the operator kernels for the supported ONNX subset.

// opWhere selects elementwise from x (cond true) or y (cond false).
func opWhere(n *Node, in []*Tensor) ([]*Tensor, error) {
	cond, err := in[0].AsBool()
	if err != nil {
		return nil, err
	}
	shape, err := broadcastShapes(in[0].Shape, in[1].Shape)
	if err != nil {
		return nil, err
	}
	if s, err := broadcastShapes(shape, in[2].Shape); err != nil {
		return nil, err
	} else {
		shape = s
	}
	cv := broadcastValues(boolsToFloats(cond), in[0].Shape, shape)
	xv, err := in[1].AsFloat64()
	if err != nil {
		return nil, err
	}
	yv, err := in[2].AsFloat64()
	if err != nil {
		return nil, err
	}
	xv = broadcastValues(xv, in[1].Shape, shape)
	yv = broadcastValues(yv, in[2].Shape, shape)
	out := make([]float64, len(cv))
	for i := range cv {
		if cv[i] != 0 {
			out[i] = xv[i]
		} else {
			out[i] = yv[i]
		}
	}
	t, err := makeFloat(shape, pickFloatDtype(in[1], in[2]), out)
	if err != nil {
		return nil, err
	}
	return one(t), nil
}

// opMatMul implements general N-D matrix multiplication with batch
// broadcasting. 1-D inputs are treated as vectors per the ONNX spec.
func opMatMul(a, b *Tensor) (*Tensor, error) {
	aS, bS := append([]int64{}, a.Shape...), append([]int64{}, b.Shape...)
	a1d, b1d := len(aS) == 1, len(bS) == 1
	if a1d {
		aS = append([]int64{1}, aS...)
	}
	if b1d {
		bS = append(bS, 1)
	}
	if len(aS) < 2 || len(bS) < 2 {
		return nil, fmt.Errorf("MatMul: inputs must have rank >= 1")
	}
	m, k := aS[len(aS)-2], aS[len(aS)-1]
	k2, nn := bS[len(bS)-2], bS[len(bS)-1]
	if k != k2 {
		return nil, fmt.Errorf("MatMul: inner dimensions differ (%d vs %d)", k, k2)
	}
	batch, err := broadcastShapes(aS[:len(aS)-2], bS[:len(bS)-2])
	if err != nil {
		return nil, err
	}
	aFull := append(batch, m, k)
	bFull := append(batch, k2, nn)
	outShape := append(append([]int64{}, batch...), m, nn)
	if a1d {
		outShape = outShape[1:]
	}
	if b1d {
		outShape = outShape[:len(outShape)-1]
	}
	// Fast path: when both operands are float32, accumulate in float32
	// directly on the raw data, skipping the float64 conversion entirely.
	// This is the common case for sentence-transformer ONNX exports.
	if a.Dtype == Float32 && b.Dtype == Float32 {
		av32, ok := a.Data.([]float32)
		if !ok {
			return nil, fmt.Errorf("MatMul: float32 tensor has unexpected data layout")
		}
		bv32, ok := b.Data.([]float32)
		if !ok {
			return nil, fmt.Errorf("MatMul: float32 tensor has unexpected data layout")
		}
		af := broadcastValues32(av32, aS, aFull)
		bf := broadcastValues32(bv32, bS, bFull)
		nBatch := shapeSize(batch)
		out := make([]float32, nBatch*m*nn)
		for bb := int64(0); bb < nBatch; bb++ {
			for i := int64(0); i < m; i++ {
				rowA := (bb*m + i) * k
				rowO := (bb*m + i) * nn
				for j := int64(0); j < nn; j++ {
					var s float32
					rowB := (bb*k2)*nn + j
					for t := int64(0); t < k; t++ {
						s += af[rowA+t] * bf[rowB+t*nn]
					}
					out[rowO+j] = s
				}
			}
		}
		return NewTensor(outShape, Float32, out)
	}
	av, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	bv, err := b.AsFloat64()
	if err != nil {
		return nil, err
	}
	av = broadcastValues(av, aS, aFull)
	bv = broadcastValues(bv, bS, bFull)
	nBatch := shapeSize(batch)
	out := make([]float64, nBatch*m*nn)
	for bb := int64(0); bb < nBatch; bb++ {
		for i := int64(0); i < m; i++ {
			rowA := (bb*m + i) * k
			rowO := (bb*m + i) * nn
			for j := int64(0); j < nn; j++ {
				s := 0.0
				rowB := (bb*k2)*nn + j
				for t := int64(0); t < k; t++ {
					s += av[rowA+t]*bv[rowB+t*nn] + 0
				}
				out[rowO+j] = s
			}
		}
	}
	return makeFloat(outShape, pickFloatDtype(a, b), out)
}

// opGemm computes Y = alpha * A @ B + beta * C, where C is optional.
func opGemm(n *Node, in []*Tensor) (*Tensor, error) {
	alpha := attrFloat(n, "alpha", 1)
	beta := attrFloat(n, "beta", 1)
	transA := attrInt(n, "transA", 0) == 1
	transB := attrInt(n, "transB", 0) == 1
	a, b := in[0], in[1]
	if a.Nd() != 2 || b.Nd() != 2 {
		return nil, fmt.Errorf("Gemm: inputs must be 2-D")
	}
	av, err := a.AsFloat64()
	if err != nil {
		return nil, err
	}
	bv, err := b.AsFloat64()
	if err != nil {
		return nil, err
	}
	var m, kk, nn int64
	if transA {
		// A is stored [k, m]; after transpose it is [m, k].
		kk, m = a.Shape[0], a.Shape[1]
	} else {
		m, kk = a.Shape[0], a.Shape[1]
	}
	if transB {
		nn = b.Shape[0]
	} else {
		nn = b.Shape[1]
	}
	// Effective shapes after transpose.
	effA, effB := a.Shape, b.Shape
	if transA {
		effA = []int64{a.Shape[1], a.Shape[0]}
	}
	if transB {
		effB = []int64{b.Shape[1], b.Shape[0]}
	}
	if effA[1] != effB[0] {
		return nil, fmt.Errorf("Gemm: inner dimensions differ (%d vs %d)", effA[1], effB[0])
	}
	// Flatten A and B in their effective layout.
	af := flattenEffective(av, a.Shape, transA, m, kk)
	bf := flattenEffective(bv, b.Shape, transB, kk, nn)
	var c *Tensor
	if len(in) > 2 {
		c = in[2]
	}
	out := make([]float64, m*nn)
	for i := int64(0); i < m; i++ {
		for j := int64(0); j < nn; j++ {
			s := 0.0
			for t := int64(0); t < kk; t++ {
				s += af[i*kk+t]*bf[t*nn+j] + 0
			}
			out[i*nn+j] = alpha * s
		}
	}
	if c != nil && beta != 0 {
		cv, err := c.AsFloat64()
		if err != nil {
			return nil, err
		}
		cShape, err := broadcastShapes(c.Shape, []int64{m, nn})
		if err != nil {
			return nil, err
		}
		if !sameShape(cShape, []int64{m, nn}) {
			return nil, fmt.Errorf("Gemm: cannot broadcast C shape %v to %v", c.Shape, []int64{m, nn})
		}
		cvb := broadcastValues(cv, c.Shape, []int64{m, nn})
		for i := range out {
			out[i] += beta * cvb[i]
		}
	}
	return makeFloat([]int64{m, nn}, pickFloatDtype(in[0], in[1]), out)
}

// flattenEffective returns A's values in row-major order of its effective
// (post-transpose) [rows, cols] layout.
func flattenEffective(v []float64, shape []int64, trans bool, rows, cols int64) []float64 {
	if !trans {
		return v
	}
	out := make([]float64, len(v))
	for i := int64(0); i < rows; i++ {
		for j := int64(0); j < cols; j++ {
			out[i*cols+j] = v[j*rows+i]
		}
	}
	return out
}

// opLayerNorm implements LayerNormalization: per-axis (mean, variance)
// normalization followed by scale and bias. It emits Y, and optionally
// mean and inverse standard deviation when the graph declares three outputs.
func opLayerNorm(n *Node, in []*Tensor) ([]*Tensor, error) {
	x := in[0]
	rank := x.Nd()
	axis, err := normalizeAxis(attrInt(n, "axis", -1), rank)
	if err != nil {
		return nil, err
	}
	eps := attrFloat(n, "epsilon", 1e-5)
	v, err := x.AsFloat64()
	if err != nil {
		return nil, err
	}
	scaleV, err := in[1].AsFloat64()
	if err != nil {
		return nil, err
	}
	biasV, err := in[2].AsFloat64()
	if err != nil {
		return nil, err
	}
	// scale and bias broadcast over the axis dimension (typically [H] or
	// [1, H]).
	sShape, err := broadcastShapes(in[1].Shape, x.Shape[axis:])
	if err != nil {
		return nil, err
	}
	scaleV = broadcastValues(scaleV, in[1].Shape, sShape)
	bShape, err := broadcastShapes(in[2].Shape, x.Shape[axis:])
	if err != nil {
		return nil, err
	}
	biasV = broadcastValues(biasV, in[2].Shape, bShape)

	pre := int64(1)
	for i := int64(0); i < axis; i++ {
		pre *= x.Shape[i]
	}
	ax := x.Shape[axis]
	post := int64(1)
	for i := axis + 1; i < int64(rank); i++ {
		post *= x.Shape[i]
	}
	out := make([]float64, len(v))
	means := make([]float64, pre*post)
	invstds := make([]float64, pre*post)
	for p := int64(0); p < pre; p++ {
		for q := int64(0); q < post; q++ {
			base := p*ax*post + q
			m := 0.0
			for i := int64(0); i < ax; i++ {
				m += v[base+i*post]
			}
			m /= float64(ax)
			var2 := 0.0
			for i := int64(0); i < ax; i++ {
				d := v[base+i*post] - m
				var2 += d * d
			}
			var2 /= float64(ax)
			invstd := 1 / math.Sqrt(var2+eps)
			means[p*post+q] = m
			invstds[p*post+q] = invstd
			for i := int64(0); i < ax; i++ {
				out[base+i*post] = (v[base+i*post]-m)*invstd*scaleV[i] + biasV[i]
			}
		}
	}
	y, err := makeFloat(x.Shape, pickFloatDtype(x), out)
	if err != nil {
		return nil, err
	}
	outs := []*Tensor{y}
	if len(n.Outputs) >= 2 {
		mShape := append(append([]int64{}, x.Shape[:axis]...), post)
		mt, err := makeFloat(mShape, pickFloatDtype(x), means)
		if err != nil {
			return nil, err
		}
		outs = append(outs, mt)
		if len(n.Outputs) >= 3 {
			st, err := makeFloat(mShape, pickFloatDtype(x), invstds)
			if err != nil {
				return nil, err
			}
			outs = append(outs, st)
		}
	}
	return outs, nil
}

// opBatchNorm implements inference-mode BatchNormalization:
// Y = (X - mean) / sqrt(var + epsilon) * scale + B, all applied per channel.
func opBatchNorm(n *Node, in []*Tensor) ([]*Tensor, error) {
	x := in[0]
	if x.Nd() < 2 {
		return nil, fmt.Errorf("BatchNormalization: input must have rank >= 2")
	}
	eps := attrFloat(n, "epsilon", 1e-5)
	v, err := x.AsFloat64()
	if err != nil {
		return nil, err
	}
	scaleV, err := in[1].AsFloat64()
	if err != nil {
		return nil, err
	}
	bV, err := in[2].AsFloat64()
	if err != nil {
		return nil, err
	}
	meanV, err := in[3].AsFloat64()
	if err != nil {
		return nil, err
	}
	varV, err := in[4].AsFloat64()
	if err != nil {
		return nil, err
	}
	trailing := int64(1)
	for i := 2; i < x.Nd(); i++ {
		trailing *= x.Shape[i]
	}
	c := x.Shape[1]
	out := make([]float64, len(v))
	for i, val := range v {
		ch := (int64(i) / trailing) % c
		out[i] = (val-meanV[ch])*scaleV[ch]/math.Sqrt(varV[ch]+eps) + bV[ch]
	}
	t, err := makeFloat(x.Shape, pickFloatDtype(x), out)
	if err != nil {
		return nil, err
	}
	outs := []*Tensor{t}
	// save_mean / save_invstdvariant outputs are not needed for inference.
	return outs[:len(n.Outputs)], nil
}

// opReduce dispatches ReduceMean/Sum/Max/Min with axes from the optional
// second input or the "axes" attribute.
func opReduce(n *Node, in []*Tensor, fn func([]float64) float64) ([]*Tensor, error) {
	keepDims := attrInt(n, "keepdims", 1) == 1
	var axes []int64
	if len(in) > 1 && in[1] != nil {
		av, err := in[1].AsInt64()
		if err != nil {
			return nil, err
		}
		axes = append(axes, av...)
	} else {
		axes = attrInts(n, "axes", nil)
	}
	if len(axes) == 0 {
		axes = nil // reduce all axes
	}
	// Deduplicate.
	seen := make(map[int64]bool)
	uniq := make([]int64, 0, len(axes))
	for _, a := range axes {
		if !seen[a] {
			seen[a] = true
			uniq = append(uniq, a)
		}
	}
	axes = uniq
	// Reduce with no axes specified reduces all axes (ONNX semantics).
	if len(axes) == 0 {
		full := make([]int64, in[0].Nd())
		for i := range full {
			full[i] = int64(i)
		}
		axes = full
	}
	t, err := reduceAlong(in[0], axes, keepDims, fn)
	if err != nil {
		return nil, err
	}
	return one(t), nil
}

// opReshape reshapes the input; the new shape comes from the second input
// tensor or the "shape" attribute. -1 infers one dimension and 0 copies the
// corresponding input dimension.
func opReshape(n *Node, in []*Tensor) ([]*Tensor, error) {
	var shape []int64
	if len(in) > 1 && in[1] != nil {
		sv, err := in[1].AsInt64()
		if err != nil {
			return nil, err
		}
		shape = sv
	} else {
		shape = attrInts(n, "shape", nil)
	}
	if len(shape) == 0 {
		return nil, fmt.Errorf("Reshape: no shape provided")
	}
	inSize := in[0].Size()
	known := int64(1)
	unknown := -1
	for i, d := range shape {
		switch {
		case d == 0:
			if i >= len(in[0].Shape) {
				return nil, fmt.Errorf("Reshape: 0 at position %d exceeds input rank %d", i, len(in[0].Shape))
			}
			shape[i] = in[0].Shape[i]
			known *= shape[i]
		case d == -1:
			if unknown >= 0 {
				return nil, fmt.Errorf("Reshape: at most one -1 allowed")
			}
			unknown = i
		default:
			if d < 0 {
				return nil, fmt.Errorf("Reshape: negative dimension %d not allowed", d)
			}
			known *= d
		}
	}
	if unknown >= 0 {
		if known == 0 || inSize%known != 0 {
			return nil, fmt.Errorf("Reshape: cannot infer dimension for size %d", inSize)
		}
		shape[unknown] = inSize / known
	}
	if shapeSize(shape) != inSize {
		return nil, fmt.Errorf("Reshape: new shape %v has %d elements, input has %d", shape, shapeSize(shape), inSize)
	}
	t := &Tensor{Shape: shape, Dtype: in[0].Dtype, Data: in[0].Data}
	return one(t), nil
}

// opTranspose permutes dimensions per the "perm" attribute (default:
// reverse).
func opTranspose(n *Node, in []*Tensor) (*Tensor, error) {
	a := in[0]
	rank := a.Nd()
	// Per ONNX opset >= 13, perm is an input; older opsets use an attribute.
	var perm []int64
	if len(in) >= 2 && in[1] != nil {
		if pv, err := in[1].AsInt64(); err == nil {
			perm = pv
		}
	}
	if len(perm) == 0 {
		perm = attrInts(n, "perm", nil)
	}
	if len(perm) == 0 {
		perm = make([]int64, rank)
		for i := range perm {
			perm[i] = int64(rank - 1 - i)
		}
	}
	if len(perm) != rank {
		return nil, fmt.Errorf("Transpose: perm has %d entries, rank is %d", len(perm), rank)
	}
	seen := make([]bool, rank)
	for _, p := range perm {
		if p < 0 || p >= int64(rank) || seen[p] {
			return nil, fmt.Errorf("Transpose: invalid perm %v", perm)
		}
		seen[p] = true
	}
	outShape := make([]int64, rank)
	for i, p := range perm {
		outShape[i] = a.Shape[p]
	}
	nElem := int(a.Size())
	switch a.Dtype {
	case Float32:
		fv, _ := a.AsFloat64()
		outF := make([]float32, nElem)
		fillPermutation(a.Shape, perm, nElem, func(inPos int) float64 { return fv[inPos] }, func(outPos int, v float64) { outF[outPos] = float32(v) })
		return NewTensor(outShape, a.Dtype, outF)
	case Int64:
		iv, _ := a.AsInt64()
		outI := make([]int64, nElem)
		fillPermutation(a.Shape, perm, nElem, func(inPos int) float64 { return float64(iv[inPos]) }, func(outPos int, v float64) { outI[outPos] = int64(v) })
		return NewTensor(outShape, a.Dtype, outI)
	case Int32:
		iv, _ := a.AsInt64()
		outI := make([]int32, nElem)
		fillPermutation(a.Shape, perm, nElem, func(inPos int) float64 { return float64(iv[inPos]) }, func(outPos int, v float64) { outI[outPos] = int32(v) })
		return NewTensor(outShape, a.Dtype, outI)
	case Bool:
		bv, _ := a.AsBool()
		outB := make([]bool, nElem)
		fillPermutation(a.Shape, perm, nElem, func(inPos int) float64 { return boolsToFloats(bv)[inPos] }, func(outPos int, v float64) { outB[outPos] = v != 0 })
		return NewTensor(outShape, a.Dtype, outB)
	default:
		return nil, fmt.Errorf("Transpose: unsupported dtype %s", a.Dtype)
	}
}

// fillPermutation maps output positions to input positions under perm and
// copies values via typed closures (values round-trip as float64, which is
// lossless for the supported dtypes).
func fillPermutation(inShape []int64, perm []int64, n int, get func(inPos int) float64, set func(outPos int, v float64)) {
	rank := len(inShape)
	outShape := make([]int64, rank)
	for i, p := range perm {
		outShape[i] = inShape[p]
	}
	outStride := make([]int64, rank)
	for i := rank - 1; i >= 0; i-- {
		outStride[i] = 1
		if i < rank-1 {
			outStride[i] = outStride[i+1] * outShape[i+1]
		}
	}
	inIdx := make([]int64, rank)
	for inPos := 0; inPos < n; inPos++ {
		pos := inPos
		outPos := 0
		for d := rank - 1; d >= 0; d-- {
			inIdx[d] = int64(pos % int(inShape[d]))
			pos /= int(inShape[d])
		}
		for d := 0; d < rank; d++ {
			outPos += int(inIdx[d]) * int(outStride[perm[d]])
		}
		set(outPos, get(inPos))
	}
}

// opFlatten flattens to 2-D around the given axis.
func opFlatten(n *Node, a *Tensor) ([]*Tensor, error) {
	axis, err := normalizeAxis(attrInt(n, "axis", 1), a.Nd())
	if err != nil {
		return nil, err
	}
	pre := int64(1)
	for i := int64(0); i < axis; i++ {
		pre *= a.Shape[i]
	}
	post := a.Size() / pre
	t := &Tensor{Shape: []int64{pre, post}, Dtype: a.Dtype, Data: a.Data}
	return one(t), nil
}

// opConcat joins tensors along the given axis.
func opConcat(n *Node, in []*Tensor) (*Tensor, error) {
	axis, err := normalizeAxis(attrInt(n, "axis", 0), in[0].Nd())
	if err != nil {
		return nil, err
	}
	rank := in[0].Nd()
	outShape := append([]int64{}, in[0].Shape...)
	outShape[axis] = 0
	offsets := make([]int64, len(in))
	for i, t := range in {
		if t.Nd() != rank || t.Dtype != in[0].Dtype {
			return nil, fmt.Errorf("Concat: rank or dtype mismatch")
		}
		for i := int64(0); i < int64(rank); i++ {
			if i != axis && t.Shape[i] != in[0].Shape[i] {
				return nil, fmt.Errorf("Concat: dimension mismatch at %d", i)
			}
		}
		outShape[axis] += t.Shape[axis]
		offsets[i] = outShape[axis] - t.Shape[axis]
	}
	nOut := shapeSize(outShape)
	switch in[0].Dtype {
	case Float32:
		out := make([]float32, nOut)
		fillConcat(in, axis, offsets, outShape, out)
		return NewTensor(outShape, Float32, out)
	case Double:
		out := make([]float64, nOut)
		fillConcat(in, axis, offsets, outShape, out)
		return NewTensor(outShape, Double, out)
	case Int64:
		out := make([]int64, nOut)
		fillConcat(in, axis, offsets, outShape, out)
		return NewTensor(outShape, Int64, out)
	case Int32:
		out := make([]int32, nOut)
		fillConcat(in, axis, offsets, outShape, out)
		return NewTensor(outShape, Int32, out)
	case Bool:
		out := make([]bool, nOut)
		fillConcat(in, axis, offsets, outShape, out)
		return NewTensor(outShape, Bool, out)
	case Uint8:
		out := make([]uint8, nOut)
		fillConcat(in, axis, offsets, outShape, out)
		return NewTensor(outShape, Uint8, out)
	default:
		return nil, fmt.Errorf("Concat: unsupported dtype %s", in[0].Dtype)
	}
}

// fillConcat maps output positions to the contributing input tensor along
// the concat axis and copies the element.
func fillConcat[T any](in []*Tensor, axis int64, offsets []int64, outShape []int64, dst []T) {
	rank := len(outShape)
	parts := make([][]T, len(in))
	for i, t := range in {
		parts[i] = t.Data.([]T)
	}
	outIdx := make([]int64, rank)
	for outPos := 0; outPos < len(dst); outPos++ {
		pos := outPos
		for d := rank - 1; d >= 0; d-- {
			outIdx[d] = int64(pos % int(outShape[d]))
			pos /= int(outShape[d])
		}
		ia := outIdx[axis]
		k := 0
		for j := 1; j < len(in); j++ {
			if ia >= offsets[j] {
				k = j
			}
		}
		var flat int64
		for d := 0; d < rank; d++ {
			id := outIdx[d]
			if d == int(axis) {
				id = ia - offsets[k]
			}
			flat += id * inStride(in[k].Shape, d)
		}
		dst[outPos] = parts[k][flat]
	}
}

// opSplit splits the first axis of data into the declared number of output
// tensors, using the split input or "split" attribute for sizes.
func opSplit(n *Node, in []*Tensor) ([]*Tensor, error) {
	data := in[0]
	rank := data.Nd()
	// ONNX Split: axis attribute (default 0).
	axis, err := normalizeAxis(attrInt(n, "axis", 0), rank)
	if err != nil {
		return nil, err
	}
	axisStride := int64(1)
	for i := axis + 1; i < int64(rank); i++ {
		axisStride *= data.Shape[i]
	}
	var sizes []int64
	if len(in) > 1 && in[1] != nil {
		sv, err := in[1].AsInt64()
		if err != nil {
			return nil, err
		}
		sizes = sv
	} else {
		sizes = attrInts(n, "split", nil)
	}
	// -1 in the split list means "infer".
	known := int64(0)
	unknown := -1
	for i, s := range sizes {
		if s == -1 {
			if unknown >= 0 {
				return nil, fmt.Errorf("Split: at most one -1 allowed")
			}
			unknown = i
		} else {
			known += s
		}
	}
	total := data.Shape[axis]
	if unknown >= 0 {
		sizes[unknown] = total - known
	} else if known != total {
		return nil, fmt.Errorf("Split: split sizes sum to %d but axis %d has %d elements", known, axis, total)
	}
	for _, s := range sizes {
		if s < 0 {
			return nil, fmt.Errorf("Split: inferred split size is negative (axis %d has %d elements)", axis, total)
		}
	}
	if len(sizes) != len(n.Outputs) {
		return nil, fmt.Errorf("Split: %d sizes for %d declared outputs", len(sizes), len(n.Outputs))
	}
	outs := make([]*Tensor, len(sizes))
	offset := int64(0)
	for i, s := range sizes {
		outShape := append([]int64{}, data.Shape...)
		outShape[axis] = s
		start := offset * axisStride
		end := (offset + s) * axisStride
		var sliceData any
		switch data.Dtype {
		case Float32:
			v := data.Data.([]float32)
			c := make([]float32, end-start)
			copy(c, v[start:end])
			sliceData = c
		case Double:
			v := data.Data.([]float64)
			c := make([]float64, end-start)
			copy(c, v[start:end])
			sliceData = c
		case Int64:
			v := data.Data.([]int64)
			c := make([]int64, end-start)
			copy(c, v[start:end])
			sliceData = c
		case Int32:
			v := data.Data.([]int32)
			c := make([]int32, end-start)
			copy(c, v[start:end])
			sliceData = c
		case Bool:
			v := data.Data.([]bool)
			c := make([]bool, end-start)
			copy(c, v[start:end])
			sliceData = c
		default:
			return nil, fmt.Errorf("Split: unsupported dtype %s", data.Dtype)
		}
		t, err := NewTensor(outShape, data.Dtype, sliceData)
		if err != nil {
			return nil, err
		}
		outs[i] = t
		offset += s
	}
	if offset != total {
		return nil, fmt.Errorf("Split: sizes sum to %d, axis has %d elements", offset, total)
	}
	return outs, nil
}

// opSlice implements Slice with optional starts/ends/axes/steps inputs.
// All of them default to the ONNX defaults: starts={0}, ends={max},
// axes={0}, steps={1}.
func opSlice(in []*Tensor) (*Tensor, error) {
	data := in[0]
	rank := data.Nd()
	getInts := func(idx int) ([]int64, error) {
		if idx >= len(in) || in[idx] == nil {
			return nil, nil
		}
		return in[idx].AsInt64()
	}
	starts, err := getInts(1)
	if err != nil {
		return nil, err
	}
	ends, err := getInts(2)
	if err != nil {
		return nil, err
	}
	axes, err := getInts(3)
	if err != nil {
		return nil, err
	}
	steps, err := getInts(4)
	if err != nil {
		return nil, err
	}
	if len(starts) != len(ends) || len(starts) != len(axes) {
		return nil, fmt.Errorf("Slice: starts/ends/axes length mismatch")
	}
	if len(steps) == 0 {
		steps = make([]int64, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	}
	if len(steps) != len(starts) {
		return nil, fmt.Errorf("Slice: steps length mismatch")
	}
	// Per-dimension slice parameters.
	params := make([]dimParam, rank)
	for i := range params {
		params[i] = dimParam{start: 0, end: data.Shape[i], step: 1, full: true}
	}
	for i := 0; i < len(starts); i++ {
		ax := axes[i]
		if ax < 0 {
			ax += int64(rank)
		}
		if ax < 0 || ax >= int64(rank) {
			return nil, fmt.Errorf("Slice: axis %d out of range", axes[i])
		}
		d := data.Shape[ax]
		s, e := starts[i], ends[i]
		if s < 0 {
			s += d
		}
		if e < 0 {
			e += d
		}
		if s < 0 {
			s = 0
		}
		if e > d {
			e = d
		}
		// ONNX allows ends far beyond the dimension (e.g. INT64_MAX).
		params[ax] = dimParam{start: s, end: e, step: steps[i], full: false}
	}
	outShape := make([]int64, rank)
	for i, p := range params {
		if p.step == 0 {
			return nil, fmt.Errorf("Slice: step must be non-zero")
		}
		var n int64
		if p.full {
			n = data.Shape[i]
		} else if p.step > 0 {
			if p.end > p.start {
				n = (p.end - p.start + p.step - 1) / p.step
			}
		} else if p.start > p.end {
			n = (p.start-p.end-1)/(-p.step) + 1
		}
		outShape[i] = n
	}
	nOut := shapeSize(outShape)
	switch data.Dtype {
	case Float32:
		src := data.Data.([]float32)
		out := make([]float32, nOut)
		copySlice(src, out, data.Shape, outShape, params)
		return NewTensor(outShape, data.Dtype, out)
	case Int64:
		src := data.Data.([]int64)
		out := make([]int64, nOut)
		copySlice(src, out, data.Shape, outShape, params)
		return NewTensor(outShape, data.Dtype, out)
	case Int32:
		src := data.Data.([]int32)
		out := make([]int32, nOut)
		copySlice(src, out, data.Shape, outShape, params)
		return NewTensor(outShape, data.Dtype, out)
	case Bool:
		src := data.Data.([]bool)
		out := make([]bool, nOut)
		copySlice(src, out, data.Shape, outShape, params)
		return NewTensor(outShape, data.Dtype, out)
	case Double:
		src := data.Data.([]float64)
		out := make([]float64, nOut)
		copySlice(src, out, data.Shape, outShape, params)
		return NewTensor(outShape, data.Dtype, out)
	default:
		return nil, fmt.Errorf("Slice: unsupported dtype %s", data.Dtype)
	}
}

// dimParam holds per-dimension slice parameters; full means the dimension
// was not mentioned by starts/ends (copy 1:1).
type dimParam struct {
	start, end int64
	step       int64
	full       bool
}

// copySlice copies elements from src (inShape) to dst (outShape) following
// per-dimension slice parameters.
func copySlice[T any](src, dst []T, inShape, outShape []int64, params []dimParam) {
	rank := len(inShape)
	dstIdx := make([]int64, rank)
	for dstOut := 0; dstOut < len(dst); dstOut++ {
		// Map the current dst index to a src index.
		var srcPos, dstPos int64
		for d := 0; d < rank; d++ {
			var si int64
			if params[d].full {
				si = dstIdx[d]
			} else {
				si = params[d].start + dstIdx[d]*params[d].step
			}
			srcPos += si * inStride(inShape, d)
			dstPos += dstIdx[d] * inStride(outShape, d)
		}
		dst[dstPos] = src[srcPos]
		for d := rank - 1; d >= 0; d-- {
			dstIdx[d]++
			if dstIdx[d] < outShape[d] {
				break
			}
			dstIdx[d] = 0
		}
	}
}

func inStride(shape []int64, d int) int64 {
	s := int64(1)
	for i := d + 1; i < len(shape); i++ {
		s *= shape[i]
	}
	return s
}

// opPad implements constant-mode Pad. "pads" order is
// [x1_begin, x2_begin, ..., x1_end, x2_end, ...].
func opPad(n *Node, in []*Tensor) (*Tensor, error) {
	data := in[0]
	rank := data.Nd()
	pads := attrInts(n, "pads", nil)
	if len(pads) != 2*rank {
		return nil, fmt.Errorf("Pad: pads must have 2*rank entries, got %d", len(pads))
	}
	mode := attrString(n, "mode", "constant")
	if mode != "constant" {
		return nil, fmt.Errorf("Pad: mode %q not supported (only constant)", mode)
	}
	value := 0.0
	for i := range n.Attrs {
		if n.Attrs[i].Name == "value" {
			if n.Attrs[i].T != nil {
				if v, err := n.Attrs[i].T.AsFloat64(); err == nil && len(v) > 0 {
					value = v[0]
				}
			} else {
				value = n.Attrs[i].Float()
			}
			break
		}
	}
	outShape := make([]int64, rank)
	for i := 0; i < rank; i++ {
		outShape[i] = data.Shape[i] + pads[i] + pads[rank+i]
		if outShape[i] < 0 {
			return nil, fmt.Errorf("Pad: resulting dimension %d is negative", outShape[i])
		}
	}
	nOut := shapeSize(outShape)
	switch data.Dtype {
	case Float32:
		out := make([]float32, nOut)
		for i := range out {
			out[i] = float32(value)
		}
		fillPadded(data.Data.([]float32), data.Shape, outShape, pads, out, func(v float32) float32 { return v })
		return NewTensor(outShape, data.Dtype, out)
	case Int64:
		out := make([]int64, nOut)
		fillPadded(data.Data.([]int64), data.Shape, outShape, pads, out, func(v int64) int64 { return v })
		return NewTensor(outShape, data.Dtype, out)
	default:
		return nil, fmt.Errorf("Pad: unsupported dtype %s", data.Dtype)
	}
}

// fillPadded places src (in inShape) into dst (outShape) offset by pads;
// pad regions keep their current (value-initialized) contents.
func fillPadded[T any](src []T, inShape, outShape []int64, pads []int64, dst []T, id func(T) T) {
	rank := len(inShape)
	inIdx := make([]int64, rank)
	outIdx := make([]int64, rank)
	for outPos := 0; outPos < len(dst); outPos++ {
		// decompose outPos
		pos := outPos
		for d := rank - 1; d >= 0; d-- {
			outIdx[d] = int64(pos % int(outShape[d]))
			pos /= int(outShape[d])
		}
		var inPos int64
		valid := true
		for d := 0; d < rank; d++ {
			inIdx[d] = outIdx[d] - pads[d]
			if inIdx[d] < 0 || inIdx[d] >= inShape[d] {
				valid = false
				break
			}
			inPos += inIdx[d] * inStride(inShape, d)
		}
		if valid {
			dst[outPos] = id(src[inPos])
		}
	}
}

// opTile repeats data per the repeats tensor.
func opTile(data, repeats *Tensor) (*Tensor, error) {
	rep, err := repeats.AsInt64()
	if err != nil {
		return nil, err
	}
	rank := data.Nd()
	// Repeats are right-aligned; missing leading dimensions repeat once.
	for len(rep) < rank {
		rep = append([]int64{1}, rep...)
	}
	if int64(len(rep)) != int64(rank) {
		return nil, fmt.Errorf("Tile: repeats rank %d does not match data rank %d", len(rep), rank)
	}
	for _, r := range rep {
		if r < 0 {
			return nil, fmt.Errorf("Tile: repeats must be non-negative")
		}
	}
	outShape := make([]int64, len(rep))
	for i, r := range rep {
		outShape[i] = data.Shape[i] * r
	}
	nOut := shapeSize(outShape)
	switch data.Dtype {
	case Float32:
		src := data.Data.([]float32)
		out := make([]float32, nOut)
		fillTile(src, data.Shape, outShape, rep, out)
		return NewTensor(outShape, data.Dtype, out)
	case Int64:
		src := data.Data.([]int64)
		out := make([]int64, nOut)
		fillTile(src, data.Shape, outShape, rep, out)
		return NewTensor(outShape, data.Dtype, out)
	default:
		return nil, fmt.Errorf("Tile: unsupported dtype %s", data.Dtype)
	}
}

func fillTile[T any](src []T, inShape, outShape []int64, rep []int64, dst []T) {
	rank := len(inShape)
	for outPos := 0; outPos < len(dst); outPos++ {
		pos := outPos
		inPos := 0
		for d := rank - 1; d >= 0; d-- {
			o := pos % int(outShape[d])
			pos /= int(outShape[d])
			if rep[d] > 0 {
				inPos += (o % int(inShape[d])) * int(inStride(inShape, d))
			}
		}
		dst[outPos] = src[inPos]
	}
}

// opExpand resizes data to the given shape (broadcasting semantics: the
// source may have fewer dimensions, which are right-aligned; size-1
// dimensions stretch).
func opExpand(data, shapeT *Tensor) (*Tensor, error) {
	shape, err := shapeT.AsInt64()
	if err != nil {
		return nil, err
	}
	outShape, err := broadcastShapes(data.Shape, shape)
	if err != nil {
		return nil, err
	}
	nOut := shapeSize(outShape)
	switch data.Dtype {
	case Float32:
		src := data.Data.([]float32)
		out := make([]float32, nOut)
		broadcastCopy(src, data.Shape, outShape, out)
		return NewTensor(outShape, Float32, out)
	case Double:
		src := data.Data.([]float64)
		out := make([]float64, nOut)
		broadcastCopy(src, data.Shape, outShape, out)
		return NewTensor(outShape, Double, out)
	case Int64:
		src := data.Data.([]int64)
		out := make([]int64, nOut)
		broadcastCopy(src, data.Shape, outShape, out)
		return NewTensor(outShape, Int64, out)
	case Int32:
		src := data.Data.([]int32)
		out := make([]int32, nOut)
		broadcastCopy(src, data.Shape, outShape, out)
		return NewTensor(outShape, Int32, out)
	case Bool:
		src := data.Data.([]bool)
		out := make([]bool, nOut)
		broadcastCopy(src, data.Shape, outShape, out)
		return NewTensor(outShape, Bool, out)
	default:
		return nil, fmt.Errorf("Expand: unsupported dtype %s", data.Dtype)
	}
}

// broadcastCopy maps output positions to source positions with
// right-aligned broadcasting (size-1 source dimensions wrap).
func broadcastCopy[T any](src []T, inShape, outShape []int64, dst []T) {
	rankOut := len(outShape)
	rankIn := len(inShape)
	for outPos := 0; outPos < len(dst); outPos++ {
		pos := outPos
		inPos := 0
		for d := rankOut - 1; d >= 0; d-- {
			o := pos % int(outShape[d])
			pos /= int(outShape[d])
			fi := d - (rankOut - rankIn)
			if fi < 0 {
				break
			}
			if inShape[fi] != 1 {
				inPos += (o % int(inShape[fi])) * int(inStride(inShape, fi))
			}
		}
		dst[outPos] = src[inPos]
	}
}

// opGather gathers slices of data indexed by the indices tensor along the
// given axis (default 0).
func opGather(n *Node, in []*Tensor) (*Tensor, error) {
	data, idx := in[0], in[1]
	axis, err := normalizeAxis(attrInt(n, "axis", 0), data.Nd())
	if err != nil {
		return nil, err
	}
	iv, err := idx.AsInt64()
	if err != nil {
		return nil, err
	}
	for _, i := range iv {
		if i < 0 || i >= data.Shape[axis] {
			return nil, fmt.Errorf("Gather: index %d out of range [0, %d)", i, data.Shape[axis])
		}
	}
	outShape := append(append([]int64{}, data.Shape[:axis]...), idx.Shape...)
	outShape = append(outShape, data.Shape[axis+1:]...)
	pre := int64(1)
	for i := int64(0); i < axis; i++ {
		pre *= data.Shape[i]
	}
	inner := int64(1)
	for i := axis + 1; i < int64(data.Nd()); i++ {
		inner *= data.Shape[i]
	}
	ix := idx.Size()
	switch data.Dtype {
	case Float32:
		src := data.Data.([]float32)
		out := make([]float32, pre*ix*inner)
		for p := int64(0); p < pre; p++ {
			for k := int64(0); k < ix; k++ {
				copy(out[(p*ix+k)*inner:(p*ix+k+1)*inner], src[(p*data.Shape[axis]+iv[k])*inner:(p*data.Shape[axis]+iv[k]+1)*inner])
			}
		}
		return NewTensor(outShape, Float32, out)
	case Int64:
		src := data.Data.([]int64)
		out := make([]int64, pre*ix*inner)
		for p := int64(0); p < pre; p++ {
			for k := int64(0); k < ix; k++ {
				copy(out[(p*ix+k)*inner:(p*ix+k+1)*inner], src[(p*data.Shape[axis]+iv[k])*inner:(p*data.Shape[axis]+iv[k]+1)*inner])
			}
		}
		return NewTensor(outShape, Int64, out)
	default:
		return nil, fmt.Errorf("Gather: unsupported dtype %s", data.Dtype)
	}
}

// opSqueeze removes size-1 dimensions. Axes come from the second input
// (opset >= 13) or the "axes" attribute; an empty list removes all size-1
// dimensions.
func opSqueeze(n *Node, in []*Tensor) (*Tensor, error) {
	data := in[0]
	var axes []int64
	if len(in) > 1 && in[1] != nil {
		av, err := in[1].AsInt64()
		if err != nil {
			return nil, err
		}
		axes = av
	} else {
		axes = attrInts(n, "axes", nil)
	}
	rank := data.Nd()
	remove := make([]bool, rank)
	if len(axes) == 0 {
		for i, d := range data.Shape {
			if d == 1 {
				remove[i] = true
			}
		}
	} else {
		for _, a := range axes {
			aa, err := normalizeAxis(a, rank)
			if err != nil {
				return nil, err
			}
			if data.Shape[aa] != 1 {
				return nil, fmt.Errorf("Squeeze: dimension %d has size %d, not 1", a, data.Shape[aa])
			}
			remove[aa] = true
		}
	}
	var outShape []int64
	for i, d := range data.Shape {
		if !remove[i] {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = []int64{}
	}
	t := &Tensor{Shape: outShape, Dtype: data.Dtype, Data: data.Data}
	return t, nil
}

// opUnsqueeze inserts size-1 dimensions at the given axes.
func opUnsqueeze(n *Node, in []*Tensor) (*Tensor, error) {
	data := in[0]
	var axes []int64
	if len(in) > 1 && in[1] != nil {
		av, err := in[1].AsInt64()
		if err != nil {
			return nil, err
		}
		axes = av
	} else {
		axes = attrInts(n, "axes", nil)
	}
	if len(axes) > data.Nd()+1 {
		return nil, fmt.Errorf("Unsqueeze: too many axes")
	}
	// Normalize all axes against the final rank, then splice them in.
	outShape := make([]int64, 0, data.Nd()+len(axes))
	finalRank := int64(data.Nd()) + int64(len(axes))
	pos := 0
	for _, a := range axes {
		aa := a
		if aa < 0 {
			aa += finalRank
		}
		if aa < 0 || aa > finalRank {
			return nil, fmt.Errorf("Unsqueeze: invalid axis %d", a)
		}
		for pos < int(aa) && pos < data.Nd() {
			outShape = append(outShape, data.Shape[pos])
			pos++
		}
		outShape = append(outShape, 1)
	}
	for pos < data.Nd() {
		outShape = append(outShape, data.Shape[pos])
		pos++
	}
	t := &Tensor{Shape: outShape, Dtype: data.Dtype, Data: data.Data}
	return t, nil
}

// opRange produces [start, start+delta, ...] while < limit.
func opRange(in []*Tensor) (*Tensor, error) {
	sv, err := in[0].AsFloat64()
	if err != nil {
		return nil, err
	}
	lv, err := in[1].AsFloat64()
	if err != nil {
		return nil, err
	}
	dv, err := in[2].AsFloat64()
	if err != nil {
		return nil, err
	}
	start, limit, delta := sv[0], lv[0], dv[0]
	if delta == 0 {
		return nil, fmt.Errorf("Range: delta must be non-zero")
	}
	count := int64(0)
	if (limit-start)*delta > 0 {
		count = int64(math.Ceil((limit - start) / delta))
	}
	if count < 0 {
		count = 0
	}
	intLike := in[0].Dtype == Int32 || in[0].Dtype == Int64
	if intLike {
		out := make([]int64, count)
		for i := int64(0); i < count; i++ {
			out[i] = int64(start + delta*float64(i))
		}
		t, err := NewTensor([]int64{count}, Int64, out)
		if err != nil {
			return nil, err
		}
		return t, nil
	}
	out := make([]float64, count)
	for i := int64(0); i < count; i++ {
		out[i] = start + delta*float64(i)
	}
	t, err := makeFloat([]int64{count}, pickFloatDtype(in[0], in[1], in[2]), out)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// opCast converts a tensor to the dtype named by the "to" attribute.
func opCast(a *Tensor, n *Node) ([]*Tensor, error) {
	to := DataType(attrInt(n, "to", int64(a.Dtype)))
	if to == a.Dtype {
		return one(a), nil
	}
	fv, ferr := a.AsFloat64()
	iv, ierr := a.AsInt64()
	bv, berr := a.AsBool()
	switch to {
	case Float32:
		if ferr != nil {
			return nil, ferr
		}
		out := make([]float32, len(fv))
		for i, v := range fv {
			out[i] = float32(v)
		}
		t, err := NewTensor(a.Shape, Float32, out)
		return wrap(t, err)
	case Double:
		if ferr != nil {
			return nil, ferr
		}
		t, err := NewTensor(a.Shape, Double, fv)
		return wrap(t, err)
	case Int32:
		if ierr != nil {
			return nil, ierr
		}
		out := make([]int32, len(iv))
		for i, v := range iv {
			out[i] = int32(v)
		}
		t, err := NewTensor(a.Shape, Int32, out)
		return wrap(t, err)
	case Int64:
		if ierr != nil {
			return nil, ierr
		}
		t, err := NewTensor(a.Shape, Int64, iv)
		return wrap(t, err)
	case Bool:
		if berr != nil {
			return nil, berr
		}
		t, err := NewTensor(a.Shape, Bool, bv)
		return wrap(t, err)
	case Uint8:
		if ierr != nil {
			return nil, ierr
		}
		out := make([]uint8, len(iv))
		for i, v := range iv {
			out[i] = uint8(v)
		}
		t, err := NewTensor(a.Shape, Uint8, out)
		return wrap(t, err)
	default:
		return nil, fmt.Errorf("Cast: target dtype %s not supported", to)
	}
}

// opConstant produces a tensor from the "value" attribute (or the
// scalar value_int / value_float / value_string attributes).
func opConstant(n *Node) ([]*Tensor, error) {
	for i := range n.Attrs {
		a := &n.Attrs[i]
		switch a.Name {
		case "value":
			if a.T == nil {
				return nil, fmt.Errorf("Constant: missing value attribute")
			}
			return one(a.T), nil
		case "value_int":
			t, err := NewTensor([]int64{}, Int64, []int64{a.I})
			return wrap(t, err)
		case "value_float":
			t, err := NewTensor([]int64{}, Float32, []float32{a.F})
			return wrap(t, err)
		case "value_string":
			// Not needed for numeric inference paths.
			return nil, fmt.Errorf("Constant: string constants are not supported")
		}
	}
	return nil, fmt.Errorf("Constant: no value attribute found")
}

// opDropout passes the input through unchanged in inference mode (training
// mode is not supported).
func opDropout(n *Node, in []*Tensor) ([]*Tensor, error) {
	train := false
	if len(in) >= 3 && in[2] != nil {
		if v, err := in[2].AsInt64(); err == nil && len(v) > 0 {
			train = v[0] != 0
		}
	}
	if !train {
		train = attrInt(n, "training_mode", 0) == 1
	}
	if train {
		return nil, fmt.Errorf("Dropout: training mode is not supported in this inference runtime")
	}
	outs := []*Tensor{in[0]}
	if len(n.Outputs) >= 2 {
		// Mask output: all true.
		mask := make([]bool, in[0].Size())
		for i := range mask {
			mask[i] = true
		}
		m, err := NewTensor(in[0].Shape, Bool, mask)
		if err != nil {
			return nil, err
		}
		outs = append(outs, m)
	}
	return outs, nil
}
