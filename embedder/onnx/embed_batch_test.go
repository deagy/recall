package onnx

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildAddModel builds a model y = x + 10 where 10 is a shared initializer.
// The shared-initializer case is exactly what BatchRun relies on: multiple
// parallel runs read the same initializer tensor concurrently.
func buildAddModel(t testing.TB) *Model {
	t.Helper()
	c := mustTensor(NewTensor([]int64{1}, Float32, []float32{10}))
	nodes := []NodeSpec{{Op: "Add", Inputs: []string{"x", "c"}, Outputs: []string{"y"}}}
	inits := map[string]*Tensor{"c": c}
	inputs := []NamedType{{Name: "x", Dtype: Float32}}
	outputs := []NamedType{{Name: "y", Dtype: Float32}}
	m, err := Load(Encode(nodes, inits, inputs, outputs))
	require.NoError(t, err)
	return m
}

func addFeed(v float32) map[string]*Tensor {
	return map[string]*Tensor{"x": mustTensor(NewTensor([]int64{1}, Float32, []float32{v}))}
}

func TestBatchRun_Empty(t *testing.T) {
	m := buildAddModel(t)
	outs, err := m.BatchRun(context.Background(), nil, 0)
	require.NoError(t, err)
	assert.Nil(t, outs)
}

func TestBatchRun_OrderAndValues(t *testing.T) {
	m := buildAddModel(t)
	vals := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	inputs := make([]map[string]*Tensor, len(vals))
	for i, v := range vals {
		inputs[i] = addFeed(v)
	}
	// concurrent=3 with 8 inputs exercises the multi-worker pool and the
	// in-order reassembly of results.
	outs, err := m.BatchRun(context.Background(), inputs, 3)
	require.NoError(t, err)
	require.Len(t, outs, len(vals))
	for i, out := range outs {
		y := out["y"]
		require.NotNil(t, y)
		f, err := y.AsFloat64()
		require.NoError(t, err)
		assert.InDelta(t, float64(vals[i])+10, f[0], 1e-6, "index %d", i)
	}
}

func TestBatchRun_ConcurrentOne(t *testing.T) {
	m := buildAddModel(t)
	inputs := []map[string]*Tensor{addFeed(1), addFeed(2), addFeed(3)}
	outs, err := m.BatchRun(context.Background(), inputs, 1)
	require.NoError(t, err)
	require.Len(t, outs, 3)
	f, _ := outs[2]["y"].AsFloat64()
	assert.InDelta(t, 13, f[0], 1e-6)
}

func TestBatchRun_DefaultConcurrency(t *testing.T) {
	m := buildAddModel(t)
	inputs := []map[string]*Tensor{addFeed(5), addFeed(6), addFeed(7), addFeed(8)}
	outs, err := m.BatchRun(context.Background(), inputs, 0)
	require.NoError(t, err)
	require.Len(t, outs, 4)
	f, _ := outs[0]["y"].AsFloat64()
	assert.InDelta(t, 15, f[0], 1e-6)
}

func TestBatchRun_ExcessConcurrencyClamped(t *testing.T) {
	m := buildAddModel(t)
	inputs := []map[string]*Tensor{addFeed(1), addFeed(2)}
	// concurrent far exceeds len(inputs); must be clamped, not fail.
	outs, err := m.BatchRun(context.Background(), inputs, 1000)
	require.NoError(t, err)
	require.Len(t, outs, 2)
}

func TestBatchRun_ErrorPropagates(t *testing.T) {
	m := buildAddModel(t)
	inputs := []map[string]*Tensor{
		addFeed(1),
		{}, // missing required input "x" -> Run error
		addFeed(3),
	}
	_, err := m.BatchRun(context.Background(), inputs, 2)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing graph input")
}

func BenchmarkBatchRun(b *testing.B) {
	m := buildAddModel(b)
	n := 64
	inputs := make([]map[string]*Tensor, n)
	for i := 0; i < n; i++ {
		inputs[i] = addFeed(float32(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.BatchRun(context.Background(), inputs, 0); err != nil {
			b.Fatal(err)
		}
	}
}
