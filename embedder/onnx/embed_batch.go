package onnx

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

// defaultBatchConcurrency returns the default worker count for parallel
// batch execution: the number of logical CPUs, capped at 8 (memory use
// grows linearly with concurrency, since each sequence holds its full
// intermediate tensor state).
func defaultBatchConcurrency() int {
	n := runtime.NumCPU()
	if n < 1 {
		n = 1
	}
	if n > 8 {
		n = 8
	}
	return n
}

// BatchRun runs the graph once per input set, executing up to `concurrent`
// sequences in parallel. Each sequence gets its own executor environment,
// and the graph's initializers are shared read-only (the interpreter never
// mutates initializers). concurrent <= 0 uses a sensible default.
//
// Results are returned in input order. When ctx is cancelled, sequences
// that have not started return the context error; in-flight sequences are
// allowed to finish so goroutines do not leak.
func (m *Model) BatchRun(ctx context.Context, inputs []map[string]*Tensor, concurrent int) ([]map[string]*Tensor, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	if concurrent <= 0 {
		concurrent = defaultBatchConcurrency()
	}
	if concurrent > len(inputs) {
		concurrent = len(inputs)
	}
	type result struct {
		index int
		out   map[string]*Tensor
		err   error
	}
	ch := make(chan result, len(inputs))
	var next int32
	var workers int
	if concurrent > 1 {
		workers = concurrent
	} else {
		workers = 1
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(atomic.AddInt32(&next, 1)) - 1
				if i >= len(inputs) {
					return
				}
				out, err := m.Run(ctx, inputs[i])
				if err != nil {
					ch <- result{index: i, err: err}
					return
				}
				ch <- result{index: i, out: out}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(ch)
	}()
	outs := make([]map[string]*Tensor, len(inputs))
	var firstErr error
	for r := range ch {
		if firstErr == nil && r.err != nil {
			firstErr = fmt.Errorf("onnx: batch item %d: %w", r.index, r.err)
		}
		if r.out != nil {
			outs[r.index] = r.out
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	for i, out := range outs {
		if out == nil {
			return nil, fmt.Errorf("onnx: batch item %d produced no output", i)
		}
	}
	return outs, nil
}
