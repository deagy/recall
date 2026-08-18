package onnx

import (
	"context"
	"fmt"
	"os"
)

// LoadFile reads a model file from disk and parses it.
func LoadFile(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: failed to read model %q: %w", path, err)
	}
	return Load(data)
}

// Run executes the model with the given inputs (named after the graph's
// feed inputs) and returns the graph outputs.
func (m *Model) Run(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
	x := newExecutor(m.Graph)
	return x.run(ctx, inputs)
}

// FeedInputs returns the tensors the caller must provide to Run.
func (m *Model) FeedInputs() []NamedType { return m.Graph.FeedInputs() }

// Outputs returns the graph's declared output tensors.
func (m *Model) Outputs() []NamedType { return m.Graph.Outputs }
