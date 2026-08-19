# embedder/onnx

A small, dependency-free ONNX runtime (pure Go, zero CGO) for running local
embedding models — the same runtime `embedder.OnnxEmbedder` uses.

## What it implements

- **Model loading** (`Model`, `Node`, operator graph) from ONNX protobuf
  files.
- **Tensors** (`Tensor`, `DataType`) with shape/dtype validation and
  `AsFloat64` conversion.
- **Operators** needed by sentence-encoder models: matmul, add, sub, mul,
  div, reshape, transpose, reductions, layer/group/batch norm, activations
  (gelu, relu, sigmoid, softmax, ...), embedding/onehot, concat, split,
  etc. (see `ops.go`).

This is a targeted runtime for transformer encoder models, not a general
ONNX specification implementation; unsupported operators fail fast with a
named error.

## Used by

`embedder.NewOnnxEmbedder` loads a tokenizer + `model.onnx` (bundled
`all-MiniLM-L6-v2` or a Hugging Face repo via `embedder.LoadHFModel`) and
runs inference through this package.
