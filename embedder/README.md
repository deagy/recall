# embedder

The `Embedder` interface and its implementations. Stores and pipelines
receive an `Embedder` by dependency injection — recall itself never calls an
LLM API directly.

```go
type Embedder interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimension() int
}
```

## Implementations

| Embedder | Constructor | Notes |
|----------|-------------|-------|
| `MockEmbedder` | `NewMockEmbedder(dim)` | Deterministic hash-based vectors; for tests and offline examples. |
| `OpenAIEmbedder` | `NewOpenAIEmbedder(cfg)` | `text-embedding-3-*` models; `Dimension` validated against the model. |
| `CohereEmbedder` | `NewCohereEmbedder(cfg)` | `embed-english-v3.0` / `embed-multilingual-v3.0`. |
| `OllamaEmbedder` | `NewOllamaEmbedder(cfg)` | Local Ollama server. |
| `OnnxEmbedder` | `NewOnnxEmbedder(cfg)` / `NewOnnxEmbedderFile(path, cfg)` | Local ONNX inference (see `embedder/onnx`), no CGO. |
| `CachingEmbedder` | `NewCachingEmbedder(inner, cache, ttl)` | Wraps any embedder with an LRU+TTL cache (`cache` package). |

`MultiModalEmbedder` + `MockMultiModal` cover image+text embedding for the
multimodal store. Helpers: `CosineSimilarity`, `AutoDimension`, and
Hugging Face model resolution (`LoadHFModel`, `ModelCache`) for ONNX models.

## API keys

Provider embedders take the key in their config struct; the `config` /
`app` layers read it from the environment variable named by
`EmbedderConfig.APIKeyEnv`. Keys are never stored in config files.
