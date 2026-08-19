# chunker

Strategies for splitting documents into chunks. All chunkers implement the
`Chunker` interface (`Chunk(doc, content) ([]*core.Chunk, error)`) and are
created through `Factory` functions so stores can be configured polymorphically.

## Chunkers

| Chunker | Constructor | Notes |
|---------|-------------|-------|
| `FixedChunker` | `NewFixed(cfg)` | Fixed character size with overlap; the store default. |
| `RecursiveChunker` | `NewRecursive(cfg)` | Splits on structural boundaries (paragraphs, sentences). |
| `SemanticChunker` | `NewSemantic(embedder, cfg)` | Splits where embedding similarity drops (needs an embedder). |
| `AdaptiveChunker` | `NewAdaptive(cfg)` | Chooses chunk size adaptively. |
| `ParentChildChunker` | `NewParentChild(parent, child)` | Retrieves on small child chunks, returns the parent context. |
| `StreamingChunker` | `NewStreaming(inner, embedder, cfg)` | Bounded-memory chunking for streams. |
| `DocumentAwareChunker` | `NewDocumentAware(inner)` | Wraps any chunker with document-aware boundaries. |

`Config` (see `DefaultConfig()`) controls size, overlap, and boundaries.
Coherence helpers (`ChunkCoherence`, `AnalyzeChunk`) score how well a chunk
reads as a unit.

## Minimal example

```go
chunker := chunker.NewRecursive(chunker.DefaultConfig())
chunks, err := chunker.Chunk(doc, content)
```

## Used by

`store` (via `Config.ChunkerFactory`) and `ingest`.
