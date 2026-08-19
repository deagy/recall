# Examples

Runnable, self-contained programs showing how to use recall. Every example is
**deterministic and offline**: they use `embedder.NewMockEmbedder` and
`llm.NewMockBackend`, so they need no API keys, network access, or
configuration. Run any of them from the repo root:

```sh
go run ./example          # quick tour: upload, search, hybrid, graph, graph-RAG
go run ./example/e2e      # full lifecycle: ingest -> search -> RAG -> eval -> reasoning
go run ./example/production  # API server deployment driven by the typed client
```

## What each example covers

| Program | Demonstrates |
|---------|--------------|
| [`main.go`](./main.go) | The 30-second tour: `store.MemoryStore` upload + vector search, hybrid search, `graph.KnowledgeGraph` (entities, relations, path finding), and graph-based extraction via `store.NewMemoryGraphStore`. |
| [`e2e/main.go`](./e2e/main.go) | The end-to-end RAG tutorial: directory-loader ingestion through `ingest.Pipeline` (dedup + progress) into a SQLite store, vector vs. hybrid search, the RAG pipeline with citations answered by a mock LLM, retrieval evaluation (Precision/Recall/MRR/NDCG@K) with `eval.BenchmarkSuite`, and knowledge-graph extraction + multi-hop reasoning. |
| [`production/main.go`](./production/main.go) | Service deployment: `app.BuildAPIServer` (the same assembly the `recall-server` binary uses — SQLite store, RAG pipeline, graph, reasoner) served over HTTP on an ephemeral port, driven entirely through the typed `client` package (`Health`, `Upload`, `Search`, `RAG`, `Diagnostics`), with graceful shutdown. |

## From examples to production

- **Real embeddings**: replace `embedder.NewMockEmbedder(384)` with
  `embedder.NewOpenAIEmbedder`, `NewCohereEmbedder`, `NewOllamaEmbedder`, or
  the local ONNX embedder. API keys come from environment variables — never
  hardcode them.
- **Real LLMs**: replace `llm.NewMockBackend` with `llm.NewOpenAIClient` or
  `llm.NewOllamaClient`; wrap with `llm.NewRetryBackend` /
  `NewRateLimitBackend` / `NewCircuitBreakerBackend` for resilience.
- **Real servers**: run the standalone `recall-server -config recall.yaml`
  (see [SECURITY.md](../SECURITY.md) for API-key auth) and point the
  `client` (or the `recall` CLI) at it.

## Benchmark comparison

For index/algorithm comparisons see [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)
and `scripts/benchcompare.sh`, which runs the benchmark suite and diffs
against a stored baseline.
