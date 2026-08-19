# Recall — Architecture

This document explains how Recall is structured, the key interfaces that hold
it together, how data flows through it, and the design decisions behind the
major choices. For usage see [README.md](./README.md); for the contribution
workflow see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Overview

Recall is primarily a **Go library** for building Retrieval-Augmented
Generation (RAG) applications, plus an optional **service layer** (REST API,
CLI, configuration, deployment) built on top of the same library code.

Two hard constraints shape every package:

1. **Zero CGO.** Library code and all its dependencies must build with
   `CGO_ENABLED=0`. SQLite persistence uses the pure-Go `modernc.org/sqlite`
   driver; ONNX inference runs on a bundled pure-Go interpreter
   (`embedder/onnx`); S3 access uses a self-contained SigV4 signer; JWTs are
   verified with `crypto/hmac`.
2. **Dependency injection.** External capabilities (embedding models, LLMs)
   are injected as interfaces. Recall never hard-depends on a specific model
   provider, and every provider integration is an isolated, optional
   implementation of a small interface.

## Layer Map

```
┌────────────────────────────── Service layer ──────────────────────────────┐
│ cmd/recall (CLI)   cmd/recall-server   client (typed REST client)         │
│ app (service assembly from config)   config (files/env/hot reload)        │
│ api (stdlib REST server, auth, OpenAPI)                                   │
├────────────────────────────── Applications ───────────────────────────────┤
│ ingest (pipeline)  loader (files)  connector (web/git/S3/GitHub/SQL)      │
│ pipeline (RAG context assembly)  query (parse/expand/rewrite)             │
│ reranker (cross-encoder/sparse/LLM/LTR)  reasoning (multi-hop)            │
│ feedback (Rocchio)  eval (metrics)  hitl (review/annotation)              │
│ distributed (sharding/replication)                                        │
├────────────────────────────── Core engine ────────────────────────────────┤
│ store (Memory / SQLite / GraphStore / multimodal)                         │
│ index (brute-force, HNSW, quantization, hybrid, metadata, multi-vector)   │
│ chunker (fixed/recursive/semantic/parent-child/adaptive)                  │
│ embedder (interface + providers + pure-Go ONNX)  bm25  fuse  llm  cache   │
│ graph (knowledge graph)                                                   │
├────────────────────────────── Foundation ─────────────────────────────────┤
│ core (Chunk, Document, Value, errors)                                     │
├────────────────────────────── Cross-cutting ──────────────────────────────┤
│ metrics  tracing  analytics (observability by decoration)                 │
│ testutil (deterministic test fixtures — test imports only)                │
└────────────────────────────────────────────────────────────────────────────┘
```

Dependency rule: packages depend **downward** only. `core` imports nothing
from Recall; `store` depends on `core`, `index`, `chunker`, `embedder`,
`bm25`, `fuse`, `graph`; application packages compose `store` and never
import the service layer; `app`/`api`/`cmd/*` sit at the top and wire
everything together.

## Package Responsibilities

| Package | Role |
| ------- | ---- |
| `core` | Fundamental types: `Chunk`, `Document`, typed metadata `Value`, errors |
| `chunker` | Splitting documents into chunks (fixed, recursive, semantic, parent-child, document-aware, adaptive) |
| `embedder` | `Embedder` interface + Mock/OpenAI/Cohere/Ollama/ONNX providers, failover `Pipeline`, `CachingEmbedder`, multimodal |
| `embedder/onnx` | Minimal pure-Go ONNX loader/interpreter for BERT-style embedding models |
| `index` | Vector indexes: brute-force + HNSW, SQ8/PQ quantization, hybrid, metadata pre-filter, multi-vector; `SearchOptions`/filters |
| `bm25` | BM25 keyword ranking |
| `fuse` | Score fusion (weighted, RRF) |
| `store` | `Store` implementations (memory, SQLite), `GraphStore` (memory, SQLite), health checks, backup/migration/repair |
| `graph` | Knowledge graph: entities, relations, traversal, inference |
| `reasoning` | Multi-hop reasoning: inference rules, path exploration, confidence propagation |
| `pipeline` | RAG pipeline: retrieval → context assembly → prompt rendering; citations; multimodal |
| `query` | Query parsing (intent/entities), expansion, adaptive retrieval, LLM rewriters (HyDE, step-back, sub-query, multilingual) |
| `reranker` | Two-stage fine ranking: cross-encoder, sparse, LLM judge, ensemble, learning-to-rank, A/B experiments |
| `llm` | `Backend` interface + HTTP providers, streaming, resilience decorators (retry, circuit breaker, rate limit, timeout, fallback), extraction |
| `loader` | File readers: text, markdown, CSV, JSON, HTML, PDF, DOCX, directory |
| `connector` | External sources: web, git, S3, GitHub, SQL databases |
| `ingest` | Orchestration: load → dedup → validate → transform → upload; progress, batch, incremental |
| `cache` | LRU/TTL caches for queries, embeddings, graph traversals; multi-level |
| `distributed` | Consistent hashing, sharding, scatter-gather, replication, rebalance |
| `feedback` | Relevance feedback: Rocchio (vector + lexical), expand-and-retrieve |
| `eval` | Retrieval (P@K, R@K, MRR, NDCG@K) + answer-quality metrics, datasets, regression comparison |
| `hitl` | Review queue, annotations, active learning (uncertainty sampling) |
| `analytics` | Bounded query log, trending/drop-off analysis, export sinks |
| `metrics` | Counters/gauges/histograms, Prometheus text export, logging, metric bundles |
| `tracing` | OpenTelemetry-compatible spans/IDs, `traceparent` propagation, processors |
| `api` | stdlib REST server + auth (API key, scoped keys, JWT), embedded OpenAPI spec |
| `config` | JSON/YAML config, env overrides, validation, hot reload watcher |
| `app` | Builds embedder/store/pipeline/graph/reasoner/API server from `config.Config` |
| `client` | Typed HTTP client for the `api` endpoints |
| `cmd/recall` | CLI (local + server modes) |
| `cmd/recall-server` | Standalone REST service binary |
| `testutil` | Deterministic fixtures, mocks, golden files, benchmark harness (test-only) |

## Key Interfaces

These interfaces are the seams where implementations plug in:

```go
// store.Store — every storage backend satisfies this.
type Store interface {
    Upload(ctx, doc *core.Document, content string) error
    Search(ctx, query string, opts index.SearchOptions) ([]index.SearchResult, error)
    SearchHybrid(ctx, query string, opts index.SearchOptions) ([]index.SearchResult, error)
    GetChunk(id string) (*core.Chunk, bool)
    DeleteChunk(ctx, id string) error
    DeleteDocument(ctx, docID string) error
    Count() int
    Namespaces() []string
    Close() error
}

// embedder.Embedder — injected everywhere embeddings are needed.
type Embedder interface {
    Embed(ctx, text string) ([]float32, error)
    EmbedBatch(ctx, texts []string) ([][]float32, error)
    Dimension() int
}

// llm.Backend — chat completion with optional streaming.
type Backend interface {
    Chat(ctx, req *ChatRequest) (*ChatResponse, error)
    ChatStream(ctx, req *ChatRequest, fn func(chunk *StreamChunk) error) error
}
```

Other notable seams: `store.GraphStore` (entity/relation extraction + graph
queries), `chunker.Chunker` + `chunker.Factory`, `index.Index` (vector
index), `loader.Loader`, `connector.Connector`, `fuse.Fusion`,
`reranker.Reranker`, `reasoning.InferenceRule`, `graph.NERExtractor`, and
`api.Authenticator`. Interfaces are deliberately small and defined where they
are consumed (dependency inversion).

## Data Flow

### Ingestion

```
file / web / git / S3 / GitHub / SQL
        │  loader.Loader or connector.Connector
        ▼
ingest.Pipeline ── dedup (content hash) ── validation ── transform
        │
        ▼
store.Store.Upload
        ├── chunker.Chunker        → []*core.Chunk
        ├── embedder.Embedder      → vectors per chunk
        ├── index.Index            → vector index (+ HNSW at scale)
        ├── bm25.BM25              → keyword index (hybrid search)
        └── (optional) graph extraction → entities/relations
```

### Query / RAG

```
question
   │  (optional) query parsing/expansion/rewrite (query, llm)
   ▼
store.Search / SearchHybrid
   ├── metadata filters (index.Filter)
   ├── brute force or HNSW ANN
   └── BM25 + fusion (fuse.Fusion) for hybrid
   │  (optional) reranker.Reranker — coarse→fine two-stage
   ▼
pipeline.RAGPipeline — context assembly, token budget, citations
   ▼
rendered prompt (+ optional llm.Backend answer)
```

## Design Decisions

- **Pure Go everywhere.** Zero CGO makes cross-compilation trivial (the
  Docker image is `CGO_ENABLED=0` on distroless) and removes toolchain pain.
  Consequence: SQLite via `modernc.org/sqlite`, a hand-written ONNX
  interpreter limited to sentence-transformer operator sets, hand-rolled
  SigV4, stdlib-only HTTP/JWT.
- **Library first, service second.** Every service feature (API, CLI) is a
  thin wrapper over library APIs, so library users and service users share
  one implementation. `app` guarantees recall-server and the CLI wire
  components identically from the same `config.Config`.
- **Config structs, not option sprawl.** Constructors take a single `Config`
  struct with validated defaults (e.g. `store.Config`,
  `embedder.OpenAIConfig`). Structs keep constructors stable as fields are
  added.
- **Secrets never in config files.** Config names the *environment variable*
  holding a key (`api_key_env`); the value itself is only ever read from the
  environment at build time (`app.BuildEmbedder`).
- **Observability by decoration.** Metrics and tracing wrap `store.Store`
  (`metrics.InstrumentedStore`, `tracing.InstrumentedTracingStore`) rather
  than being baked into storage code, keeping the core path fast and
  uncluttered.
- **Deterministic test doubles.** `embedder.MockEmbedder` is content-derived
  (same text → same vector), `llm.MockBackend` is scriptable, and `testutil`
  fixtures have predictable chunk IDs. This keeps tests hermetic and the
  whole suite runnable offline.
- **Fail closed for security.** Namespace-scoped API keys deny by default;
  out-of-scope entities surface as 404s rather than partial data.
- **Namespaces stamp at upload.** Every chunk carries
  `core.MetadataKeyNamespace` stamped by the store at upload time, which is
  what makes scoped retrieval filters possible across backends.
- **Versioned SQL migrations in the binary.** Store schema migrations are
  plain SQL files with `-- recall-migration: <n>` headers, applied
  transactionally and idempotently (`recall store migrate`).

## Concurrency Model

- Stores, indexes, caches, graphs, and logs guard shared state with
  `sync.RWMutex` (read-heavy paths take read locks).
- `pipeline.RAGPipeline` is immutable after construction; request-scoped
  changes use the race-safe `Clone()`/`WithSearchFilters()`.
- `config.Config` is treated as immutable after `app.Build*`; hot reload
  rebuilds rather than mutating live components.
- The ingest pipeline processes documents concurrently when
  `Options.Concurrency > 1`; progress reporting is thread-safe.
- CI runs the full suite with `-race` on hosts that have a C toolchain.

## Persistence

- **SQLite schema** (`store`): documents, chunks, embeddings (BLOB), typed
  metadata, BM25 state, plus graph tables (`entities`, `relations`) for
  `SQLiteGraphStore`. Typed `core.Value` metadata round-trips losslessly.
- **Backups** use online `VACUUM INTO`; **restore** is atomic temp-file +
  rename; **integrity** checks and corruption repair utilities ship with the
  store package and are exposed via the CLI.
- The in-memory backend (`MemoryStore`) implements identical semantics and is
  the default for tests and local smoke runs.

## Deployment Topology

Single-node service: `recall-server` (+ config file) behind your ingress;
data in a mounted SQLite volume (see `deploy/`). CLI clients talk to it with
`recall --server URL`. Multi-node: `distributed.Cluster` shards stores by
consistent hashing with configurable replication and scatter-gather search;
`recall cluster status` probes node `/diagnostics`. Container images are
multi-stage distroless with `HEALTHCHECK` via `recall-server -health-probe`;
Kubernetes manifests include probes, HPA, and ConfigMap/Secret wiring
(`deploy/kubernetes/`).

