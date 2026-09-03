# Recall — Development Roadmap

This document outlines the future development roadmap for the Recall library, organized by priority tiers and target phases. The current codebase has **22 completed phases** covering core RAG functionality (storage, search, chunking, knowledge graphs, reasoning, distributed storage, caching, LLM integration, document ingestion, and advanced retrieval).

---

## Current State Assessment

| Metric | Status |
|--------|--------|
| Completed Phases | 25 (Phases 1–22, 26, 29, 32) |
| Packages | 30 library packages + 2 commands + 1 example |
| All Tests Pass | Yes |
| Overall Coverage Target | >80% in all 30 library packages (re-measured 2026-08-19; `cmd/*` + `example/*` mains exempt) |
| Zero CGO | Maintained |
| Lint | `golangci-lint` 0 issues (Phase 32, 2026-08-19) |
| LLM Backends | OpenAI, Ollama (Mock) |
| Embedding Providers | OpenAI, Cohere, Ollama, ONNX (local), Mock |
| Fusion Methods | WeightedFusion, RRF |

### Coverage by Package (re-measured 2026-08-18 — all above target)

| Package | Coverage | Package | Coverage |
|---------|----------|---------|----------|
| `core/` | 100.0% | `pipeline/` | 97.4% |
| `cache/` | 97.8% | `reasoning/` | 96.4% |
| `fuse/` | 97.6% | `bm25/` | 96.6% |
| `embedder/` | 91.6% | `ingest/` | 92.7% |
| `index/` | 91.9% | `loader/` | 90.3% |
| `distributed/` | 90.8% | `connector/` | 87.5% |
| `graph/` | 87.3% | `llm/` | 86.8% |
| `store/` | 86.6% | `query/` | 86.3% |
| `chunker/` | 86.1% | `embedder/onnx/` | 80.2% |
| `reranker/` | 89.5% |  |  |

---

## Phase 19: Test Coverage Hardening

**Status: Complete (2026-08-18)** — all 14 packages are above the 80% target
(see Current State Assessment). The checklists below are the work that got them
there.

**Goal:** Bring all packages to ≥80% test coverage.

### 19.1 LLM Package Tests (`llm/`)
- [x] Test `OpenAIClient.Chat()` with mocked HTTP responses (success, error, timeout)
- [x] Test `OpenAIClient.ChatStream()` chunk processing
- [x] Test `OllamaClient.Chat()` and `ChatStream()`
- [x] Test `LLMExtractor.ExtractEntities()` with valid/invalid JSON
- [x] Test `LLMExtractor.ExtractRelations()` edge cases
- [x] Test `MockBackend` with various response scenarios
- [x] Test `ResponseFormat` with JSON schema
- [x] Test `ChatRequest` validation (missing model, invalid temperature)

### 19.2 Store Package Tests (`store/`)
- [x] Test `SQLiteStore` migration and schema evolution
- [x] Test `SQLiteStore` concurrent access (RWMutex correctness)
- [x] Test `SQLiteGraphStore` persistence round-trip with large graphs
- [x] Test `MemoryStore` namespace isolation under concurrency
- [x] Test `GraphStore` extraction with empty/nil inputs
- [x] Test `SQLiteStore` recovery after unexpected close
- [x] Test `MemoryStore` delete cascading (chunk → document)

### 19.3 Distributed Package Tests (`distributed/`)
- [x] Test consistent hashing with node add/remove (hash ring stability)
- [x] Test scatter-gather with partial node failures
- [x] Test replication strategies under concurrent writes
- [x] Test shard rebalancing when nodes join/leave
- [x] Test `DistributedStore` with single-node cluster

### 19.4 Core Package Tests (`core/`)
- [x] Test `Value` type serialization/deserialization edge cases
- [x] Test `Document` equality and comparison
- [x] Test `Chunk` metadata manipulation
- [x] Test custom error wrapping and unwrapping

### 19.5 Index Package Tests (`index/`)
- [x] Test HNSW graph construction with edge cases (duplicate IDs, empty graphs)
- [x] Test filter combinations (term + range + date range)
- [x] Test memory index with mixed metadata types
- [x] Test HNSW parameter sensitivity (M, efConstruction, efSearch)

### 19.6 Reasoning Package Tests (`reasoning/`)
- [x] Test inference rules with cyclic graphs
- [x] Test confidence propagation with deep paths
- [x] Test natural language query parsing edge cases
- [x] Test entity extraction with ambiguous text

### 19.7 Chunker Package Tests (`chunker/`)
- [x] Test semantic chunker with edge cases (single sentence, empty text)
- [x] Test streaming chunker with incremental input
- [x] Test chunk quality metrics with degenerate inputs

**Estimated Effort:** 3–4 weeks

---

## Phase 20: Real Embedding Providers

**Goal:** Support production embedding models beyond the mock embedder.

### 20.1 OpenAI Embeddings
- [x] `OpenAIEmbedder` implementing `embedder.Embedder`
- [x] Support for `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- [x] Batch embedding with automatic batching
- [x] Rate limiting and retry with exponential backoff
- [x] Dimension validation and caching

### 20.2 Local Embedding Models (ONNX Runtime)
- [x] `OnnxEmbedder` using pure-Go ONNX inference — `embedder/onnx/` (hand-rolled protobuf wire codec, tensor runtime, ~50-operator executor covering the sentence-transformer operator family) + `embedder/onnx_embedder.go` adapter implementing `embedder.Embedder`, wired into the `Pipeline` failover chain; tokenization is dependency-injected via `TokenizerFunc`
- [x] First-party support for `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text-v1.5` — bundled fully-offline WordPiece tokenizer (`embedder/tokenizer_wordpiece.go`) with the shared BERT-uncased vocab embedded (zero CGO, zero network), a `BundledTokenizer(modelName, model)` registry, and per-model configs (lowercasing, max length, CLS/SEP, padding)
- [x] CPU optimization — float32 `MatMul` fast path (no float64 round-trip on the hottest kernel), `Model.BatchRun` worker-pool that executes sequences in parallel over shared read-only initializers, and `OnnxEmbedder.BatchConcurrency` (0 = auto from `runtime.NumCPU()`, capped at 8)
- [x] Model download and caching — `embedder.ModelCache` (SHA-256-keyed on-disk cache, atomic temp-file writes, TTL, concurrent-fetch dedup) + `LoadHFModel` (HuggingFace resolver URL, configurable base URL for mirrors/offline, cache-aware)

### 20.3 Cohere Embeddings
- [x] `CohereEmbedder` for `embed-english-v3.0`, `embed-multilingual-v3.0`
- [x] Input type support (search_document, search_query)
- [x] Truncation strategies

### 20.4 Hugging Face Transformers (Go)
- [x] `OnnxEmbedder` runs sentence-transformers ONNX exports directly (pure-Go, no CGO); HF `safetensors`/GPTQ/AWQ-native loading out of scope
- [x] Support for sentence-transformers models (via ONNX export + injected tokenizer)
- [ ] Quantized model support (GPTQ, AWQ)

### 20.5 Embedding Pipeline
- [x] `embedder.Pipeline` for chaining embedders (e.g., retry on different provider)
- [x] `embedder.CachingEmbedder` wrapper (integrate with cache package)
- [x] Embedding dimension auto-detection

**Estimated Effort:** 4–5 weeks
**Priority:** High (critical for production use)

---

## Phase 21: Document Ingestion Pipeline

**Goal:** Support ingesting documents from various formats and sources.

### 21.1 Document Loaders
- [x] `loader.TextLoader` — plain text files (optional max-bytes cap)
- [x] `loader.MarkdownLoader` — Markdown with heading-based chunking (ATX sections, breadcrumb metadata, slug IDs)
- [x] `loader.HTMLLoader` — HTML with content extraction (pure-Go `golang.org/x/net/html`; drops script/style/nav/header/footer, block-aware line breaks, title capture)
- [x] `loader.PDFLoader` — PDF plain-text extraction (pure-Go `ledongthuc/pdf`; page count in metadata; encrypted/scanned files yield errors)
- [x] `loader.DocxLoader` — DOCX extraction (stdlib only: `archive/zip` + `encoding/xml` over `word/document.xml`)
- [x] `loader.CSVLoader` — CSV with configurable column mapping (header/separator/ID/content columns)
- [x] `loader.JSONLoader` — JSON with nested extraction (dotted field paths, object or array)
- [x] `loader.DirectoryLoader` — recursive file system scanning (per-extension dispatch, partial-failure reporting)

### 21.2 Source Connectors
- [x] `connector.WebConnector` — fetch URLs with rate limiting (token-bucket pacing, content-type filter, byte cap, HTML text extraction)
- [x] `connector.GitConnector` — clone and index git repositories (shallow/full, git CLI via exec, temp-clone cleanup)
- [x] `connector.S3Connector` — S3 bucket indexing (self-contained SigV4 signer — no AWS SDK — virtual + path style, prefix listing, MinIO-compatible)
- [x] `connector.GitHubConnector` — GitHub repo/issue indexing (raw README + issues via REST, PRs filtered, optional token)
- [x] `connector.DatabaseConnector` — SQL database table indexing (injected `*sql.DB`, column mapping, driver-agnostic)

### 21.3 Ingestion Pipeline
- [x] `ingest.Pipeline` orchestrating load → filter → transform → upload (chunk + embed + index via the store)
- [x] `ingest.Deduplicator` — content-hash based duplicate detection (persistable JSON)
- [x] `ingest.Validator` — schema validation for structured documents (size bounds, required metadata, source prefixes)
- [x] `ingest.Progress` — thread-safe progress tracking with per-document and per-phase callbacks
- [x] `ingest.RunBatch` — parallel batch ingestion across sources with configurable concurrency
- [x] `ingest.Incremental` — delta ingestion (only new/changed documents, persisted state)

**Estimated Effort:** 4–5 weeks
**Priority:** Medium-High

---

## Phase 22: Reranking

**Goal:** Add cross-encoder reranking for improved retrieval quality.

### 22.1 Cross-Encoder Rerankers
- [x] `reranker.CrossEncoderReranker` — lightweight cross-encoder (pure Go ONNX)
- [x] `reranker.SparseReranker` — BM25-based re-scoring
- [x] `reranker.LLMReranker` — LLM-as-judge reranking
- [x] `reranker.EnsembleReranker` — combine multiple rerankers

### 22.2 Reranking Pipeline Integration
- [x] `pipeline.Reranker` interface
- [x] `pipeline.RAGPipeline.WithReranker()` builder
- [x] Two-stage retrieval: coarse (vector) → fine (rerank)
- [x] Configurable top-K at each stage
- [x] Reranking score attribution in `SearchResult`

### 22.3 Learning-to-Rank (Future)
- [x] `reranker.LTRanker` — simple pointwise LTR model
- [x] `reranker.AdaptiveLTRanker` — feedback-driven reranker adaptation (auto-refit at threshold)
- [x] `reranker.Experiment` — A/B testing framework for reranker comparison (NDCG@K, MRR@K, Precision@K, win rate, Welch t-test)

**Estimated Effort:** 3–4 weeks
**Priority:** Medium-High (significant quality improvement)

---

## Phase 23: Observability & Monitoring

**Goal:** Add production-grade observability.

### 23.1 Metrics
- [x] `metrics.StoreMetrics` — search latency (p50/p95/p99), throughput, error rates
- [x] `metrics.EmbeddingMetrics` — embedding latency, dimension stats
- [x] `metrics.CacheMetrics` — hit/miss ratio, eviction count
- [x] `metrics.GraphMetrics` — traversal depth, inference counts
- [x] Export to Prometheus format (`Registry.HTTPHandler()` for a `/metrics` endpoint)
- [x] Structured logging with correlation IDs (`metrics.Logger`)
- [x] `metrics.InstrumentedStore` — drop-in `store.Store` wrapper that records store metrics + structured logs

### 23.2 Tracing
- [x] OpenTelemetry-compatible tracing (128-bit trace / 64-bit span IDs, span kinds, attributes, status; `SpanProcessor` is the bridge to an OTel collector — the heavy OTel SDK is intentionally not bundled to preserve zero-dependency/zero-CGO)
- [x] W3C `traceparent` inject/parse + `StartRemoteSpan` for distributed (cross-node) trace correlation
- [x] Trace spans for store operations (search, hybrid search, upload) with metadata span tags (namespace, document ID, query, query type, top-k, result count)
- [x] Store-level tracing: `InstrumentedTracingStore` records the upload → search → retrieve path as parent/child spans
- [x] Pluggable `SpanProcessor`s: `InMemoryProcessor` (grouped by trace) and `ConsoleProcessor`

### 23.3 Health & Diagnostics
- [x] `store.HealthCheck()` — connectivity probe, chunk count, namespaces, and (for SQLite) structural integrity; returns a `store.HealthReport`
- [x] `store.IntegrityCheck()` — verify data consistency (added in Phase 25.2; surfaced by `HealthCheck`)
- [x] `distributed.ClusterHealth()` — node status + overall operability (Phase 25.3), extended with `ShardStats`/`ShardDistribution` for per-node shard distribution
- [x] Expose diagnostics via endpoint: `store.HealthHandler` and `distributed.HealthHandler` serve `/healthz` (200/503) and `/diagnostics` (JSON)

### 23.4 Query Analytics
- [x] `analytics.QueryLog` — bounded, thread-safe log of queries with latency and results (ring buffer)
- [x] `analytics.PopularQueries` — trending query detection (frequency + avg latency), case/whitespace-normalized
- [x] `analytics.DropOff` — queries with no good results (errored, zero results, or top score below threshold)
- [x] Export to configurable sinks: `Sink` interface + `FileSink` (NDJSON) and `HTTPSink` (POST to a collector/queue bridge); a message-queue sink plugs in via the same interface
- [x] `analytics.InstrumentedAnalyticsStore` — drop-in `store.Store` wrapper that records query records

**Estimated Effort:** 3–4 weeks
**Priority:** Medium

---

## Phase 24: Feedback Loop & Evaluation

**Goal:** Enable continuous improvement of retrieval quality.

### 24.1 Relevance Feedback
- [x] `feedback.RelevanceFeedback` — adjusts a query based on user feedback (Rocchio), with `BoostRelevant` re-ranking
- [x] `feedback.Rocchio` — classic Rocchio query expansion in **both** vector form (`Q' = αQ + β·mean(relevant) − γ·mean(irrelevant)`) and lexical/term form (returns an expanded query string)
- [x] `feedback.ExpandAndRetrieve` — retrieve → user marks relevant → re-rank (embed → gather feedback embeddings → Rocchio → re-search → boost relevant)
- [x] Store feedback in metadata for future training — `feedback.Collector` (thread-safe training store) + `Feedback.ToMetadata()`

### 24.2 Evaluation Framework
- [x] `eval.RetrievalEval` — Precision@K, Recall@K, MRR, NDCG@K (pure functions + `ComputeRetrievalMetrics`, graded or binary relevance)
- [x] `eval.RAGEval` — answer quality (faithfulness, relevance, correctness) via a pluggable `Judge` interface; deterministic `OverlapJudge` included, LLM judge pluggable
- [x] `eval.BenchmarkSuite` — regression testing: `Run` / `RunWithAnswers` over a `RetrievalSystem`, plus `Compare` (baseline vs. current with tolerance → regressions/improvements)
- [x] `eval.Dataset` — JSON load/save of evaluation datasets (`EvalQuery`: relevant IDs, graded relevance, context, reference answer)
- [x] `eval.Report` — aggregate report with JSON/Markdown output; `SaveJSON`/`LoadReport` enable golden-file regression checks in CI

### 24.3 Human-in-the-Loop
- [x] `hitl.ReviewQueue` — thread-safe, de-duplicated priority queue of chunks awaiting human review (highest uncertainty first), with approve/reject lifecycle
- [x] `hitl.Annotation` + `hitl.AnnotationStore` — human annotations (relevance/correction/feedback) stored by value, indexed by chunk and ID, with most-recent-relevance lookup
- [x] `hitl.ActiveLearning` — uncertainty prioritization: `UncertaintyFromScores` (least-confidence), `Margin` (top-1/top-2 gap), and `Select` (enqueue top-N uncertain candidates, skipping already-reviewed)
- [ ] Web UI for annotation (optional, Phase 30)

### 24.4 Automated Testing
- [x] `testutil.FixtureStore` — in-memory store with preloaded data (deterministic mock embedder, predictable chunk IDs, single-chunk chunking)
- [x] `testutil.MockLLM` — deterministic LLM responses for testing (scripted `llm.Backend`: ordered responses, last-one-repeat, streaming, call tracking)
- [x] `testutil.GoldenFile` — compare results against golden files (`Golden`/`GoldenJSON` with an `UpdateGolden` refresh mode, `GoldenDiff`)
- [x] `testutil.MockEmbedder` — deterministic test embedder (re-export of `embedder.MockEmbedder` + `DeterministicEmbed`)
- [x] `testutil.BenchmarkHarness` — benchmarking wrapper (warmup + correct `ResetTimer`/`StopTimer` handling, error propagation)
- [x] CI integration for regression testing — existing workflow (vet, `gofmt -s`, build, `-race` tests, benchmark smoke) plus a new overall-coverage gate (≥ 80%)
- [x] Fixed pre-existing data race in `ingest.Pipeline.process` (unlocked `Result.Skipped++`/`Result.Uploaded++` in the worker pool)

**Estimated Effort:** 3–4 weeks
**Priority:** Medium

---

## Phase 25: Resilience & Reliability

**Status: Complete (2026-08-18)** — all four sub-phases implemented and tested (see checklist below).

**Goal:** Add production reliability features.

### 25.1 LLM Backend Resilience
- [x] `llm.RetryBackend` — retry with exponential backoff (+ jitter, pluggable retryable predicate, context-aware)
- [x] `llm.CircuitBreakerBackend` — trip on failure threshold (closed/open/half-open, cooldown + single probe, `State()`)
- [x] `llm.RateLimitBackend` — token bucket rate limiting (capacity + refill rate, blocking acquire, context-aware)
- [x] `llm.FallbackBackend` — fallback to secondary provider (ordered failover, aggregates last error)
- [x] `llm.Middleware` — chain backends with interceptors (`MiddlewareFunc` composer + `RetryMiddleware`/`CircuitBreakerMiddleware`/`RateLimitMiddleware`/`FallbackMiddleware`)

### 25.2 Store Resilience
- [x] `store.Checkpoint` — periodic SQLite WAL checkpoints (`SQLiteStore.Checkpoint` with PASSIVE/FULL/TRUNCATE/RESTART + `StartAutoCheckpoint` background loop stopped on `Close`)
- [x] `store.Backup` — point-in-time backup and restore (`SQLiteStore.Backup` via online `VACUUM INTO` + `RestoreSQLite` atomic file restore)
- [x] `store.Migration` — schema versioning and automatic migration (`Migration` + `Migrator` tracked via `PRAGMA user_version` + `schema_migrations`, transactional, auto-applied from `Config.Migrations`)
- [x] `store.CorruptionDetection` — detect and repair corrupted data (`SQLiteStore.IntegrityCheck` via `integrity_check` + `foreign_key_check`, `Repair` rebuilds the FTS index)

### 25.3 Distributed Resilience
- [x] `distributed.NodeHealth` — periodic health checks (`NodeHealth` background prober; marks nodes offline after N consecutive failed probes, recovers on success)
- [x] `distributed.AutoRebalance` — automatic shard rebalancing (`AutoRebalancer` rebuilds the ring via `Cluster.RebalanceActive` when the active node set changes)
- [x] `distributed.FaultTolerance` — operate with degraded nodes (`Cluster.Health()` healthy/degraded/down summary, `Quorum`/`QuorumMet`, `ReplicateOp` quorum-based write replication)
- [x] `distributed.Consensus` — leader election for writes (`Consensus` deterministic leader among online nodes with a monotonically increasing `Term`)

### 25.4 Context Management
- [x] `pipeline.SmartContextWindow` — priority-based chunk inclusion (`SmartContextWindow.Select`: highest-score chunks first within the token budget; opt-in via `RAGPipeline.WithSmartContext`)
- [x] `pipeline.ContextCompression` — summarize long contexts (`ContextCompressor` + pluggable `Summarizer` + deterministic `ExtractiveSummarizer`)
- [x] `pipeline.CitationTracking` — track which chunk contributed what (`TrackCitations`/`RenderCitations`; opt-in via `RAGPipeline.WithCitations` populating `RAGResponse.Citations`)
- [x] `pipeline.HallucinationDetection` — verify claims against sources (`HallucinationDetector.Check`/`HallucinationRate`, lexical claim-support)

**Estimated Effort:** 3–4 weeks
**Priority:** Medium

---

## Phase 26: Advanced Retrieval

**Status: Complete (2026-08-18)** — all four sub-phases implemented and tested (see checklist below).

**Goal:** State-of-the-art retrieval techniques.

### 26.1 Advanced Indexing
- [x] `index.ScalarQuantization` — 8-bit SQ for memory efficiency (`ScalarQuantizer` + `QuantizedIndex`)
- [x] `index.ProductQuantization` — PQ for large-scale ANN (`ProductQuantizer` with k-means++ codebooks, ADC search via `PQIndex`)
- [x] `index.HybridIndex` — combine HNSW + BM25 in single index (vector + BM25 + pluggable fusion, keyword-only hits retained)
- [x] `index.MetadataIndex` — fast metadata-based filtering index (inverted postings, `Candidates` pre-filter)
- [x] `index.MultiVector` — multiple embeddings per chunk (query + passage; MaxSim / mean / top-mean aggregation)

### 26.2 Query Techniques
- [x] `query.Rewrite` — LLM-powered query rewriting (`Rewriter`)
- [x] `query.HyDE` — Hypothetical Document Embeddings (`HyDE`)
- [x] `query.StepBack` — step-back prompting for better retrieval (`StepBack`)
- [x] `query.SubQuery` — decompose complex queries (`SubQueryDecomposer`: LLM-first with heuristic fallback)
- [x] `query.Multilingual` — multilingual query support (`DetectLanguage` script heuristics + `Multilingual` multi-query expansion via pluggable `Translator`)

### 26.3 Chunking Advances
- [x] `chunker.ParentChild` — parent chunk retrieval with child detail (`ParentChildChunker` with parent cache + `ExpandChunks`)
- [x] `chunker.DocumentAware` — respect document boundaries (`DocumentAwareChunker`, no cross-boundary chunks/overlap)
- [x] `chunker.Adaptive` — auto-tune chunk size based on content (`AdaptiveChunker`, sentence-length driven)

### 26.4 Multi-Modal
- [x] `embedder.MultiModalEmbedder` — text + image embeddings (interface + deterministic `MockMultiModal`)
- [x] `store.MultiModalStore` — store and retrieve across modalities (cross-modal `SearchText`/`SearchImage`)
- [x] `pipeline.MultiModalPipeline` — multi-modal RAG (mixed text/image context assembly, optional LLM backend)

**Estimated Effort:** 5–6 weeks
**Priority:** Medium-Low

---

## Phase 27: API & Service Layer

**Goal:** Expose Recall as a service.

### 27.1 REST API
- [x] `api.Server` — HTTP server using standard library
- [x] `POST /upload` — upload documents
- [x] `GET /search` — vector search
- [x] `POST /hybrid-search` — hybrid search
- [x] `POST /rag` — full RAG pipeline query
- [x] `GET /graph/{entity}` — graph entity lookup
- [x] `POST /graph/reason` — graph reasoning
- [x] OpenAPI/Swagger specification
- [x] API authentication (API keys, JWT)

> **27.1 complete** — the `api` package exposes `Server`/`Handler` built on the
> standard library `net/http` (Go 1.22+ `ServeMux` method+path routing).
> Operational endpoints `/healthz`, `/readyz`, and `/diagnostics` are
> unauthenticated by default; data endpoints are protected by an optional
> `Authenticator` (static API keys via `X-API-Key`/Bearer, or HS256 JWTs
> verified with stdlib `crypto/hmac` — no external auth dependency). The
> OpenAPI 3.0 document is embedded with `go:embed` and served at
> `GET /openapi.json`. `cmd/recall-server` is the runnable entrypoint with
> graceful SIGINT/SIGTERM shutdown.

### 27.2 gRPC (Future)
- [ ] Protocol buffer definitions
- [ ] gRPC server with streaming support
- [ ] Bidirectional streaming for RAG responses

### 27.3 Configuration
- [x] YAML/JSON configuration file support
- [x] Environment variable overrides
- [x] Configuration validation
- [x] Hot-reload support

> **27.3 complete** — the `config` package loads JSON or YAML (by extension),
> applies defaults, and validates (combined multi-problem errors). Environment
> overrides use `RECALL__SECTION__KEY` (double-underscore nesting, e.g.
> `RECALL__SERVER__PORT`); malformed values are ignored so a typo never blocks
> startup. `config.Watcher` polls the file and hot-reloads via a callback,
> skipping (and remembering via `LastError`) invalid edits so a bad change
> never takes a running service down.

### 27.4 Deployment
- [x] Dockerfile
- [x] docker-compose for multi-node deployment
- [x] Kubernetes manifests (Deployment, Service, HPA)
- [x] Health check endpoints for orchestrators

> **27.4 complete** — `deploy/Dockerfile` is a multi-stage, `CGO_ENABLED=0`
> pure-Go build on a distroless/nonroot base (no C toolchain needed). Because
> distroless ships no `curl`, `recall-server` gains a `-health-probe` mode for
> the in-image `HEALTHCHECK`. `deploy/docker-compose.yml` runs a single node
> with a mounted config + SQLite volume and documents a multi-node template;
> `deploy/kubernetes/recall.yaml` provides a ConfigMap, Secret, Deployment
> (readiness `/readyz` + liveness `/healthz` probes), Service, and HPA.
> Example configs live in `deploy/config/` and are guarded by a config-package
> regression test.

**Estimated Effort:** 4–5 weeks
**Priority:** Medium-Low

---

## Phase 28: Security & Access Control

**Goal:** Add security features for multi-tenant deployments.

> **Status (2026-08-18): partially complete.** The service auth primitives
> (28.1 API keys + JWT) shipped in Phase 27.1, and **namespace-scoped API
> keys** (28.1 namespace-level isolation) plus the README "Security
> Guidance" section shipped on 2026-08-18. Remaining items are **deferred
> until Recall is used as a multi-tenant hosted service** — for library and
> single-tenant service use, the operational guidance in the README is
> sufficient.

### 28.1 Authentication & Authorization
- [x] API key authentication (Phase 27.1: `api.NewAPIKeyAuth`, `X-API-Key`/Bearer)
- [x] JWT token validation (Phase 27.1: `api.NewJWTAuth`, HS256 via stdlib `crypto/hmac`)
- [ ] Role-based access control (RBAC) — deferred (multi-tenant)
- [x] Namespace-level isolation enforcement — **namespace-scoped API keys** (`api.ScopedAPIKeyAuth`, `auth.scoped_keys`): per-key namespace restrictions on uploads, search, RAG, and graph/reasoning endpoints, fail closed (2026-08-18)
- [ ] Row-level security for metadata filters — deferred (multi-tenant)

### 28.2 Data Security
- [ ] Encryption at rest — deferred: page encryption (SQLCipher) is incompatible with the zero-CGO constraint; guidance points to filesystem/volume-level encryption (README "Security Guidance")
- [ ] Encryption in transit — deferred: TLS termination at the load balancer/proxy is the documented approach (bundled `deploy/` assets assume it); in-process `Server.ListenAndServeTLS` is available
- [x] Secrets management — environment overrides (`RECALL__*`) and the K8s Secret in `deploy/kubernetes/recall.yaml` (Phases 27.3/27.4); vault integration deferred
- [ ] Audit logging for all operations — deferred (multi-tenant)

### 28.3 Input Validation
- [ ] Query sanitization — deferred (not applicable: no dynamic SQL is built from query input)
- [x] Document size limits (Phase 27.1: `server.max_upload_bytes` enforced via per-request `MaxBytesReader`)
- [x] Embedding dimension validation (pre-existing: indexes reject chunks whose embedding dimension does not match)
- [x] SQL injection prevention (inherent: `modernc.org/sqlite` prepared statements, all queries parameterized)

### 28.4 Compliance
- [ ] Data retention policies — deferred (multi-tenant)
- [ ] Right to deletion (GDPR) — deferred (multi-tenant)
- [ ] Data export functionality — deferred (multi-tenant)
- [ ] Compliance reporting — deferred (multi-tenant)

**Estimated Effort:** 3–4 weeks
**Priority:** Low (needed for enterprise deployments); the deferred items above activate when (and if) a multi-tenant hosted service is required

---

## Phase 29: CLI Tool

**Goal:** Provide a command-line interface for common operations.

**Status:** Complete — `cmd/recall` ships every command below in two modes: local (in-process against the configured store) and server (HTTP client of a running recall-server via `--server`/`cli.url`), backed by the shared `app` assembly package and the typed `client` REST client.

### 29.1 Core Commands
- [x] `recall upload` — upload documents to a store (files + recursive directories; text, markdown, CSV, JSON, HTML, PDF, DOCX)
- [x] `recall search` — perform vector search
- [x] `recall hybrid-search` — perform hybrid search (`--bm25-weight`)
- [x] `recall rag` — run RAG query (context assembly + rendered prompt with citations)
- [x] `recall graph` — query the knowledge graph (entity by ID/label with neighbors + relations, `graph list`)
- [x] `recall reason` — run multi-hop reasoning (natural-language query or `--from`/`--to` path exploration)

### 29.2 Management Commands
- [x] `recall store info` — display store statistics (health, chunks, namespaces, schema version, integrity)
- [x] `recall store migrate` — run schema migrations (versioned `-- recall-migration:` SQL files, transactional, idempotent)
- [x] `recall store backup` — create a backup (online `VACUUM INTO`)
- [x] `recall store restore` — restore from backup (atomic temp-file + rename, `--force` guard)
- [x] `recall cluster status` — check distributed cluster health (probes each node's /diagnostics; exit 1 on down/unreachable nodes)

### 29.3 Evaluation Commands
- [x] `recall eval` — run evaluation benchmarks (Precision/Recall/MRR/NDCG@K over an eval dataset, `--save` report)
- [x] `recall eval compare` — compare two configurations (tolerance-gated regression check, exit 2 on regression — CI gate)

### 29.4 Implementation
- [x] Use `github.com/spf13/cobra` for CLI framework
- [x] Subcommands with flags and positional arguments
- [x] Output formatting (table, JSON, YAML via `-o/--output` or `cli.output`)
- [x] Configuration file support (~/.recall.yaml, .yml, .json; `--config`; `RECALL__SECTION__KEY` env overrides; `cli` config section with url/api_key/timeout/output/cluster_nodes)

**Estimated Effort:** 2–3 weeks
**Priority:** Low-Medium

---

## Phase 30: Web UI (Optional)

**Goal:** Provide a visual interface for exploration and debugging.

### 30.1 Dashboard
- [ ] Store statistics (documents, chunks, namespaces)
- [ ] Search query interface
- [ ] Result visualization with highlighting
- [ ] Graph visualization (force-directed layout)

### 30.2 Annotation Interface
- [ ] Chunk review and annotation
- [ ] Relevance labeling
- [ ] Feedback submission

### 30.3 Monitoring Dashboard
- [ ] Query analytics
- [ ] Performance metrics
- [ ] Error tracking

### 30.4 Implementation
- [ ] Embeddable web component or standalone app
- [ ] React/Vue frontend with Go backend
- [ ] OR: integrate with existing tools (Grafana for metrics, etc.)

**Estimated Effort:** 4–6 weeks
**Priority:** Low

---

## Phase 31: SDK Wrappers (Community)

**Goal:** Enable ecosystem integration.

### 31.1 Python SDK
- [ ] `recall-python` — Python client library
- [ ] Async support (asyncio)
- [ ] LangChain integration (`langchain-recall`)
- [ ] LlamaIndex integration (`llama-index-retrievers-recall`)

### 31.2 TypeScript SDK
- [ ] `recall-js` — Node.js/browser client
- [ ] Next.js integration
- [ ] LangChain.js integration

### 31.3 Plugin Ecosystem
- [ ] Plugin interface for custom chunkers, embedders, rerankers
- [ ] Plugin marketplace / registry
- [ ] Community plugin examples

**Estimated Effort:** Ongoing
**Priority:** Low

---

## Phase 32: Project Hygiene & Documentation

**Status: Complete (2026-08-19)**

**Goal:** Improve project infrastructure and documentation.

### 32.1 Documentation
- [x] `CHANGELOG.md` — version history
- [x] `CONTRIBUTING.md` — contribution guidelines
- [x] `SECURITY.md` — security policy and reporting
- [x] `CODE_OF_CONDUCT.md` — community standards
- [x] `ARCHITECTURE.md` — detailed architecture decisions
- [x] `GOVERNANCE.md` — project governance
- [x] Per-package README files — 33 of 36 packages. `govern/`, `example/e2e/` and `example/production/` have none; `govern/`'s package comment carries its documentation instead.
- [x] Migration guide for version upgrades (`docs/MIGRATION.md`)

### 32.2 CI/CD
- [x] GitHub Actions workflow (`.github/workflows/go.yml`)
  - [x] Lint (`golangci-lint`, pinned version)
  - [x] Test (`go test -cover`, overall ≥80% coverage gate)
  - [x] Build verification
  - [x] Coverage threshold enforcement
  - [x] Benchmark regression detection (`scripts/benchcompare.sh` baseline gate)
  - [x] Dependency vulnerability scanning (`govulncheck`)
  - [x] License compliance checking
- [x] Semantic versioning with automated tagging (`.github/workflows/tag.yml`)
- [x] Automated release notes generation (`.github/workflows/release.yml` + `scripts/release-build.sh` cross-platform binaries)

### 32.3 Code Quality
- [x] `golangci-lint` configuration (`.golangci.yml`, schema v2: errcheck/govet/ineffassign/staticcheck/unused + gosec, misspell, revive, unconvert; curated exclusions documented inline)
- [x] Static analysis (staticcheck, revive) — **0 issues repo-wide**
- [x] Dependency vulnerability scanning (`govulncheck` in CI)
- [x] License compliance checking (CI)

### 32.4 Examples Enhancement
- [x] More examples in `example/`
- [x] End-to-end tutorial (`example/e2e`: ingest → search → RAG → evaluate → reasoning)
- [x] Production deployment example (`example/production`: `app.BuildAPIServer` + typed `client`)
- [x] Benchmark comparison guide (`docs/BENCHMARKS.md` + `scripts/benchcompare.sh`)

**Estimated Effort:** 2–3 weeks
**Priority:** Medium

---

## Summary: Recommended Execution Order

```
Priority 1 (Foundation)          Priority 2 (Production)        Priority 3 (Growth)          Priority 4 (Ecosystem)
┌─────────────────────┐         ┌─────────────────────┐        ┌─────────────────────┐      ┌─────────────────────┐
│ Phase 19: Test      │         │ Phase 20: Embedding │        │ Phase 23:           │      │ Phase 27: API       │
│   Coverage          │         │   Providers         │        │   Observability     │      │   & Service Layer   │
│ Phase 21: Doc       │         │ Phase 22: Reranking │        │ Phase 24: Feedback  │      │ Phase 28: Security  │
│   Ingestion         │         │ Phase 25: Resilience│        │   & Evaluation      │      │ Phase 29: CLI       │
│ Phase 26: Advanced  │         │ Phase 21: Doc       │        │ Phase 26: Advanced  │      │ Phase 30: Web UI    │
│   Retrieval         │         │   Ingestion (cont.) │        │   Retrieval (cont.) │      │ Phase 31: SDKs      │
└─────────────────────┘         └─────────────────────┘        └─────────────────────┘      └─────────────────────┘
         │                               │                              │                          │
         ▼                               ▼                              ▼                          ▼
    ~4 weeks                        ~10 weeks                      ~10 weeks                    ~12 weeks
```

### Total Estimated Effort: ~36–38 weeks for full roadmap

### Quick Wins (can be done in parallel, ~2 weeks each):
1. ~~**Test coverage hardening** (Phase 19)~~ — done 2026-08-18 (all 14 packages ≥80%)
2. ~~**Project hygiene** (Phase 32)~~ — done 2026-08-19 (lint at 0 issues, CI/release workflows, governance + per-package docs, e2e/production examples)
3. ~~**CLI tool** (Phase 29)~~ — done 2026-08-19 (`cmd/recall`: local + server modes, all data/store/cluster/eval commands)

---

## Backlog (Deferred Items)

Items deliberately not implemented yet, with the reason they were deferred:

- **Incremental BM25 keyword index for large shards** (`distributed`) —
  `ShardIndex.SearchHybrid` builds its BM25 index per call (O(n) per
  query), which is fine for in-process shards. Very large shards at high
  query rates should maintain the keyword index incrementally as chunks are
  added/removed. (Recorded during Phase 3 remediation, 2026-08-21.)
- **Network transport for the distributed package** — `distributed`
  currently simulates a cluster in-process (node `Address` fields are
  metadata; fan-out is local function calls). A future phase can swap the
  in-process fan-out for a real transport (e.g. gRPC) behind the same
  interfaces.