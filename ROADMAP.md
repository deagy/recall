# Recall — Development Roadmap

This document outlines the future development roadmap for the Recall library, organized by priority tiers and target phases. The current codebase has **22 completed phases** covering core RAG functionality (storage, search, chunking, knowledge graphs, reasoning, distributed storage, caching, LLM integration, document ingestion, and advanced retrieval).

---

## Current State Assessment

| Metric | Status |
|--------|--------|
| Completed Phases | 23 (Phases 1–22, 26) |
| Packages | 19 source packages + 1 example |
| All Tests Pass | ✅ Yes |
| Overall Coverage Target | ✅ >80% in all 19 packages (re-measured 2026-08-18) |
| Zero CGO | ✅ Maintained |
| LLM Backends | OpenAI, Ollama (Mock) |
| Embedding Providers | OpenAI, Cohere, Ollama, ONNX (local), Mock |
| Fusion Methods | WeightedFusion, RRF |

### Coverage by Package (re-measured 2026-08-18 — ✅ all above target)

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

**Status: ✅ Complete (2026-08-18)** — all 14 packages are above the 80% target
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
- [ ] `metrics.StoreMetrics` — search latency, throughput, error rates
- [ ] `metrics.EmbeddingMetrics` — embedding latency, dimension stats
- [ ] `metrics.CacheMetrics` — hit/miss ratio, eviction count
- [ ] `metrics.GraphMetrics` — traversal depth, inference counts
- [ ] Export to Prometheus format (`/metrics` endpoint)
- [ ] Structured logging with correlation IDs

### 23.2 Tracing
- [ ] OpenTelemetry integration
- [ ] Distributed trace spans for search, embedding, chunking
- [ ] Store-level tracing (upload → search → retrieve)
- [ ] Span tags for metadata (namespace, document ID, query type)

### 23.3 Health & Diagnostics
- [ ] `store.HealthCheck()` — connectivity, index integrity
- [ ] `store.IntegrityCheck()` — verify data consistency
- [ ] `distributed.ClusterHealth()` — node status, shard distribution
- [ ] Expose diagnostics via configurable endpoint or CLI

### 23.4 Query Analytics
- [ ] `analytics.QueryLog` — log queries with latency and results
- [ ] `analytics.PopularQueries` — trending query detection
- [ ] `analytics.DropOffDetection` — queries with no good results
- [ ] Export to configurable sink (file, HTTP, message queue)

**Estimated Effort:** 3–4 weeks
**Priority:** Medium

---

## Phase 24: Feedback Loop & Evaluation

**Goal:** Enable continuous improvement of retrieval quality.

### 24.1 Relevance Feedback
- [ ] `feedback.RelevanceFeedback` — adjust query based on user feedback
- [ ] `feedback.Rocchio` — classic Rocchio algorithm for query expansion
- [ ] `feedback.ExpandAndRetrieve` — retrieve → user marks relevant → re-rank
- [ ] Store feedback in metadata for future training

### 24.2 Evaluation Framework
- [ ] `eval.RetrievalEval` — precision, recall, MRR, NDCG at K
- [ ] `eval.RAGEval` — answer quality metrics (faithfulness, relevance)
- [ ] `eval.BenchmarkSuite` — regression testing for retrieval quality
- [ ] `eval.Dataset` — load/save evaluation datasets
- [ ] `eval.Report` — generate evaluation reports

### 24.3 Human-in-the-Loop
- [ ] `hitl.ReviewQueue` — queue chunks for human review
- [ ] `hitl.Annotation` — store human annotations
- [ ] `hitl.ActiveLearning` — prioritize uncertain chunks for review
- [ ] Web UI for annotation (optional, Phase 30)

### 24.4 Automated Testing
- [ ] `testutil.FixtureStore` — in-memory store with preloaded data
- [ ] `testutil.MockLLM` — deterministic LLM responses for testing
- [ ] `testutil.GoldenFile` — compare results against golden files
- [ ] CI integration for regression testing

**Estimated Effort:** 3–4 weeks
**Priority:** Medium

---

## Phase 25: Resilience & Reliability

**Status: ✅ Complete (2026-08-18)** — all four sub-phases implemented and tested (see checklist below).

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

**Status: ✅ Complete (2026-08-18)** — all four sub-phases implemented and tested (see checklist below).

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
- [ ] `api.Server` — HTTP server using standard library
- [ ] `POST /upload` — upload documents
- [ ] `GET /search` — vector search
- [ ] `POST /hybrid-search` — hybrid search
- [ ] `POST /rag` — full RAG pipeline query
- [ ] `GET /graph/{entity}` — graph entity lookup
- [ ] `POST /graph/reason` — graph reasoning
- [ ] OpenAPI/Swagger specification
- [ ] API authentication (API keys, JWT)

### 27.2 gRPC (Future)
- [ ] Protocol buffer definitions
- [ ] gRPC server with streaming support
- [ ] Bidirectional streaming for RAG responses

### 27.3 Configuration
- [ ] YAML/JSON configuration file support
- [ ] Environment variable overrides
- [ ] Configuration validation
- [ ] Hot-reload support

### 27.4 Deployment
- [ ] Dockerfile
- [ ] docker-compose for multi-node deployment
- [ ] Kubernetes manifests (Deployment, Service, HPA)
- [ ] Health check endpoints for orchestrators

**Estimated Effort:** 4–5 weeks
**Priority:** Medium-Low

---

## Phase 28: Security & Access Control

**Goal:** Add security features for multi-tenant deployments.

### 28.1 Authentication & Authorization
- [ ] API key authentication
- [ ] JWT token validation
- [ ] Role-based access control (RBAC)
- [ ] Namespace-level isolation enforcement
- [ ] Row-level security for metadata filters

### 28.2 Data Security
- [ ] Encryption at rest (SQLite page encryption)
- [ ] Encryption in transit (TLS for distributed mode)
- [ ] Secrets management (environment variables, vault integration)
- [ ] Audit logging for all operations

### 28.3 Input Validation
- [ ] Query sanitization
- [ ] Document size limits
- [ ] Embedding dimension validation
- [ ] SQL injection prevention (for SQLite backend)

### 28.4 Compliance
- [ ] Data retention policies
- [ ] Right to deletion (GDPR)
- [ ] Data export functionality
- [ ] Compliance reporting

**Estimated Effort:** 3–4 weeks
**Priority:** Low (needed for enterprise deployments)

---

## Phase 29: CLI Tool

**Goal:** Provide a command-line interface for common operations.

### 29.1 Core Commands
- [ ] `recall upload` — upload documents to a store
- [ ] `recall search` — perform vector search
- [ ] `recall hybrid-search` — perform hybrid search
- [ ] `recall rag` — run RAG query
- [ ] `recall graph` — query the knowledge graph
- [ ] `recall reason` — run multi-hop reasoning

### 29.2 Management Commands
- [ ] `recall store info` — display store statistics
- [ ] `recall store migrate` — run schema migrations
- [ ] `recall store backup` — create a backup
- [ ] `recall store restore` — restore from backup
- [ ] `recall cluster status` — check distributed cluster health

### 29.3 Evaluation Commands
- [ ] `recall eval` — run evaluation benchmarks
- [ ] `recall eval compare` — compare two configurations

### 29.4 Implementation
- [ ] Use `github.com/spf13/cobra` for CLI framework
- [ ] Subcommands with flags and positional arguments
- [ ] Output formatting (table, JSON, YAML)
- [ ] Configuration file support (~/.recall.yaml)

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

**Goal:** Improve project infrastructure and documentation.

### 32.1 Documentation
- [ ] `CHANGELOG.md` — version history
- [ ] `CONTRIBUTING.md` — contribution guidelines
- [ ] `SECURITY.md` — security policy and reporting
- [ ] `CODE_OF_CONDUCT.md` — community standards
- [ ] `ARCHITECTURE.md` — detailed architecture decisions
- [ ] `GOVERNANCE.md` — project governance
- [ ] Per-package README files
- [ ] Migration guide for version upgrades

### 32.2 CI/CD
- [ ] GitHub Actions workflow
  - [ ] Lint (`golangci-lint`)
  - [ ] Test (`go test -cover`)
  - [ ] Build verification
  - [ ] Coverage threshold enforcement
  - [ ] Benchmark regression detection
- [ ] Semantic versioning with automated tagging
- [ ] Automated release notes generation

### 32.3 Code Quality
- [ ] `golangci-lint` configuration
- [ ] Static analysis (staticcheck, revive)
- [ ] Dependency vulnerability scanning
- [ ] License compliance checking

### 32.4 Examples Enhancement
- [ ] More comprehensive examples in `example/`
- [ ] End-to-end tutorial (ingest → search → RAG → evaluate)
- [ ] Production deployment example
- [ ] Benchmark comparison guide

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
1. ~~**Test coverage hardening** (Phase 19)~~ — ✅ done 2026-08-18 (all 14 packages ≥80%)
2. **Project hygiene** (Phase 32) — improves developer experience
3. **CLI tool** (Phase 29) — improves usability