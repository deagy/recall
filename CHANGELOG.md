# Changelog

All notable changes to **Recall** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Breaking changes are called out with a **Breaking** label and always carry a
note in [docs/MIGRATION.md](./docs/MIGRATION.md).

## [Unreleased]

### Fixed

- `govern`: a whitespace-only query is now refused. `Validate` compared the
  query to the empty string exactly, so `"   "` passed, reached the store,
  scored 0 against everything and wrote an audit row — a retrieval that
  happened, was recorded, and asked nothing. The refusal exists so that no
  unqualified query is served, and `"  "` is as unqualified as `""`.

  Inherited from the engine this package was ported from, which used the same
  exact comparison, and invisible to the ported contract fixture because the
  fixture is what was ported. Found by an independent audit of the migration.

## [0.3.1] — 2026-09-01

### Fixed

- CI: `govern`'s `TestTheContractMatchesItsOrigin` never received the origin
  file it compares against, so it hard-failed under CI for want of the file
  rather than for divergence — and v0.3.0 was tagged with that job red. The
  workflow now checks out the origin repository and points the guard at its
  contract. No library code changed; `v0.3.1` is `v0.3.0` with a CI that
  actually runs the guard.

## [0.3.0] — 2026-08-31

### Added

- `govern`: a fail-closed retrieval view over a recall store. It refuses a
  request that has not decided its scope, has not declared a classification, or
  cannot be recorded, and it refuses **before** touching the store — an
  interface that only refuses after opening a connection has already revealed
  that the caller asked. It decides no policy: classification and source names
  are opaque required strings, and the vocabulary belongs to the embedding
  system.

### Breaking

- **Breaking** (`distributed`, in-development package): hybrid search now
  takes the raw query text and the query embedding separately instead of a
  single `[]float32` that the hybrid path internally re-approximated
  lexically. New signatures: `ShardManager.SearchHybrid(ctx, query string,
  queryEmb []float32, opts)`, `Shard.SearchHybrid(ctx, query, queryEmb,
  opts)`, `ScatterGatherSearchHybrid(ctx, sm, query, queryEmb, opts,
  config)`, `ShardIndex.SearchHybrid(ctx, query, queryEmb, opts)`. Pass the
  query and its embedding (e.g. from your embedder) as separate arguments.

### Fixed

- `distributed`: replica shards are now created with the **deterministic
  ID the API reports** (`<shard>-<kind>-<node>`), so
  `ReplicationResult.ReplicaID` and `GetReplicationStatus` can actually
  resolve and count the replicas (previously auto-IDed shards were created
  and status could never find them). Replication is now idempotent:
  repeated `ReplicateData` calls reuse existing replica shards and update
  their data instead of creating unbounded duplicates.
- `distributed`: auto-generated shard IDs come from a monotonic
  `ShardManager` counter that never shrinks after `DeleteShard`, so an ID
  can no longer be reused and silently overwrite a live shard's map entry.
- `distributed`: `ScatterGatherConfig.Timeout` is now actually applied —
  a positive timeout runs the fan-out under a derived context (an earlier
  caller deadline is never extended); `0` remains "no timeout".
  `MaxResultsPerShard` is enforced before merging, and the vector/hybrid
  fan-outs share one code path.
- `distributed`: `ShardIndex` now snapshots the shard's chunk map under the
  shard's read lock in `NewShardIndex` instead of reading live state with
  no locking (a concurrent write raced the old path under `-race`);
  `ShardIndex.SearchHybrid` is now real hybrid search (BM25 + vector
  cosine over the caller-provided query embedding, fused with `opts.Fusion`
  or the `(1-BM25Weight, BM25Weight)` weighted sum, matching
  `index.HybridIndex` semantics) instead of scoring BM25 against a fake
  hashed 128-dim embedding; the O(n²) selection sort was replaced by
  `sort.Slice` with a deterministic chunk-ID tie-break.
- `distributed`: `replicateQuorum` sorts the node list before picking the
  quorum set (map-iteration order made the set nondeterministic, so
  repeated replicates of the same data kept creating new replica shards),
  and `replicatePrimaryReplica` skips the primary by node ID rather than
  assuming ring position 0.

- `reasoning`: `Engine.InferRelations` now honors the configured
  `MinConfidence` (previously validated but never applied; inferences were
  filtered by an arbitrary, order-dependent threshold derived from the first
  stored relation). The `Graph.Relations()` copy is also hoisted out of the
  rule loop (was a full locked copy per relation — O(R²) allocations).
- `chunker`: `FixedChunker` no longer emits chunks larger than
  `MaxTokens*4` runes when a single part exceeds the limit (oversized parts
  are now pre-split regardless of accumulated state); the join size check
  accounts for separator length. A zero-value `Separator` now defaults to
  `"\n\n"` at construction, so `Config{}` no longer joins parts with an
  empty string.
- `fuse`: `WeightedFusion.Fuse` with more than two score maps now splits
  `(1-alpha)` evenly across the remaining maps, so per-map weights always
  sum to 1 (previously every map after the first received the full
  `1-alpha`).
- `docs`: `Engine.ExplorePaths` godoc corrected — it returns a
  breadth-first search tree (at most one path per intermediate entity), not
  all simple paths between the endpoints.

### Behavior

- `distributed`: `DistributedStore.Search`/`SearchHybrid` now embed the
  query with the store's embedder when one is configured; the lexical
  `generateQueryEmbedding` remains only as a documented fallback for
  embedder-less stores (its vectors are not semantically meaningful).
- `distributed`: `ShardIndex` results reflect the shard's state at
  `NewShardIndex` time (snapshot semantics), not the shard's current state.
- `distributed`: `CreateShardWithID` with an already-registered ID now
  returns `ErrShardExists` instead of silently replacing the existing
  shard; `ShardManager.StoreChunk` tolerates the concurrent-create race by
  reusing the winner.
- `reasoning`: inferences below `MinConfidence` (default 0.3) are now
  dropped from `InferRelations`/`Reason` results; `/graph/reason` output may
  exclude low-confidence inferences that were previously returned.
- `chunker`: `FixedChunker` chunk boundaries shift slightly (separator-aware
  packing, pre-split oversized parts); document uploads may chunk
  differently.

## [0.2.0] — 2026-08-21

Remediation release: Phase 1 data-race fixes and Phase 2 core-correctness
fixes. Includes one **Breaking** change; see the v0.2.0 notes in
[docs/MIGRATION.md](./docs/MIGRATION.md).

### Fixed

- `store`: `MemoryStore.DeleteDocument` no longer iterates the live
  `docChunks[docID]` map outside the store lock. It now snapshots the
  document's chunk IDs into a slice while holding the lock and iterates
  the snapshot, eliminating a fatal
  `concurrent map iteration and map write` when a concurrent `Upload`
  writes to the same document's chunk map (reproduced under `-race`).
  IDs added by an in-flight upload after the snapshot are left for a
  subsequent delete.
- `ingest`: `Incremental.Save` no longer marshals the live `hashes` map
  outside the lock. It copies the map under the lock, marshals the copy,
  and re-arms `dirty` on a failed encode or write so a failed save does
  not lose the pending state (previously `dirty` was cleared before
  marshaling). The dead `ids`/`sort.Strings` code was removed.
  `concurrent map iteration and map write` reproduced under `-race`
  pre-fix.
- Regression tests: `TestMemoryStore_DeleteDocument_ConcurrentUpload`
  (gated-embedder coordination of Upload's write burst with concurrent
  `DeleteDocument` iterations) and
  `TestIncremental_Save_ConcurrentMark` (concurrent `Mark` during
  `Save`) plus `TestIncremental_Save_FailureKeepsDirty` (failed save
  keeps state dirty).

- **Breaking** `index`/`store`: deleting a missing chunk now reports
  `core.ErrNotFound` instead of returning `nil`. `MemoryIndex.Delete`
  (plain and HNSW modes) and `MemoryStore.DeleteChunk` (which now tries
  every namespace index) return the error when no index contains the ID.
  See the migration note in
  [docs/MIGRATION.md](./docs/MIGRATION.md).
- `bm25`: re-adding an existing document no longer double-counts
  `docCount`/`docFreq` (IDF and `avgDocLen` drift), and removing an unknown
  document is a no-op instead of decrementing `docCount`.
- `store`: SQLite `created_at`/`updated_at` now store real UTC RFC3339
  timestamps; they previously stored local wall-clock time with a literal
  `Z` suffix (fake UTC marker).
- `store`: SQLite `DeleteChunk`/`DeleteDocument` now remove the associated
  `embeddings` rows in the same transaction; the declared `ON DELETE CASCADE`
  never fired because `PRAGMA foreign_keys` is off by default in SQLite, so
  deletes left orphaned embedding rows.
- `store`/`index`: `SearchOptions.MinScore` is now honored on all search
  paths — SQLite brute-force, SQLite HNSW, SQLite FTS5 hybrid fusion, and
  MemoryStore hybrid fusion (applied to the fused score).
  `MemoryStore.SearchHybrid` no longer pre-filters the vector leg by
  `MinScore`, so hybrid fusion sees every chunk's true vector score.
- Regression tests: BM25 count consistency (`bm25`), not-found delete
  semantics (`index`, `store`), UTC timestamps + embedding cleanup
  (`store/sqlite_test.go`), and Memory vs SQLite MinScore parity for vector
  and hybrid search (`store/minscore_parity_test.go`).

## [0.1.0] — 2026-08-19

The initial release contains everything developed through Phase 32 of the
[roadmap](./ROADMAP.md). Summary by area:

### Added — Phase 32: Project Hygiene & Documentation

- `golangci-lint` configuration (`.golangci.yml`, schema v2) with the default
  linters plus gosec, misspell, revive, and unconvert; every exclusion is
  narrow and documented inline. All real findings fixed in code; targeted
  `//nolint:gosec` on four analyzed false positives. **Repo-wide lint is
  0 issues** (previous baseline: 587).
- CI: `.github/workflows/go.yml` — lint (pinned golangci-lint),
  build & test (vet, gofmt, `go test -race`, overall ≥80% coverage
  gate, coverage artifact), govulncheck + license scan, and a
  benchmark-regression job gated by `scripts/benchcompare.sh`.
- Release automation: `.github/workflows/tag.yml` (semver tag creation)
  and `.github/workflows/release.yml` (tag-triggered cross-platform
  binaries via `scripts/release-build.sh` + generated release notes).
- Governance documentation: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `ARCHITECTURE.md`, `GOVERNANCE.md`.
- Per-package README files for all 32 packages.
- Examples: `example/e2e` (deterministic end-to-end tutorial: ingest →
  search → RAG → evaluation → graph/reasoning) and
  `example/production` (in-process API server driven by the typed
  client), plus `example/README.md`.
- Guides: `docs/BENCHMARKS.md` (benchmark comparison) and
  `docs/MIGRATION.md` (version upgrades).

### Changed

- `llm`: request/response conversion now uses struct conversions between
  the identical message types (staticcheck S1016).
- `embedder/onnx`: `NewTensor` uses a type switch; Reshape dimension loop
  uses a tagged switch (staticcheck).
- `store`: `SQLiteGraphStore.LoadFromDB` now surfaces corrupted
  property JSON instead of silently loading empty properties.
- `connector/web`: rate-limit timer uses `time.Until` (S1024).
- `example`: `main.go` now checks upload errors.
- Doc comments reworded to canonical staticcheck forms in `cache`,
  `pipeline`, `query`, `distributed`, and `testutil`.

### Fixed

- CI license check: `go-licenses` misclassifies the BSD-3-Clause LICENSE
  of `modernc.org/mathutil` (transitive dependency of
  `modernc.org/sqlite`); the license was manually verified and the
  package is now explicitly excluded with a documented reason.
- Release workflow: re-tagging a version now replaces the existing
  GitHub release instead of failing on the duplicate tag name.

### Added — Core & Retrieval

- Core data model: `core.Chunk`, `core.Document`, typed `core.Value` metadata
  (Phases 1–3)
- Text chunking: fixed, recursive, semantic (similarity-based + streaming),
  parent-child, document-aware, and adaptive chunkers (Phases 1, 16, 26)
- Embedding abstraction with injectable embedders: Mock, OpenAI, Cohere,
  Ollama (HTTP with retry/backoff), `embedder.Pipeline` failover chain,
  `CachingEmbedder`, multi-modal embedder, and a pure-Go ONNX runtime
  (`embedder/onnx`) with bundled offline WordPiece tokenizers for
  `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text-v1.5`
  (Phases 2, 20)
- In-memory index with brute-force and HNSW ANN search (auto-enabled at 1K+
  chunks), metadata filters (term/range/date/custom) (Phases 2–3, 6)
- Advanced indexing: SQ8 scalar quantization, product quantization (PQ/ADC),
  hybrid index, inverted metadata index, multi-vector index (Phase 26)
- BM25 keyword ranking and score fusion (weighted, RRF) (Phase 4)
- SQLite persistence via pure-Go `modernc.org/sqlite` — zero CGO (Phase 5)
- Hybrid search (vector + BM25) on all store backends (Phase 4)

### Added — RAG, Graph & Reasoning

- RAG pipeline: context assembly, prompt templates, token budgeting, smart
  context window, citations, multi-modal pipeline (Phases 7, 25.4, 26)
- Knowledge graph: entities/relations, BFS/DFS, transitive closure, path
  finding, common-neighbor inference, pluggable NER and pattern-based relation
  extraction, TransE graph embeddings with link prediction
  (Phases 8–9, 11, 17)
- Multi-hop reasoning engine: inference rules (transitive, symmetric,
  anti-symmetric, inverse, composition, hierarchy), depth-limited path
  exploration, confidence propagation, natural-language query reasoning
  (Phase 10)
- Reranking: cross-encoder (pure-Go ONNX), BM25 sparse, LLM-as-judge,
  ensemble, pointwise + adaptive learning-to-rank, A/B experiment framework
  (Phases 22, 22.3)
- Advanced query processing: intent detection, entity extraction, expansion,
  adaptive retrieval, LLM rewrite/HyDE/step-back/sub-query/multilingual
  (Phases 13, 26)

### Added — Ingestion

- Document loaders: text, markdown, CSV, JSON, HTML, PDF, DOCX, directory
  (Phase 21)
- Source connectors: web (rate-limited), git, S3 (self-contained SigV4),
  GitHub, SQL databases (Phase 21)
- Ingestion pipeline: load → dedup → validation → transform → upload with
  progress callbacks, parallel batch ingestion, incremental re-ingestion
  (Phase 21)

### Added — Resilience & Distributed

- LLM backend resilience: retry, circuit breaker, rate limiting, timeout,
  fallback decorators (Phase 25.1)
- Store resilience: checkpoint, backup/restore, versioned SQL migrations,
  corruption detection/repair (Phase 25.2)
- Distributed storage: consistent hashing, sharding, scatter-gather search,
  replication strategies, node health, auto-rebalance, fault tolerance
  (Phases 15, 25.3)

### Added — Observability, Feedback & Evaluation

- Metrics (Prometheus text export), OpenTelemetry-compatible tracing, health
  checks/diagnostics, query analytics (Phases 23.1–23.4)
- Relevance feedback (Rocchio), evaluation framework (Precision/Recall/MRR/
  NDCG@K + answer-quality metrics), human-in-the-loop (review queue,
  annotations, active learning) (Phases 24.1–24.3)
- Intelligent caching: LRU/TTL, query/embedding/graph traversal caches,
  multi-level L1/L2, cache warming (Phase 18)

### Added — Service Layer

- REST API service (`api` package, stdlib-only) with OpenAPI 3.0 spec,
  API-key/JWT/scoped-key authentication, CORS, body limits; `recall-server`
  standalone binary with graceful shutdown (Phase 27.1)
- Configuration package: JSON/YAML, env overrides (`RECALL__SECTION__KEY`),
  validation, hot reload (Phase 27.3)
- Deployment: distroless multi-stage Dockerfile, docker-compose, Kubernetes
  manifests (Phase 27.4)
- CLI (`cmd/recall`): local + server modes, upload/search/hybrid-search/rag/
  graph/reason, store info/migrate/backup/restore, cluster status, eval
  commands (Phase 29)
- Typed REST client (`client`) and shared service assembly (`app`) (Phase 29)

### Added — Security

- Namespace-scoped API keys enforced on every data endpoint, failing closed;
  namespace stamping on upload; operational security guidance (Phase 28,
  recommended subset — RBAC/encryption/audit deferred to multi-tenant hosting)

### Added — Testing & Quality

- All packages ≥80% statement coverage; deterministic `testutil` fixtures
  (Phase 19, 24.4)
- CI: vet, gofmt gate, build, race-enabled tests, ≥80% coverage gate,
  benchmark smoke run (Phase 24.4)
- Project hygiene: LICENSE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT,
  ARCHITECTURE, GOVERNANCE, per-package READMEs, CHANGELOG, migration guide,
  golangci-lint + govulncheck + license compliance in CI, release automation
  (Phase 32)

[Unreleased]: https://github.com/deagy/recall/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/deagy/recall/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/deagy/recall/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/deagy/recall/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/deagy/recall/releases/tag/v0.1.0
