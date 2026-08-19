# Recall — Development Plan

> **Note:** This file tracks completed phases. For the full future roadmap, see [ROADMAP.md](./ROADMAP.md).

## Completed Phases

- [x] Phase 1: Core data model + chunking
- [x] Phase 2: Embedding + in-memory index
- [x] Phase 3: Query engine with filters
- [x] Phase 4: Hybrid search (BM25 + vector fusion)
- [x] Phase 5: SQLite persistence (modernc.org/sqlite, pure Go)
- [x] Phase 6: HNSW ANN index (auto-enabled for 1K+ chunks)
- [x] Phase 7: RAG pipeline (context assembly, prompt templates, token management)
- [x] Phase 8: Knowledge graph (entity/relation extraction, BFS/DFS traversal, transitive closure, path finding, common-neighbor inference)
- [x] Phase 9: Graph-based RAG (GraphStore interface, entity/relation extraction from text, graph-augmented retrieval)

---

## Improvement Opportunities

### 1. Embedder Tests (`embedder/`)
**Priority:** Low
**Status:** ✅ Complete — 15 tests added covering MockEmbedder dimension validation, embedding consistency, normalization, batch processing, cosine similarity, and sqrt utility.
**Action:** Add unit tests for MockEmbedder (dimension validation, embedding generation consistency).

### 2. Pluggable NER + Relation Extraction (`graph/`)
**Priority:** High
**Status:** ✅ Complete — `NERExtractor` interface defined, `PatternRelationExtractor` with 8 default patterns (works_at, located_in, founded_by, part_of, related_to, taught_by, parent_of, ceo_of), pluggable for custom NER models.
**Action:**
- Define `NERExtractor` interface with `Extract(text string) ([]*Entity, error)`
- Add pattern-based relation extractor (e.g., "X works at Y", "X is CEO of Y")
- Pluggable so users can bring their own NER models later

### 3. SQLite Graph Persistence (`store/`)
**Priority:** Medium
**Status:** ✅ Complete — `SQLiteGraphStore` persists entities and relations to SQLite with `GraphPersistence` interface, `LoadFromDB()`, `Clear()`, and full round-trip persistence.
**Action:** Add `SQLiteGraphStore` that persists entities and relations to SQLite tables.

### 4. Example Package (`example/`)
**Priority:** Medium
**Status:** ✅ Complete — `example/main.go` demonstrates upload+search, hybrid search, knowledge graph, and graph-based RAG workflows.
**Action:** Create `example/` with usage examples for common patterns (upload, search, hybrid, graph queries).

### 5. GoDoc Documentation
**Priority:** Medium
**Status:** ✅ Complete — All public types, methods, and interfaces across all packages have godoc-friendly comments.
**Action:** Add package-level and function-level doc comments for all public types and methods.

### 6. Benchmark Tests (`*_test.go`)
**Priority:** Medium
**Status:** ✅ Complete — Benchmarks added for HNSW (Add/Search/SearchLarge), BM25 (AddDocument/Search/SearchLargeCorpus), chunking (SmallDoc/LargeDoc/VeryLargeDoc), and graph traversal (AddEntity/AddRelation/FindPath/Neighbors/TransitiveClosure/CommonNeighbors).
**Action:** Add benchmarks for:
- HNSW vs brute-force similarity search
- BM25 scoring
- Chunking throughput
- Graph traversal performance

---

## Future Phases

### Phase 10: Multi-hop Reasoning Engine
**Goal:** Automated path generation and relationship inference chains.
**Features:**
- `ReasoningEngine` interface
- Configurable depth-limited path exploration
- Inference rules (transitive, symmetric, anti-symmetric)
- Confidence propagation along paths
- Natural language query → graph traversal → inferred answer

---

## Notes

- All phases maintain zero-CGO requirement for core functionality
- SQLite uses `modernc.org/sqlite` (pure Go driver)
- Test coverage target: >80% for all packages
- For future phases and roadmap, see [ROADMAP.md](./ROADMAP.md)
---

## Phase 5: Production Readiness

### Completed

- [x] Comprehensive benchmark tests for all packages:
  - HNSW index (Add, Search, SearchLarge)
  - BM25 (AddDocument, Search, SearchLargeCorpus)
  - Chunking (SmallDoc, LargeDoc, VeryLargeDoc for both Fixed and Recursive)
  - Graph traversal (AddEntity, AddRelation, FindPath, Neighbors, TransitiveClosure, CommonNeighbors)
  - Store operations (Upload, Search, Delete)
- [x] Performance benchmarks running successfully:
  - HNSW: ~876 ns/op for Add, ~69 μs/op for Search
  - BM25: ~2.6 μs/op for AddDocument, ~6 μs/op for Search
  - Chunking: ~81 ns/op for small docs, ~28 μs/op for large docs
  - Graph: ~144 ns/op for AddEntity, ~1 μs/op for FindPath
  - Store: ~3.7 μs/op for Upload, ~2.6 μs/op for Search

### Performance Highlights

- HNSW search scales well: 1K docs ~69 μs, 10K docs ~713 μs
- BM25 search is fast: ~6 μs/op for 1K corpus
- Chunking is efficient: ~81 ns/op for small documents
- Graph operations are optimized: ~144 ns/op for entity addition
- Store operations are efficient: ~3.7 μs/op for upload

### Next Steps

- Consider adding more benchmarks for edge cases
- Profile hot paths for further optimization
- Add comparison benchmarks between HNSW and brute-force

---

## Phase 10: Multi-hop Reasoning Engine

### Completed

- [x] Enhanced ReasoningEngine with advanced features:
  - Better natural language query processing
  - Entity matching with fuzzy matching
  - Confidence threshold filtering
  - Path ranking by confidence

- [x] Additional Inference Rules:
  - `InverseRule` - Infer inverse relations (e.g., works_at → works_for)
  - `CompositionRule` - Compose relations (e.g., located_in + works_at → works_in_location)
  - `CommonInterestRule` - Infer common interests based on shared relations
  - `HierarchyRule` - Handle hierarchical relations (is_a, part_of)

- [x] Confidence Propagation:
  - `ProductAggregator` - Multiplies confidence scores with decay
  - `MinAggregator` - Takes minimum confidence score with decay
  - `AverageAggregator` - Takes average confidence score with decay
  - `DefaultAggregator()` - Returns ProductAggregator with 0.9 decay

- [x] Query Processing:
  - `EntityExtractor` for natural language entity extraction
  - Pattern-based entity recognition (person, location, organization)
  - Query expansion with synonyms (Go → golang, gopher)

- [x] Comprehensive benchmarks:
  - InverseRule: ~23 ns/op
  - CompositionRule: ~150 ns/op
  - Confidence aggregation: ~3-4 ns/op
  - Entity extraction: ~2.9 μs/op
  - Path exploration: ~2.6 μs/op
  - Inference: ~56 μs/op
  - Reasoning: ~258 ns/op

### Performance Highlights

- Inference rules are extremely fast: ~23 ns/op
- Confidence aggregation is efficient: ~3-4 ns/op
- Entity extraction handles natural language: ~2.9 μs/op
- Path exploration scales well: ~2.6 μs/op

---

## Phase 29: CLI Tool

### Completed

- [x] `cmd/recall` cobra CLI with two execution modes:
  - **local** — in-process against the configured store (SQLite or memory)
  - **server** — typed HTTP client of a running recall-server (`--server` flag or `cli.url`)
- [x] Core commands: `upload` (files + recursive directories), `search`, `hybrid-search`, `rag`, `graph` (+ `graph list`), `reason` (NL query or `--from`/`--to` path exploration)
- [x] Management commands: `store info`, `store migrate` (versioned SQL migration files), `store backup` (online VACUUM INTO), `store restore` (atomic rename), `cluster status` (node /diagnostics probes, exit 1 on down nodes)
- [x] Evaluation commands: `eval` (Precision/Recall/MRR/NDCG@K, `--save` report), `eval compare` (tolerance-gated regression check, exit 2 — CI gate)
- [x] Output formatting: table, JSON, YAML (`-o/--output` or `cli.output`)
- [x] Configuration: `--config` → `$HOME/.recall.yaml` (.yml/.json) → defaults; `RECALL__SECTION__KEY` env overrides; new `cli` config section (url, api_key, timeout, output, cluster_nodes)
- [x] New `app` package — shared service assembly so recall-server and the CLI wire components identically (recall-server refactored onto it)
- [x] New `client` package — typed REST client for the recall-server API (CLI server-mode transport)
- [x] Tests: `cmd/recall` at 87.6% coverage (end-to-end local + server mode, exit codes, config resolution), `app` 95.9%, `client` 84.8%, `config` 92.8%

---

## Phase 32: Project Hygiene & Documentation

### Completed (2026-08-19)

- [x] **Linting** — `.golangci.yml` (schema v2): default linters (errcheck, govet, ineffassign, staticcheck, unused) + gosec, misspell, revive, unconvert; gofmt as formatter. All exclusions are narrow and documented inline (test-file idioms, deferred-close/rollback cleanup, non-sensitive file permissions, md5 for hash-ring sharding, errcheck-covered G104). Fixed every real finding in code (llm message conversions, ONNX type-switch, sqlite graph error handling, doc comments, example error checks, four analyzed gosec false positives via targeted `//nolint:gosec`). Result: **`golangci-lint run` = 0 issues** (baseline was 587).
- [x] **CI/CD** — `.github/workflows/go.yml`: Lint job (pinned golangci-lint), Build & Test job (vet, gofmt check, build, `go test -race`, overall ≥80% coverage gate, coverage artifact, bench smoke), Vulnerability & License Scan job (govulncheck, go-licenses), Benchmark Regression job (`scripts/benchcompare.sh` baseline gate). `.github/workflows/tag.yml`: manual semver bump → annotated tag. `.github/workflows/release.yml`: tag-triggered build/test + cross-platform binaries via `scripts/release-build.sh` + GitHub release with generated notes.
- [x] **Scripts** — `scripts/benchcompare.sh` (benchmark baseline/compare), `scripts/release-build.sh` (cross-platform release artifacts); both executable, `bash -n` clean.
- [x] **Governance docs** — `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `ARCHITECTURE.md`, `GOVERNANCE.md`.
- [x] **Per-package READMEs** — all 32 packages (30 library + 2 cmd), each with purpose, key types/constructors, minimal usage, and cross-links.
- [x] **Examples** — `example/e2e` (deterministic offline tutorial: ingest → search → RAG → evaluation → graph/reasoning), `example/production` (in-process API server on an ephemeral port driven by the typed `client`), `example/README.md` index; all verified by running them.
- [x] **Guides** — `docs/BENCHMARKS.md` (benchmark comparison guide), `docs/MIGRATION.md` (version upgrade guide).
- [x] **Changelog** — `CHANGELOG.md` established (Keep a Changelog format).
- [x] **Status docs** — `README.md` (Current Status + Testing sections), `ROADMAP.md` (Phase 32 complete, state table), this file.

- Reasoning engine provides fast results: ~258 ns/op
