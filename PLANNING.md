# Recall — Development Plan

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
**Status:** No tests exist; only interface + mock implementation.
**Action:** Add unit tests for MockEmbedder (dimension validation, embedding generation consistency).

### 2. Pluggable NER + Relation Extraction (`graph/`)
**Priority:** High
**Status:** Current extraction uses capitalized-word heuristics only.
**Action:**
- Define `NERExtractor` interface with `Extract(text string) ([]*Entity, error)`
- Add pattern-based relation extractor (e.g., "X works at Y", "X is CEO of Y")
- Pluggable so users can bring their own NER models later

### 3. SQLite Graph Persistence (`store/`)
**Priority:** Medium
**Status:** GraphStore is memory-only.
**Action:** Add `SQLiteGraphStore` that persists entities and relations to SQLite tables.

### 4. Example Package (`example/`)
**Priority:** Medium
**Status:** Referenced in README architecture diagram but doesn't exist.
**Action:** Create `example/` with usage examples for common patterns (upload, search, hybrid, graph queries).

### 5. GoDoc Documentation
**Priority:** Medium
**Status:** Public APIs lack godoc-friendly comments.
**Action:** Add package-level and function-level doc comments for all public types and methods.

### 6. Benchmark Tests (`*_test.go`)
**Priority:** Medium
**Status:** No `Benchmark*` functions exist.
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