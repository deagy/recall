# Recall — Improvement Plan

> **Created:** 2026-08-15
> **Goal:** >80% test coverage in every library package, no known correctness bugs in the core, `gofmt` clean, and benchmarks that do not regress.
> **Status:** Phase 1 — Formatting & Quick Wins

---

## Executive Summary

Recall is a well-architected Go RAG library with 10/11 phases complete. The codebase has clean interfaces, good layering, and thread safety. However, before it can be considered production-ready, the following must be addressed:

| Category | Issues Found | Priority |
|----------|-------------|----------|
| Formatting | `gofmt` violations in 3+ files | Critical |
| Module Path | `go.mod` says `github.com/deagy/recall` but directory is `sdk/recall` | Critical |
| Bugs | 4 bugs in SQLite hybrid search, fuse, and reasoning | High |
| Performance | O(n²) insertion sorts, O(n log n) HNSW rebuilds on delete | Medium |
| Test Coverage | 6 packages below 80% target (store at 51.8%) | High |
| Production Readiness | Weak NER, no HNSW in SQLite, no context cancellation | Medium |

**Estimated effort:** 40–60 hours of focused work across 5 phases.

---

## Phase 1: Formatting & Quick Wins

**Goal:** Clean code baseline. No behavioral changes.
**Estimated effort:** 1–2 hours

### Task 1.1: Fix `gofmt` Formatting Violations

**Files affected:**
- `bm25/bm25.go` — single-line if statements (lines 53–54, 73, 80), struct field alignment (lines 56–59)
- `embedder/embedder.go` — formatting around `sqrt` function
- `chunker/chunker_test.go` — spacing in test config

**Action:**
```bash
gofmt -s -w .
```

**Acceptance criteria:**
- `gofmt -d .` returns no output
- All tests still pass

### Task 1.2: Fix Module Path Mismatch

**Current state:**
- `go.mod`: `module github.com/deagy/recall`
- Working directory: `/home/deagy/sdk/recall`
- `.clinerules` references: `github.com/deagy/sdk/recall`

**Decision needed:** Choose one path and align everywhere.

**Option A — Keep `github.com/deagy/recall`** (recommended if repo is at that path):
- No code changes needed; imports already use this path
- Update `.clinerules` to reference `github.com/deagy/recall`
- Directory name doesn't affect Go module resolution

**Option B — Change to `github.com/deagy/sdk/recall`:**
- Update `go.mod` module declaration
- Update ALL import paths across 11 packages (40+ files)
- Update README examples

**Recommendation:** Option A — the imports are already consistent with `github.com/deagy/recall`.

**Acceptance criteria:**
- `go build ./...` succeeds
- No stale import references

### Task 1.3: Replace Insertion Sort with `sort.Slice`

**Files affected:**
- `store/memory.go:351-361` — `sortResults()` function
- `store/sqlite.go:549-559` — `sortResultsByScore()` function

**Current code (memory.go):**
```go
func sortResults(results []index.SearchResult) {
    for i := 1; i < len(results); i++ {
        key := results[i]
        j := i - 1
        for j >= 0 && results[j].Score < key.Score {
            results[j+1] = results[j]
            j--
        }
        results[j+1] = key
    }
}
```

**Fix:**
```go
import "sort"

func sortResults(results []index.SearchResult) {
    sort.Slice(results, func(i, j int) bool {
        return results[i].Score > results[j].Score
    })
}
```

**Acceptance criteria:**
- Same sort behavior (descending by score)
- `go test ./store/...` passes
- Benchmark shows improvement for >50 results

### Task 1.4: Replace Custom `sqrt` with `math.Sqrt`

**File affected:** `embedder/embedder.go`

**Current code (lines 101–111):**
```go
func sqrt(x float64) float64 {
    if x <= 0 { return 0 }
    z := x / 2
    for i := 0; i < 20; i++ {
        z -= (z*z - x) / (2 * z)
    }
    return z
}
```

**Fix:**
1. Add `"math"` to imports
2. Replace all calls to `sqrt(...)` with `math.Sqrt(...)`
3. Delete the custom `sqrt` function

**Acceptance criteria:**
- `go test ./embedder/...` passes
- Cosine similarity values match within floating-point precision

---

## Phase 2: Bug Fixes

**Goal:** Eliminate all identified bugs. No new features.
**Estimated effort:** 4–6 hours

### Task 2.1: Fix SQLite Hybrid Search BM25 Fallback (Dead Code)

**File:** `store/sqlite.go`

**Bug:** In `SQLiteStore.Upload()`, the per-namespace `bm25s` map is created but never populated. The hybrid search path queries FTS5 for BM25 scores, but the fallback `bm25s[ns]` is always empty. The SQLite store has its own search implementation that bypasses `MemoryIndex` entirely.

**Fix options:**
- **Option A (recommended):** Remove the `bm25s` field from `SQLiteStore` and rely solely on FTS5 for keyword search. Remove the dead code path in `fuseScores` that falls back to empty BM25 scores.
- **Option B:** Populate `bm25s` in `SQLiteStore.Upload()` by calling `bm25s[ns].AddDocument()` for each chunk.

**Acceptance criteria:**
- `SearchHybrid` returns meaningful BM25 scores
- No dead code paths that return empty results silently

### Task 2.2: Fix `fuseScores` Alpha=0 Override

**File:** `store/sqlite.go:519-523`

**Current code:**
```go
alpha := opts.BM25Weight
if alpha == 0 {
    alpha = 0.5 // default
}
fusedScore = alpha*vecScore + (1-alpha)*bm25Score
```

**Bug:** When `BM25Weight` is explicitly set to `0` (meaning "pure vector similarity, no BM25"), it gets overridden to `0.5`.

**Fix:**
```go
if opts.BM25Weight == 0 {
    // Pure vector similarity — skip BM25 fusion
    fusedScore = vecScore
} else {
    alpha := opts.BM25Weight
    fusedScore = alpha*vecScore + (1-alpha)*bm25Score
}
```

**Also fix same bug in:** `store/memory.go` — check if `SearchHybrid` has the same logic.

**Acceptance criteria:**
- `BM25Weight=0` returns pure vector scores
- `BM25Weight=1` returns pure BM25 scores
- `BM25Weight=0.5` returns equal weighting
- Test verifies all three cases

### Task 2.3: Fix `TransitiveRule` Semantics

**File:** `reasoning/inference.go:18-34`

**Current code:**
```go
// TransitiveRule infers that if A->B and B->C, then A->C.
type TransitiveRule struct { ... }
func (r *TransitiveRule) Apply(rel *graph.Relation) (*graph.Relation, bool) {
    if rel.Weight < r.MinWeight { return nil, false }
    inferred := graph.NewRelation(rel.To, rel.From, rel.Type+"_(reverse)", rel.Weight*0.8)
    return inferred, true
}
```

**Bug:** Doc says "A→B, B→C ⇒ A→C" (true transitivity) but implementation generates a **reversed** relation `B→A` with weight decay. This is a "symmetric reverse" rule, not transitivity.

**Fix:**
1. Rename `TransitiveRule` → `ReverseRule` to match actual behavior
2. Update doc comment: "Generates a reverse relation B→A from A→B with weight decay"
3. Add a NEW `TransitiveRule` that takes two relations (requires API change)
4. Update `DefaultRules()` accordingly

**Acceptance criteria:**
- Rule names match their actual behavior
- Tests verify reverse relation generation
- New transitive rule (if added) correctly infers A→C from A→B, B→C

### Task 2.4: Fix HNSW Rebuild on Every Delete

**File:** `index/hnsw.go:108-110`

**Current code:**
```go
func (m *MemoryIndex) Delete(_ context.Context, id string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    delete(m.chunks, id)
    m.bm25.RemoveDocument(id)
    if m.hnswEnabled {
        m.buildHNSW()  // O(n log n) full rebuild
    }
    return nil
}
```

**Bug:** Every delete triggers a full HNSW graph rebuild. For a 10K chunk index, this is ~50K operations per delete.

**Fix:** Mark the node as deleted (tombstone) and rebuild only when the ratio of tombstones to live nodes exceeds a threshold (e.g., 20%).

```go
type MemoryIndex struct {
    // ... existing fields ...
    deletedIDs       map[string]bool
    tombstoneThreshold float64  // default 0.2
}

func (m *MemoryIndex) Delete(_ context.Context, id string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    delete(m.chunks, id)
    m.bm25.RemoveDocument(id)
    if m.hnswEnabled {
        if m.deletedIDs == nil {
            m.deletedIDs = make(map[string]bool)
        }
        m.deletedIDs[id] = true
        if float64(len(m.deletedIDs)) / float64(len(m.chunks)+len(m.deletedIDs)) > m.tombstoneThreshold {
            m.buildHNSW()
            m.deletedIDs = make(map[string]bool)
        }
    }
    return nil
}
```

**Acceptance criteria:**
- Single delete is O(1) for HNSW-enabled indexes
- Full rebuild happens automatically when tombstone ratio exceeds threshold
- Search results are correct after rebuild
- Tests verify delete + search correctness

---

## Phase 3: Performance & Robustness

**Goal:** Make the library survive the failure modes a production workload actually hits — concurrent access, cancellation mid-query, and a corpus larger than memory.
**Estimated effort:** 6–10 hours

### Task 3.1: Add Context Cancellation Support

**Files affected:**
- `store/sqlite.go` — `Search()`, `SearchHybrid()`, `Upload()`
- `store/memory.go` — `Search()`, `SearchHybrid()`, `Upload()`
- `index/hnsw.go` — `Search()`

**Current state:** Methods accept `ctx context.Context` but never check `ctx.Done()`.

**Fix for SQLite store:**
```go
func (s *SQLiteStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
    rows, err := s.db.QueryContext(ctx, `...`)
    if err != nil { return nil, err }
    defer rows.Close()
    for rows.Next() {
        select {
        case <-ctx.Done():
            return nil, ctx.Err()
        default:
        }
        // ... scan and process ...
    }
}
```

**Acceptance criteria:**
- SQLite queries respect context cancellation
- Long-running searches can be interrupted
- No goroutine leaks on context cancel

### Task 3.2: Add HNSW Support to SQLite Store

**File:** `store/sqlite.go`

**Current state:** SQLite store does brute-force cosine similarity over all chunks loaded from DB. No ANN optimization.

**Fix:** Add an in-memory HNSW index that mirrors the SQLite data for search purposes.
1. Add a `hnswIndex *index.HNSW` field to `SQLiteStore`
2. On `Upload()`, after storing in SQLite, also add to HNSW
3. On `DeleteChunk()`, mark as deleted in HNSW (tombstone approach from Task 2.4)
4. In `Search()`, use HNSW for candidate retrieval, then verify against SQLite

**Acceptance criteria:**
- SQLite store search is faster than brute-force for >1K chunks
- Results match brute-force search (within HNSW approximation tolerance)
- HNSW is rebuilt when tombstone ratio exceeds threshold

### Task 3.3: Improve Entity Extraction Heuristics

**File:** `store/graph_store.go:45-70`

**Current state:** Entity extraction is a trivial uppercase-letter heuristic that produces garbage (extracts "Go", "Is", "It", etc.).

**Fix:** Implement a basic NER using:
1. **Stopword filtering** — exclude common words (is, the, a, an, etc.)
2. **Multi-word entity detection** — consecutive capitalized words form one entity
3. **Minimum length** — require at least 2 characters
4. **Pluggable extractor interface** — define `NERExtractor` interface for future LLM-based extraction

```go
type NERExtractor interface {
    Extract(text string) ([]*graph.Entity, error)
}

type HeuristicNER struct {
    Stopwords map[string]bool
    MinLength int
}

func (n *HeuristicNER) Extract(text string) ([]*graph.Entity, error) {
    // Tokenize, filter stopwords, group consecutive caps
}
```

**Acceptance criteria:**
- "Alice works at Google" extracts [Alice, Google] (not [Alice, Works, At, Google])
- "Go is a programming language" extracts [Go] (not [Go, Is, A, Programming, Language])
- Pluggable interface allows future LLM-based extraction

---

## Phase 4: Test Coverage

**Goal:** Achieve >80% coverage for all packages.
**Estimated effort:** 16–24 hours

### Current Coverage Status

| Package | Current | Target | Gap |
|---------|---------|--------|-----|
| bm25 | 96.6% | >80% | Done |
| embedder | 95.5% | >80% | Done |
| fuse | 86.9% | >80% | Done |
| reasoning | 72.9% | >80% | +7.1% |
| chunker | 71.0% | >80% | +9.0% |
| graph | 69.6% | >80% | +10.4% |
| index | 69.7% | >80% | +10.3% |
| core | 66.2% | >80% | +13.8% |
| pipeline | 65.0% | >80% | +15.0% |
| store | 51.8% | >80% | +28.2% |

### Task 4.1: Store Package Tests (Target: >80%)

**File:** `store/memory_test.go`, `store/sqlite_test.go`

**Missing test cases:**
1. `TestSQLiteStore_UploadAndSearch` — full upload → search roundtrip
2. `TestSQLiteStore_SearchHybrid` — hybrid search with BM25 weights
3. `TestSQLiteStore_DeleteChunk` — delete and verify removal
4. `TestSQLiteStore_DeleteDocument` — cascade delete
5. `TestSQLiteStore_MultipleNamespaces` — namespace isolation
6. `TestSQLiteStore_ContextCancellation` — cancel during search
7. `TestMemoryStore_HybridSearchEdgeCases` — BM25Weight=0, BM25Weight=1
8. `TestMemoryStore_DeleteRebuildHNSW` — verify HNSW after delete
9. `TestSQLiteStore_FTS5Search` — verify FTS5 is actually used
10. `TestSQLiteStore_MetadataFilter` — filter by metadata in SQLite

**Estimated tests:** 15–20 new tests

### Task 4.2: Pipeline Package Tests (Target: >80%)

**File:** `pipeline/rag_test.go`

**Missing test cases:**
1. `TestRAGPipeline_QueryWithMinScore` — filter by minimum score
2. `TestRAGPipeline_QueryHybrid` — hybrid search in pipeline
3. `TestRAGPipeline_MaxTokensTruncation` — context window truncation
4. `TestRAGPipeline_CustomTemplate` — custom system/user templates
5. `TestContextWindow_EmptyChunks` — empty content handling
6. `TestContextWindow_TokenEstimation` — verify token estimation accuracy
7. `TestTemplate_RenderSystemOnly` — render system prompt separately
8. `TestTemplate_RenderUserOnly` — render user prompt separately

**Estimated tests:** 8–10 new tests

### Task 4.3: Core Package Tests (Target: >80%)

**File:** `core/chunk_test.go`, `core/document_test.go`, `core/value_test.go`

**Missing test cases:**
1. `TestChunk_GetMetadata_NilMap` — nil metadata map
2. `TestChunk_GetMetadataString_Empty` — missing key returns ""
3. `TestDocument_AddTag_Duplicate` — duplicate tag not added
4. `TestDocument_AddTag_New` — new tag added
5. `TestValue_ToFloat64_Number` — Number type conversion
6. `TestValue_ToFloat64_String` — String-to-float conversion
7. `TestValue_ToFloat64_Invalid` — non-numeric string
8. `TestValue_ToBool_Boolean` — Boolean type conversion
9. `TestValue_ToBool_String` — String-to-bool conversion
10. `TestValue_ToBool_Invalid` — non-boolean string

**Estimated tests:** 10 new tests

### Task 4.4: Index Package Tests (Target: >80%)

**File:** `index/memory_test.go`, `index/hnsw_test.go`

**Missing test cases:**
1. `TestMemoryIndex_AddBatch_EmbeddingMismatch` — dimension mismatch error
2. `TestMemoryIndex_AddBatch_NilEmbedding` — nil embedding error
3. `TestMemoryIndex_Delete_HNSWEnabled` — delete with HNSW active
4. `TestMemoryIndex_Delete_TombstoneRebuild` — verify tombstone threshold
5. `TestMemoryIndex_Search_EmptyIndex` — search with no chunks
6. `TestMemoryIndex_Search_FilterMatch` — filter matches
7. `TestMemoryIndex_Search_FilterNoMatch` — filter excludes
8. `TestHNSW_Delete_TombstoneRatio` — verify rebuild at threshold
9. `TestHNSW_Search_LargeDataset` — search 10K+ vectors
10. `TestHNSW_AddBatch_Performance` — batch add benchmark

**Estimated tests:** 10 new tests

### Task 4.5: Graph Package Tests (Target: >80%)

**File:** `graph/graph_test.go`

**Missing test cases:**
1. `TestKnowledgeGraph_RemoveEntity_WithRelations` — remove entity and verify relations removed
2. `TestKnowledgeGraph_FindPath_Bidirectional` — path via incoming edges
3. `TestKnowledgeGraph_TransitiveClosure_Disconnected` — disconnected component
4. `TestKnowledgeGraph_CommonNeighbors_NoCommon` — no common neighbors
5. `TestKnowledgeGraph_ShortedPathLength_NoPath` — returns -1
6. `TestKnowledgeGraph_ShortedPathLength_SameNode` — returns 0
7. `TestKnowledgeGraph_FindEntitiesByLabel_CaseInsensitive` — case insensitive search
8. `TestKnowledgeGraph_OutgoingRelations_Copy` — returned slice is a copy

**Estimated tests:** 8 new tests

### Task 4.6: Reasoning Package Tests (Target: >80%)

**File:** `reasoning/engine_test.go`, `reasoning/inference_test.go`

**Missing test cases:**
1. `TestTransitiveRule_Rename` — verify renamed rule works
2. `TestTransitiveRule_BelowMinWeight` — returns nil
3. `TestEngine_ExplorePaths_DepthLimit` — respects maxDepth
4. `TestEngine_ExplorePaths_NoPath` — returns nil
5. `TestEngine_Reason_EmptyQuery` — no capitalized words
6. `TestEngine_InferRelations_DuplicatePrevention` — same relation not inferred twice
7. `TestEngine_Reason_MultiHop` — multi-hop path exploration
8. `TestSymmetricRule_UnmatchedType` — returns nil for non-matching types

**Estimated tests:** 8 new tests

### Task 4.7: Chunker Package Tests (Target: >80%)

**File:** `chunker/fixed_test.go`, `chunker/recursive_test.go`

**Missing test cases:**
1. `TestFixedChunker_EmptyContent` — returns nil
2. `TestFixedChunker_SinglePart` — single paragraph
3. `TestFixedChunker_OverlapGeneration` — verify overlap text
4. `TestFixedChunker_MetadataCopy` — metadata is deep-copied
5. `TestRecursiveChunker_EmptyContent` — returns nil
6. `TestRecursiveChunker_AllSmall` — no splitting needed
7. `TestRecursiveChunker_TripleNewlineSplit` — split by \n\n\n
8. `TestRecursiveChunker_SentenceSplit` — split by ". "

**Estimated tests:** 8 new tests

---

## Phase 5: Production Readiness

**Goal:** Make the library suitable for real-world RAG applications.
**Estimated effort:** 10–16 hours

### Task 5.1: Implement NER Extractor Interface

**File:** Create `graph/extract.go`

```go
// NERExtractor defines the interface for named entity recognition.
type NERExtractor interface {
    Extract(text string) ([]*Entity, error)
}

// HeuristicNER uses rule-based heuristics for entity extraction.
type HeuristicNER struct {
    Stopwords map[string]bool
    MinLength int
}

// Extract extracts entities from text using heuristics.
func (n *HeuristicNER) Extract(text string) ([]*Entity, error) {
    // Tokenize, filter stopwords, group consecutive caps
}

// PatternRelationExtractor extracts relations from text using patterns.
type PatternRelationExtractor struct {
    Patterns []RelationPattern
}

type RelationPattern struct {
    Name   string
    Regex  *regexp.Regexp
    Weight float64
}

// ExtractRelations extracts relations from text using patterns.
func (p *PatternRelationExtractor) ExtractRelations(text string) ([]*Relation, error) {
    // Apply patterns to text
}
```

**Default patterns:**
- `X works_at Y` → `works_at`
- `X located_in Y` → `located_in`
- `X founded_by Y` → `founded_by`
- `X part_of Y` → `part_of`
- `X related_to Y` → `related_to`
- `X taught_by Y` → `taught_by`
- `X parent_of Y` → `parent_of`
- `X ceo_of Y` → `ceo_of`

**Acceptance criteria:**
- `NERExtractor` interface defined and exported
- `HeuristicNER` implements basic NER
- `PatternRelationExtractor` with 8 default patterns
- Tests for each pattern
- Pluggable — users can add custom patterns

### Task 5.2: Add Integration Tests

**File:** Create `example/integration_test.go` or `recall_test.go` at root

**Test scenarios:**
1. **Full RAG Pipeline:** Upload → Search → Context Assembly → Template Render
2. **Hybrid Search Pipeline:** Upload → Hybrid Search → Fusion → Response
3. **Graph RAG Pipeline:** Upload → Entity Extraction → Relation Extraction → Path Finding → Reasoning
4. **SQLite Persistence:** Upload → Close → Reopen → Search → Verify
5. **Multi-Namespace Isolation:** Upload to ns1, ns2 → Search ns1 only → Verify ns2 not included
6. **HNSW Scaling:** Upload 5K chunks → Search → Verify accuracy vs brute-force

**Acceptance criteria:**
- All integration tests pass
- Tests demonstrate end-to-end workflows
- SQLite persistence roundtrip verified

### Task 5.3: Update Documentation

**Files:**
- `README.md` — update Current Status section
- `PLANNING.md` — mark completed improvement tasks
- Package-level doc comments — verify all public APIs documented

**Actions:**
1. Update README Current Status to reflect Phase 1–10 complete, Phase 11 (NER) in progress
2. Add "Production Readiness" section to README with:
   - Known limitations
   - Performance characteristics
   - Recommended usage patterns
3. Update PLANNING.md improvement opportunities as complete

**Acceptance criteria:**
- README accurately reflects current state
- No stale information
- New users can understand limitations

### Task 5.4: Add Benchmark Suite

**Files:**
- `bm25/bm25_bench_test.go` — document add/search at scale
- `index/hnsw_bench_test.go` — HNSW vs brute-force comparison
- `store/memory_bench_test.go` — upload/search throughput
- `graph/graph_bench_test.go` — traversal performance

**Benchmarks to add:**
1. `BenchmarkBM25_AddDocument_10K` — add 10K documents
2. `BenchmarkBM25_Search_10K` — search 10K document index
3. `BenchmarkHNSW_Search_10K_vs_BruteForce` — compare ANN vs exact
4. `BenchmarkMemoryStore_Upload_1K` — upload 1K chunks
5. `BenchmarkMemoryStore_Search_1K` — search 1K chunk index
6. `BenchmarkGraph_FindPath_100Entities` — path finding performance
7. `BenchmarkGraph_TransitiveClosure_100Entities` — closure performance

**Acceptance criteria:**
- Benchmarks run without errors
- Results documented in PLANNING.md
- Clear before/after comparison possible

---

## Execution Order

```
Phase 1 (Formatting) → Phase 2 (Bug Fixes) → Phase 3 (Performance)
                                               ↓
Phase 4 (Test Coverage) ←←←←←←←←←←←←←←←←←←←←←
                                               ↓
Phase 5 (Production Readiness)
```

**Rationale:**
1. Fix formatting first — makes code review easier
2. Fix bugs before adding tests — tests should verify correct behavior
3. Performance improvements before tests — tests should measure final state
4. Test coverage in parallel with Phase 3 — but finalize after bugs are fixed
5. Production readiness last — depends on all previous phases

---

## Verification Checklist

After all phases complete:

- [ ] `gofmt -d .` returns no output
- [ ] `go vet ./...` returns no warnings
- [ ] `go test ./... -count=1` all pass
- [ ] `go test ./... -count=1 -cover` all packages >80%
- [ ] Module path is consistent everywhere
- [ ] SQLite hybrid search returns meaningful BM25 scores
- [ ] `BM25Weight=0` returns pure vector scores
- [ ] HNSW delete doesn't trigger full rebuild for single deletes
- [ ] Context cancellation interrupts long searches
- [ ] NER extractor interface defined and implemented
- [ ] Integration tests pass
- [ ] Benchmarks run successfully
- [ ] README updated
- [ ] `go build ./...` succeeds
- [ ] Example runs without errors

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| HNSW tombstone approach introduces bugs | Extensive tests for delete + search correctness |
| NER heuristic still produces false positives | Make it pluggable; document as "basic heuristic" |
| SQLite FTS5 not available in all builds | Keep BM25 fallback as optional optimization |
| Context cancellation adds complexity | Only add where it matters (SQLite queries) |
| Test coverage regresses during bug fixes | Run coverage after each phase |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Overall test coverage | 72% (avg) | >80% (all packages) |
| `gofmt` violations | 3+ files | 0 |
| Known bugs | 4 | 0 |
| Production-ready features | 0/5 | 5/5 |
| Integration tests | 0 | 6 scenarios |
| Benchmark suites | 2 packages | 4 packages |
