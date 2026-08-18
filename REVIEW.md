# Project Review — Recall

_Reviewed: 2026-07-18 · Scope: full codebase (~25.7k LOC, 18 packages)_

## Overview & Health

| Check | Result |
|---|---|
| `go build ./...` | ✅ passes |
| `go vet ./...` | ✅ clean |
| `go test ./... -count=1` | ✅ all pass (706 test funcs, ~13.7k test LOC) |
| `gofmt -l .` | ✅ clean (sweep committed 2026-08-17, 12 files) |
| Coverage ≥80% target | ❌ 6 packages below (see table) |
| CI (`.github/workflows/go.yml`) | ✅ fixed 2026-08-17 — vet, gofmt gate, `-race` tests, smoke bench |

**Coverage vs. the >80% target:**

| Gap | Package | Coverage |
|---|---|---|
| −40 | `llm` | 40.2% |
| −30 | `store` | 49.9% |
| −24 | `distributed` | 55.8% |
| −12 | `core` | 67.7% |
| −9 | `index` | 70.9% |
| −5 | `reasoning` | 74.9% |
| −2 | `chunker` | 78.1% |

---

## 🔴 Critical bugs (correctness)

### ~~1. HNSW silently drops chunks added after activation~~ — *Fixed 2026-07-18 (see action plan #1):*
`index/hnsw.go:53-70` (`MemoryIndex.Add`/`AddBatch`), `store/sqlite.go:171-173` (`SQLiteStore.Upload`)
`buildHNSW()` only runs when `!hnswEnabled`. Once HNSW activates past 1,000 chunks, every
subsequent `Add`/`Upload` goes into `m.chunks` and BM25 but **never into the HNSW graph**, so
new documents become invisible to vector search — the realistic case for a growing corpus.
**Fix:** in `Add`/`AddBatch`, when `hnswEnabled`, also call `m.hnsw.Add(id, embedding)`.

### ~~2. Deleted chunks are still returned from HNSW search~~ — *Fixed 2026-07-18 (see action plan #1):*
`index/hnsw.go:105-124` (`Delete`), `212-224` (`searchHNSW`)
In HNSW mode `Delete` only sets a tombstone (rebuild fires only at >20% tombstone ratio),
but `searchHNSW` never checks `m.deleted[id]` and `Delete` doesn't remove the chunk from
`m.chunks`. Deleted chunks remain searchable until a rebuild; `Count()` and `GetChunk`
still see them too.
**Fix:** filter `m.deleted[id]` in `searchHNSW` and `GetChunk`; consider removing from
`m.chunks` immediately.

### ~~3. HNSW graph construction is not real HNSW~~ — *Fixed 2026-07-18 (see action plan #3):*
`index/hnsw.go:374-436` (`HNSW.Add`)
During construction, `candidates` is seeded with a **single entry node** and never expanded —
there is no greedy/beam search for ef-construction neighbors. Each node gets at most 1
connection per level (plus the symmetric reverse link). The graph is a sparse
"star-of-chains", so ANN recall degrades badly as it grows. Existing tests pass only because
they are small and never assert a recall threshold against brute force.
**Fix:** implement proper ef-construction neighbor search in `HNSW.Add`; add a recall
regression test (e.g. top-100 recall ≥ 0.9 vs brute force on 10k random vectors).

### ~~4. `SQLiteStore` has data races~~ — *Fixed 2026-07-18 (see action plan #2):*
`store/sqlite.go`
`s.mu` is declared but `Upload` writes `s.chunks` (~line 167) and `s.hnsw` (line 172 via
`buildHNSW`) **without the lock**, while `Search`/`searchHNSW` read both lock-free (only
`SearchHybrid`/`GetChunk`/`Delete*`/`Count`/`Namespaces`/`Close` take the lock).
Concurrent `Upload` + `Search` is a genuine race.
**Fix:** guard all `s.chunks`/`s.hnsw` access with `s.mu`; add a concurrent
Upload/Search test (run with `-race` in CI).

### ~~5. Consistent-hash ring is broken~~ — *Fixed 2026-07-18 (see action plan #4):*
`distributed/cluster.go`
- `hashRing` is an append-only slice that is **never sorted**; the ring walks in
  `GetReplicaNodes`/`GetNodeForChunk` do `if vHash >= hash`, which only works on a sorted ring.
  Shard placement is arbitrary and unstable.
- `removeVirtualNodes` deletes map entries but **never prunes the `hashRing` slice** → stale
  entries accumulate forever and can resolve to `c.virtualNodes[hash] == ""` → nil-node deref
  risk in `GetReplicaNodes`.
- The replica walk only scans `vHash > hash` (no wrap-around), so `ReplicationFactor` can't be
  satisfied near the tail of the ring.
- `Rebalance` is an exported no-op `TODO`.
**Fix:** keep `hashRing` sorted (re-sort after add/remove, use `sort.Search` for walks);
prune the slice on node removal; fix replica wrap-around.

### ~~6. `MemoryStore.Upload` double-indexes BM25 and never prunes it~~ — *Fixed 2026-08-17:*
`store/memory.go:99-106`
Chunks go into `MemoryIndex`'s internal BM25 **and** the store's own `s.bm25s` instance.
The index-internal one is never used by any search path (dead memory per chunk); the
store-level one is never `RemoveDocument`-ed on `DeleteChunk`/`DeleteDocument`.
**Fix applied:** kept the *index-internal* BM25 as the single keyword source (the
inverse of the suggested direction, but strictly safer: `MemoryIndex.Delete` prunes
it on every existing and future delete path, so the store can't forget). Removed the
`MemoryStore.bm25s` map entirely; `SearchHybrid` now reads keyword scores through the
new `MemoryIndex.SearchBM25`. Regression tests: `TestMemoryStore_DeleteDocument
PrunesKeywordIndex`, `TestMemoryStore_DeleteChunkPrunesKeywordIndex`,
`TestMemoryIndex_SearchBM25*`.

### ~~7. Hybrid search can never return keyword-only matches~~ — *Fixed 2026-08-17 (see action plan #5):*
`store/memory.go:232-261` (`fuseMap`)
Fusion only iterates the vector TopK and explicitly skips IDs present only in BM25
("Chunk only in BM25 results, skip"). A strong keyword match with weak vector similarity is
silently dropped — hybrid degrades to vector search with BM25 re-ranking.
**Fix:** look up BM25-only chunks from the index and include them in fusion.
Also fixed in the same change: `SQLiteStore.SearchHybrid` was never hybrid at all — the
`chunks_fts` FTS5 table was never created or populated, so `searchFTS5` always failed
silently and hybrid degraded to pure vector; `fuseFTS5Results` also applied the
`BM25Weight` inverts (weight was the *vector* weight, contradicting
`SearchOptions` docs) and fed RRF a singleton per-chunk vector map, making its vector
contribution a constant. A dead, identically-buggy `fuseScores` helper was removed.

---

## 🟠 Medium issues

- **`HNSWThreshold` hardcoded to 1,000** (`index/hnsw.go:32`), not configurable; README claims
  "100K+ chunks" which isn't credible until bugs 1–3 are fixed.
- **"Multi-namespace" is misleading** — `core.Document` has no namespace field;
  `MemoryStore.Upload` uses `ns := s.config.Namespace`, so the `indexes` map always holds
  exactly one entry. Isolated namespaces within one store are not actually supported; users
  must create separate store instances. Decide: implement per-document namespaces or fix README.
- **~~`_ = idx.Delete(...)`~~ at `store/memory.go:313`** — *Fixed 2026-08-17 (bug 6):*
  `DeleteDocument` now captures and returns the first `Delete` error. Same file still
  returns the wrong sentinel (`ErrInvalidChunk` for a nil document, line 52) — open.
- **`context.Background()` in user-facing paths** — `store/memory.go:289,313` and
  `store/sqlite.go:475-504` discard the caller's context.
- **~~Hand-rolled mocks in production packages~~ — *Fixed 2026-08-17:*** the six
  `mockery`-generated files (`store/mock_Store.go`, `store/mock_GraphStore.go`,
  `store/mock_GraphPersistence.go`, `core/mock_Value.go`, `chunker/mock_Chunker.go`,
  `chunker/mock_Factory.go`, 791 lines) are gone: the five that were **unused** across
  the repo were deleted; `chunker.MockChunker`'s one consumer (`store/memory_test.go`) now uses a
  local unexported `mockChunker` in the test file. No mock types remain in library API.
- **`HNSW` RNG** — `rand.New(rand.NewSource(42))` (`index/hnsw.go:315`): deprecated
  constructor (Go ≥1.20) and a fixed seed gives every index identical layer assignments.
- ✅ **`HNSW.Search`** implemented its own min-heap with O(n) scan + full `sort.Slice` per
  iteration — replaced with `container/heap` via the fix-3 `searchLayer` rewrite.
- **`GetChunk` returns internal `*core.Chunk` pointers** (both stores) — callers can mutate
  index state. Consider returning copies.
- **`llm` package (40.2% coverage)** — `openai.go`/`ollama.go` HTTP paths thinly tested
  (error mapping, headers, retries); highest risk of silent breakage.

---

## 🟡 Minor / process

- **CI**: ~~fix `go test -bench=.` → `go test ./... -bench=. -run=^$` (or drop the step); add
  `go vet ./...`, a `gofmt -l` check, and `go test -race ./...` steps; consider `-count=1`.~~
  — *Fixed 2026-08-17:* `.github/workflows/go.yml` now runs `go vet`, a `gofmt -s -l`
  gate, `go test ./... -count=1 -race`, and a smoke bench
  (`-run=^$ -bench=. -benchtime=1x`).
- **~~13 unformatted files~~ — *Fixed 2026-08-17:*** `cache/` (5), `chunker/` (2), `distributed/` (4), `graph/` (2) —
  `gofmt -s -w .` fixed all (12 at sweep time; `distributed/cluster.go` had already been
  reformatted by the fix-4 ring rewrite).
- **Doc drift**: `.clinerules` says module `github.com/deagy/sdk/recall`; actual go.mod is
  `github.com/deagy/recall`. README "100K+ chunks" claim needs the HNSW fixes to be credible.
- **`coverage.out`** committed at repo root — build artifact; gitignore it.
- Some commit messages contain literal `\n` strings (e.g. `a2a731e`) — cosmetic.

---

## ✅ What's genuinely good

- **Clean architecture** — interface-driven layering (`Embedder`, `Chunker`, `Fusion`,
  `InferenceRule`, `GraphStore`) with dependency injection; usable with no network deps.
- **Strong test culture** — 706 tests, benchmarks in every perf-sensitive package, edge-case
  test files, 9 packages at/above the 80% bar.
- **Consistent discipline** — wrapped errors (`%w`), sentinel errors in `core/`, `RWMutex`
  used consistently in `MemoryStore`/`MemoryIndex`/`Cluster`, no panics in real code paths,
  zero-CGO holds (pure-Go sqlite driver).
- **Good documentation** — README/PLANNING/ROADMAP/IMPROVEMENT_PLAN kept in sync per phase.

---

## Prioritized action plan

1. ✅ **HNSW incremental add + tombstone filtering** (bugs 1, 2) — *Done 2026-07-18:*
   `MemoryIndex.Add`/`AddBatch` insert into the graph after activation via `syncHNSW`;
   `Delete` removes the chunk from `m.chunks`/BM25 so searches, `Count`, and `GetChunk`
   never see it (tombstone rebuild retained); new `HNSW.Contains`; `SQLiteStore` mirror kept
   in sync on upload and pruned on `DeleteChunk`/`DeleteDocument`. Regression tests:
   `index/memory_hnsw_test.go`, `store/sqlite_hnsw_test.go`, `TestHNSW_Contains`.
2. ✅ **SQLiteStore locking** (bug 4) — *Done 2026-07-18:* `Upload` takes the write lock
   around mirror/HNSW updates; `searchHNSW` and the `Search` HNSW gate hold the read lock;
   `buildHNSW` requires the caller to hold it. Concurrency test:
   `TestSQLiteStore_ConcurrentUploadAndSearch` (4 writers × 300 docs + live reader, crosses
   the HNSW threshold). The test also surfaced a second defect — fixed in the same change:
   `:memory:` + pooled connections gave each connection a separate in-memory DB
   ("no such table: chunks" under concurrency); `NewSQLiteStore` now uses
   `db.SetMaxOpenConns(1)` — a second, previously unlisted defect caught by the new test.
3. ✅ **Real HNSW construction** (bug 3) — *Done 2026-07-18:* rewrote `HNSW.Add` with a
   proper `searchLayer` ef-construction beam search (`container/heap`, also closes the
   "hand-rolled min-heap" medium item), greedy descent between layers, M/M0-capped
   bidirectional links with a reserved slot for the new back-link (pure top-M trimming
   evicted fresh nodes and left them unreachable), and correct entry-point handling for
   newly opened levels. `Search` reuses the same beam. Regression tests:
   `index/hnsw_recall_test.go` — `TestHNSW_Recall_Clustered` (recall@10 ≥ 0.8 vs brute
   force + layer-0 degree sanity) and `TestHNSW_Recall_IncrementalInserts`; both fail
   against the legacy single-neighbor graph.
4. ✅ **Sorted hash ring + proper removal** (bug 5) — *Done 2026-07-18:* `hashRing`
   is now a sorted invariant maintained with `sort.Search` insert/delete
   (`O(ring)` shifts per vnode, fine at 150 vnodes × few nodes); `removeVirtualNodes`
   prunes the slice as well as the map; `GetReplicaNodes` walks the ring with
   wrap-around via `ringIndex` and returns up to `min(RF, node count)` distinct
   nodes with the primary first; `GetNodeForChunk` reuses the same walk; `Rebalance`
   is a real idempotent rebuild (self-heals a drifted ring, honors `ctx`).
   Regression tests: `distributed/cluster_ring_test.go` — determinism across
   insertion order, ring sorted/consistent after churn, RF satisfied with
   wrap-around (primary == `GetNodeForChunk`), RF capping at 1 node and nil on an
   empty cluster, Rebalance idempotency + self-healing + cancellation, and key
   distribution sanity. All fail against the legacy unsorted append-only ring.
5. ✅ **Hybrid fusion** (bug 7) — *Done 2026-08-17:* `fuseMap` now takes a lookup
   callback and resolves BM25-only IDs from the index (deleted chunks stay excluded);
   custom fusion runs once over the full score maps instead of per-ID.
   `SQLiteStore` got a real FTS5 pipeline: `chunks_fts` (external content table,
   `content_rowid`) + insert/delete/update triggers keep it in sync with `chunks`;
   `fuseFTS5Results` honors the documented `BM25Weight` semantics (0 = pure vector,
   1 = pure BM25, previously inverted) and fuses once over full maps so RRF sees real
   ranks; FTS5 `rank` (negative bm25, lower is better) is sign-flipped to higher-is-
   better; `LIMIT 0` when `TopK` unset is defaulted to 10; the dead buggy `fuseScores`
   helper was removed. Regression tests: `store/hybrid_fusion_test.go` — fuseMap unit
   tests (BM25-only inclusion, deleted-chunk exclusion, pure-vector boundary, RRF math)
   plus end-to-end tests for both stores (pure-BM25 ranks keyword match first,
   BM25-only chunks returned, RRF includes FTS-only match, SQLite pure-vector parity,
   deleted chunk not resurrected via the keyword side). Stash-verified: both
   end-to-end tests fail against the pre-fix code (memory dropped the keyword-only
   chunk; SQLite "pure BM25" returned a distractor because no FTS table existed).
6. **Housekeeping** — ~~`gofmt -s -w .`~~ ✅ *Done 2026-08-17* (12 files, alignment-only
   diff, full test suite green); ~~de-duplicate BM25 + prune on delete~~ ✅ *Done
   2026-08-17 (bug 6 — index-internal BM25 is the single keyword source; store-level
   `s.bm25s` removed; `DeleteChunk`/`DeleteDocument` prune via `MemoryIndex.Delete`;
   regression tests in `index/memory_test.go` + `store/hybrid_fusion_test.go`)*;
   ~~fix CI bench step~~ ✅ *Done 2026-08-17* (vet + gofmt gate + `-race` + smoke bench);
   ~~move mocks to test files~~ ✅ *Done 2026-08-17* (six unused mockery files deleted;
   `MockChunker` inlined as a test-local double);
   remaining: raise `store`/`llm`/
   `distributed` coverage, decide multi-namespace (implement or correct README).

