# Recall — Remediation Plan

> Created: 2026-08-19 · Source: full-codebase review (see summary below)
> Companion docs: [REVIEW.md](./REVIEW.md) (2026-07-18 review), [ROADMAP.md](./ROADMAP.md)
> Conventions: follow [.clinerules](./.clinerules) — gofmt/go vet/go test gates, conventional commits, README/PLANNING updates at phase completion.

Review baseline (2026-08-19): `go build` passes · `go vet` passes · `go test ./... -count=1` passes (35 packages) ·
`-race` passes on store/index/cache/distributed/graph/ingest/query/pipeline/tracing/llm (existing tests do **not** exercise the racy paths below) · `gofmt -l` clean.

**Legend:** P0 crash/data-loss class · P1 silent correctness bug · P2 semantics/consistency · P3 smell/cleanup.
Sizes: S ≤ 1h · M ≤ half day · L > half day.

---

## Phase 1 — Data races (P0)

**Status: complete (2026-08-21)** — 1.1 + 1.2 in `ebd2221` (both are the same defect class, S-sized, low-risk). Validation: full repo `go test ./... -count=1` green (32/32 packages); `-race` green for `ingest` (full package) and `store` (all 168 tests, run in chunks; the pre-existing heavyweight `TestSQLiteStore_ConcurrentUploadAndSearch` exceeds the sandbox's 30s race budget and is covered by CI's unlimited `go test -race`); `go vet ./...` and `gofmt -l .` clean; coverage store 85.3% / ingest 92.3%.

Detection caveat (documented in both test comments): the Go runtime detects concurrent map *iteration* vs *write* only via the per-map `writing` flag probe on each iteration step — the maps package carries no `-race` annotations on iteration — so the pre-fix fatal is a probabilistic red, not a deterministic one. `TestIncremental_Save_ConcurrentMark` is a strong guard (high probe rate: concurrent `Mark` writes land densely inside each marshal's iteration); `TestMemoryStore_DeleteDocument_ConcurrentUpload` coordinates the overlap on every attempt (gated embedder + index-lock parking) and maximizes, but does not guarantee, the pre-fix red in a single run.

Goal: eliminate the two confirmed race conditions and lock them in with `-race` regression tests.

### 1.1 `MemoryStore.DeleteDocument` iterates `docChunks` map unlocked
- **File:** `store/memory.go:299-326` (iteration at :314; racing write at :107)
- **Defect:** reads `chunkIDs := s.docChunks[docID]` under `s.mu`, unlocks, then iterates the live map while a concurrent `Upload` writes to it → `fatal error: concurrent map iteration and map write` (reproduced with `-race`).
- **Fix (as shipped):** snapshot the chunk IDs into a fresh slice while holding `s.mu`, then iterate the snapshot. Index deletes stay outside the store lock (they take per-index locks). Preserves the Phase 2.2 not-found/pruning behavior.
- **Tests (as shipped):** `TestMemoryStore_DeleteDocument_ConcurrentUpload` — a `gateEmbedder` parks the Upload in `EmbedBatch` (post-chunking, pre-indexing), two `DeleteDocument` goroutines start iterating the live chunk map, the Upload is released so its `AddBatch` parks the deletes mid-iteration on the index lock and the `docChunks` write burst lands inside the iterations. 30 coordinated attempts.
- **Size:** S · **Risk:** low

### 1.2 `Incremental.Save` marshals the live map outside the lock
- **File:** `ingest/incremental.go:95-125`
- **Defect:** `st.Documents` captures `inc.hashes` by reference; `json.MarshalIndent` runs after `Unlock()` while `Mark()` may write (reproduced with `-race`). Secondary: `ids` slice (lines 105-109) is dead code; `dirty=false` is set *before* marshaling, so an encode/write failure loses the pending state.
- **Fix (as shipped):** copy the map under the lock; marshal the copy; re-arm `dirty` on a failed encode **or** write (dirty is cleared with the snapshot, re-set on failure paths). Removed the unused `ids`/`sort` code (and the `sort` import).
- **Tests (as shipped):** `TestIncremental_Save_ConcurrentMark` under `-race` (200 concurrent `Mark`s racing 200 `Save`s, final quiet save persists all 200); `TestIncremental_Save_FailureKeepsDirty` (unwritable path keeps `dirty`).
- **Size:** S · **Risk:** low

### Phase 1 validation
```bash
go test ./store/... ./ingest/... -race -count=1
go vet ./... && gofmt -l .
```
**Commit:** `fix: eliminate data races in MemoryStore.DeleteDocument and Incremental.Save`


---

## Phase 2 — Core correctness bugs (P1)

**Status: complete (2026-08-19)** — 2.1 `561bf44` · 2.2 `3db8bd4` · 2.3 `77686b9` · 2.4 `4ab47c2` · 2.5 `a4e187b` · docs `0296f1f`. Validation: `go test ./bm25/... ./store/... ./index/... -count=1` green, `-race` green, full repo suite green; coverage bm25 95.8% / store 85.3% / index 91.9%. Note: 2.2 left `DeleteDocument` iterating the live `docChunks[docID]` map — the 1.1 snapshot fix must preserve the new not-found/pruning behavior.

### 2.1 BM25 document-count drift
- **File:** `bm25/bm25.go:94-117` (`AddDocument`), `176-199` (`RemoveDocument`)
- **Defect:** re-adding an existing docID increments `docCount`/`docFreq` again (reproduced: `Count()==2` after two adds of the same ID) and corrupts IDF/avgDocLen; `RemoveDocument` decrements `docCount` even for unknown IDs (reproduced: `Count()==0`). Reachable via `MemoryIndex.Add` overwriting an existing chunk ID.
- **Fix:**
  - `AddDocument`: if docID already exists, first remove its old postings (shared removal helper), then insert — counts stay consistent.
  - `RemoveDocument`: return early when `docLens` has no entry for docID.
- **Tests:** re-add keeps `Count()==1` and correct avgDocLen; remove-unknown is a no-op; scores for unaffected docs unchanged.
- **Size:** S · **Risk:** low-medium (touches scoring inputs; keep score-golden tests green)

### 2.2 `MemoryStore.DeleteChunk` never reports "not found"
- **Files:** `index/hnsw.go:120-142` (`MemoryIndex.Delete` always returns nil), `store/memory.go:287-296`
- **Defect:** reproduced — after an upload, `DeleteChunk(ctx, "no-such-chunk")` returns `nil`; the `core.ErrNotFound` return is dead code; only the first namespace index is touched before returning.
- **Fix:** `MemoryIndex.Delete` returns `core.ErrNotFound` when the ID is absent. `MemoryStore.DeleteChunk` tries all indexes, returns nil if any removed it, else `ErrNotFound`. Also prune the ID from `docChunks` on success so `DeleteDocument` doesn't replay stale IDs.
- **Compatibility note:** nil-for-missing becomes `ErrNotFound` — audit `api` handlers and `cmd` call sites; note in CHANGELOG.
- **Tests:** delete existing → nil + count drops; delete missing → `ErrNotFound`; ID pruned from `docChunks`.
- **Size:** S · **Risk:** medium (behavior change — audit callers)

### 2.3 SQLite timestamps carry a fake UTC marker
- **File:** `store/sqlite.go:211`
- **Defect:** `now.Format("2006-01-02T15:04:05Z")` writes *local* time with a literal `Z` (classic Go layout pitfall — a bare `Z` is not a timezone token).
- **Fix:** `time.Now().UTC().Format(time.RFC3339)`.
- **Tests:** upload with a non-UTC local zone; assert stored value round-trips as UTC.
- **Size:** S · **Risk:** low

### 2.4 Orphaned `embeddings` rows (foreign keys not enforced)
- **File:** `store/sqlite.go` (schema :116-157; deletes :562-615)
- **Defect:** `PRAGMA foreign_keys` is never enabled (SQLite default OFF), so `ON DELETE CASCADE` never fires; deletes leave orphan embedding rows forever.
- **Fix (prefer A):**
  - **A.** Explicitly `DELETE FROM embeddings WHERE chunk_id ...` in the same transaction as `chunks` in `DeleteChunk`/`DeleteDocument` (deterministic, pragma-independent).
  - **B.** Enable `PRAGMA foreign_keys=ON` per connection and keep cascade. With `SetMaxOpenConns(1)` a single post-open exec suffices, but the pragma is per-connection — document the constraint.
- **Tests:** delete chunk/document → `SELECT COUNT(*) FROM embeddings` drops accordingly.
- **Size:** S · **Risk:** low

### 2.5 `MinScore` ignored on several search paths
- **Files:** `store/sqlite.go` — brute-force `Search` (:340-378), `searchHNSW` (:264-305), `fuseFTS5Results` (:500-524); `store/memory.go` `fuseMap` (:208-260)
- **Defect:** inconsistent with `MemoryIndex`/`HybridIndex`, which honor `MinScore`.
- **Fix:** apply `score < opts.MinScore` filtering in all four paths (after fusion for hybrid). Document in `index.SearchOptions.MinScore` that it applies to fused scores in hybrid mode.
- **Tests:** table-driven parity tests: Memory vs SQLite × vector/hybrid × MinScore.
- **Size:** S · **Risk:** low

### Phase 2 validation
```bash
go test ./bm25/... ./store/... ./index/... -count=1 -cover
go test ./store/... -race -count=1
```
**Commits:** one `fix:` commit per item (2.1…2.5) to keep history reviewable.


---

## Phase 3 — Distributed package (P1/P2)

**Status: complete (2026-08-21)** — 3.1+3.2 `8eaa478` · 3.3 `49ffbf8` · 3.4 `92f7b81`. Validation: `go test ./... -count=1` and `CGO_ENABLED=1 go test ./distributed/ -race -count=1` green, including a new concurrent snapshot/write race test. The hybrid signature changes (in-development `distributed` package) are tracked as **Breaking** in CHANGELOG `[Unreleased]` and `docs/MIGRATION.md`.

### 3.1 Replica bookkeeping is broken (P1)
- **File:** `distributed/replication.go:108-140` (`replicatePrimaryReplica`), `237-257` (`GetReplicationStatus`)
- **Defect:** replica shards are created with `CreateShard(node.ID)` (auto ID `shard-<node>-<n>`), but the result reports and `GetReplicationStatus` looks up the never-created ID `<shard>-replica-<node>` → status can never find replicas. Repeated `ReplicateData` calls also create unbounded duplicate shards.
- **Fix:** use `CreateShardWithID(node.ID, replicaShardID)`; make replica creation idempotent (reuse existing replica shard, update its data); guard `replicaNodes[1:]` against the assumption that index 0 is primary (or document/assert it).
- **Tests:** replicate → `GetReplicationStatus` counts the created replicas; second replicate does not grow the shard count.
- **Size:** M · **Risk:** low

### 3.2 Shard ID collisions (P1)
- **File:** `distributed/shard.go:148-169` (`CreateShardWithID`)
- **Defect:** auto ID derives from `len(sm.shards)+1`; after `DeleteShard` the counter shrinks and a new shard can silently overwrite an existing map entry.
- **Fix:** keep a monotonic `nextShardSeq` counter on `ShardManager` (never decreases), or loop until a unique ID is found.
- **Tests:** create 2 shards, delete one, create another → 3 distinct shards, no data loss.
- **Size:** S · **Risk:** low

### 3.3 `Timeout` config is dead (P2)
- **File:** `distributed/scatter_gather.go:23-24` (defaults also in `distributed.go:120,139`)
- **Defect:** `ScatterGatherConfig.Timeout` is documented ("maximum time to wait for all shards") but never applied.
- **Fix:** when Timeout > 0, derive `ctx, cancel := context.WithTimeout(ctx, ...)` for the gather (respecting an earlier ctx deadline); alternatively remove the field — applying it is preferred since it is public config.
- **Bonus:** collapse `ScatterGatherSearch`/`ScatterGatherSearchHybrid` duplication into one fan-out helper parameterized by the per-shard search func.
- **Tests:** a shard that blocks longer than Timeout does not stall the gather.
- **Size:** M · **Risk:** low

### 3.4 Shard search honesty + safety (P2)
- **Files:** `distributed/shard.go:374-397`, `distributed/shard_index.go`
- **Defects:**
  - `Shard.SearchHybrid` is not hybrid (delegates to vector `Search`); `ShardIndex.SearchHybrid` rebuilds a BM25 index per call and scores against a fake 128-dim hashed embedding that cannot match real chunk embeddings.
  - `ShardIndex` methods read `shard.Data` with no locking; safety depends on callers holding the shard lock (undocumented).
  - `sortResultsByScore` in `shard_index.go:226` is O(n²) selection sort.
- **Fix:**
  - Document `ShardIndex` lock requirements in its type doc; make `Shard.Search`/`SearchHybrid` pass a snapshot (map copy under RLock) into `ShardIndex` so standalone use is safe.
  - Either implement true hybrid search (query text + embedding pair, like `index.HybridIndex.Search`) or rename/deprecate the fake path; do not claim BM25 fusion where none happens. Track full implementation as a ROADMAP item if deferred.
  - Replace selection sort with `sort.Slice` (reuse `sortSearchResults` from shard.go).
- **Tests:** concurrent shard writes during `ShardIndex` search under `-race`; hybrid returns keyword-only hits.
- **Size:** M-L · **Risk:** medium (API shape)

### Phase 3 validation
```bash
go test ./distributed/... -race -count=1
```
**Commits:** `fix: distributed replication bookkeeping and shard ID collisions`, `fix: apply scatter-gather timeout and dedupe fan-out`, `refactor: shard index locking and hybrid search`.

---

## Phase 4 — Reasoning & search semantics (P2)

**Status: complete (2026-08-21)** — 4.1 `8ce60fe` · 4.2 `42b9a85` · 4.3 `a5e5184` · 4.4 `f4e73f3`. Validation: `go test ./reasoning/... ./fuse/... ./chunker/... -count=1` green plus downstream consumers (`pipeline`, `ingest`, `app`, `api`) green; `go vet ./...` and `gofmt -l .` clean. Behavioral changes (reasoning now filters at `MinConfidence`, chunk boundaries shift) are noted in CHANGELOG `[Unreleased]` and `docs/MIGRATION.md` for the next release.

### 4.1 Reasoning engine: dead `MinConfidence`, arbitrary threshold
- **File:** `reasoning/engine.go:13-56, 91-115`
- **Defect:** `Config.MinConfidence` is validated but never stored on `Engine`; `InferRelations` filters with `ir.Weight >= e.graph.Relations()[0].Weight*0.1` — arbitrary, order-dependent, and calls `e.graph.Relations()` (full locked copy) inside the loop → O(R²) allocations.
- **Fix:** store `minConfidence` in `Engine`; filter `ir.Weight >= e.minConfidence`; hoist `relations := e.graph.Relations()` out of the loop.
- **Tests:** inferred set respects configured threshold; loop no longer re-copies relations (optional allocs bench).
- **Size:** S · **Risk:** low (behavior change is the point — note in CHANGELOG)

### 4.2 `ExplorePaths` doc vs behavior
- **File:** `reasoning/engine.go:117-193`
- **Defect:** doc says "finds all paths" but the global `visited` set yields a BFS tree (at most one path per entity).
- **Fix:** either implement true all-paths DFS with per-path visited sets (bounded by maxDepth) or correct the doc comment to "finds shortest-hop paths". Recommend: keep BFS, fix docs; file all-paths under ROADMAP if ever needed.
- **Size:** S · **Risk:** low

### 4.3 `WeightedFusion` variadic semantics
- **File:** `fuse/fuse.go:14-64`
- **Defect:** documented for 2 score maps, but variadic — maps 3+ each get weight `1-α` (weights don't sum to 1).
- **Fix:** normalize across N maps (`α` for first, `(1-α)/(N-1)` for the rest) and document it, or document that only the first two maps are fused. Prefer explicit normalization.
- **Tests:** 3-map fusion weights sum to 1.
- **Size:** S · **Risk:** low

### 4.4 `FixedChunker` oversized parts & separator default
- **File:** `chunker/fixed.go:55-108, 121-148`
- **Defects:** (a) an oversized part is sub-split only when nothing was accumulated yet; otherwise it flushes whole in the next chunk (exceeds `maxChars`); the inner flush block at :60-69 is dead code. (b) `buildChunk`/`getOverlap` use raw `f.config.Separator` while `Chunk()` defaults a local `sep` to `"\n\n"` — with a zero `Config{}`, joining uses `""` and `TrimRight` is a no-op.
- **Fix:** normalize `Separator` in `NewFixed` (like MaxTokens); run oversized parts through `splitBySize` regardless of accumulation state; delete dead flush block.
- **Tests:** every produced chunk ≤ maxChars (property-style test over synthetic long paragraphs); zero-config chunker still joins with `"\n\n"`.
- **Size:** S · **Risk:** low (chunk boundaries shift slightly — regenerate any golden fixtures)

### Phase 4 validation
```bash
go test ./reasoning/... ./fuse/... ./chunker/... -count=1
```
**Commits:** `fix: reasoning engine honors MinConfidence`, `fix: chunker size guarantees and separator defaults`, `docs: ExplorePaths semantics`, `fix: WeightedFusion weight normalization`.


---

## Phase 5 — Code smells & hardening (P3)

One commit (`refactor:` / `chore:`) per group; each item is independently reviewable.

### 5.A Mechanical quick fixes (S, no behavior change)
| # | File | Fix |
|---|---|---|
| A1 | `distributed/shard_index.go:226` | Replace O(n²) selection sort with `sort.Slice` (already covered if 3.4 lands) |
| A2 | `chunker/fixed.go:198` | Delete hand-rolled `itoa`; use `strconv.Itoa` |
| A3 | `store/sqlite.go:541` | `errors.Is(err, sql.ErrNoRows)` instead of `==` |
| A4 | `store/sqlite.go:790` | Drop unused `score` param from `matchesFilters` |
| A5 | `graph/graph.go:191-240` | `RemoveEntity`: single pass filtering instead of O(E×R) re-filter |
| A6 | `cache/lru.go` | Track `Hits`/`Misses` in `Get`/`Set` (fields already exist) or remove them from `CacheStats` |
| A7 | `ingest/pipeline.go:182` | Use the original `d.ID` for upload-failure attribution |
| A8 | `chunker/recursive.go` | Reuse rune counts inside the split loop (avoid O(n²) rescans) |

### 5.B API & security hardening (S, low risk)
- **B1** `api/handlers.go:146` — detect oversized bodies via `*http.MaxBytesError` assertion (Go 1.19+) instead of `strings.Contains` on the error text.
- **B2** `api/server.go:171-178` — `NewAPIKey` uses `v % 62` (modulo bias, first 8 alphabet chars ~25% likelier). Fix with rejection sampling (`v < 248`) or use `base64.RawURLEncoding`.
- **B3** `api/auth.go` — optional: constant-time API-key comparison (`crypto/subtle`) for defense-in-depth; keep map lookup as pre-filter if needed.
- **B4** `api/handlers.go` — `/diagnostics` is unauthenticated by default (`defaultExempt`): keep for ops use but add a one-line security note to the README/server docs recommending auth in front of public deployments.
- **B5** `store/sqlite.go:407-444` (`searchFTS5`) — remove the pointless single-quote "escaping" on the bound parameter; sanitize the FTS5 MATCH expression properly (wrap terms in double quotes after stripping them) so user queries with `(`, `"`, `*` degrade gracefully instead of silently dropping keyword search; keep the vector-only fallback.

### 5.C Test & tooling follow-ups
- **C1** Add the regression tests listed in Phases 1–4 (they double as the acceptance criteria).
- **C2** Wire `golangci-lint run` into CI parity with `.golangci.yml` if not already covered by `.github/workflows/go.yml`.
- **C3** Consider `-race` runs for `api`, `app`, `connector` packages in CI as well (cheap, currently untested under race).

### Phase 5 validation
```bash
go test ./... -count=1 -cover
gofmt -l . && go vet ./...
```
**Commits:** `refactor: mechanical cleanups (sorting, strconv, errors.Is)`, `fix: API hardening (MaxBytesError, key bias, FTS5 sanitization)`, `test: regression coverage for review findings`.

---

## Execution order & dependencies

```
Phase 1 (races) ──▶ Phase 2 (correctness) ──▶ Phase 3 (distributed)
                              │                        │
                              └────▶ Phase 4 (semantics) ──▶ Phase 5 (smells)
```
- Phases 1 and 2 are independent of each other and can be done in parallel branches; both are small and high-value.
- Phase 3.4 (shard hybrid honesty) may be split: land the locking + sort fixes now, track true hybrid search as a ROADMAP item.
- Phase 5.A items that overlap Phase 3/4 (A1) should land with their parent phase to avoid churn.

## Definition of done (per phase)

Per [.clinerules](./.clinerules) phase-completion checklist:
- [ ] Fix implemented with the listed regression tests
- [ ] `go vet ./...` clean; `gofmt -l .` empty
- [ ] `go test ./... -count=1` green; affected packages green under `-race`
- [ ] Coverage for touched packages stays ≥ 80%
- [ ] CHANGELOG.md entry for behavior changes (2.2 delete semantics, 2.3 timestamps, 4.1 thresholds)
- [ ] README.md status / PLANNING.md updated where scope changed
- [ ] Conventional commit per item; cross-reference this plan in the commit body

## Findings index (from the 2026-08-19 review)

| Finding | Severity | Tracked in |
|---|---|---|
| `MemoryStore.DeleteDocument` map race (reproduced) | P0 | 1.1 |
| `Incremental.Save` marshal race + dead code + dirty flag (reproduced) | P0 | 1.2 |
| BM25 count drift on re-add / unknown remove (reproduced) | P1 | 2.1 |
| `DeleteChunk` swallows not-found (reproduced) | P1 | 2.2 |
| SQLite literal-`Z` timestamps | P1 | 2.3 |
| Orphaned embeddings (FK pragma off) | P1 | 2.4 |
| `MinScore` ignored in SQLite/hybrid paths | P1 | 2.5 |
| Replica ID mismatch; `GetReplicationStatus` always 0 | P1 | 3.1 |
| Shard ID collisions after delete | P1 | 3.2 |
| `ScatterGatherConfig.Timeout` never applied | P2 | 3.3 |
| Fake hybrid search; unlocked `ShardIndex`; O(n²) sort | P2 | 3.4 / 5.A1 |
| `reasoning.MinConfidence` dead; arbitrary threshold | P2 | 4.1 |
| `ExplorePaths` doc overclaim | P2 | 4.2 |
| `WeightedFusion` weights for 3+ maps | P2 | 4.3 |
| `FixedChunker` oversized chunks; separator default | P2 | 4.4 |
| 13 assorted smells (see review) | P3 | 5.A/5.B/5.C |

**Explicitly out of scope** (noted, not defects): fixed-seed HNSW RNG (determinism is documented), MD5 in the hash ring (non-security use), mock embedder quality, `MaxOpenConns(1)` throughput ceiling (documented embedded-SQLite tradeoff).


