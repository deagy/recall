# ingest

The ingestion pipeline: **load → filter → transform → upload**, with
deduplication, validation, incremental re-ingestion, progress reporting, and
batched runs.

## Core

```go
p, err := ingest.NewPipeline(ingest.Options{
    Store:    store,                        // required
    Loader:   dirLoader,                    // exactly one of Loader/Connector
    Source:   "/path/or/url",
    Dedup:    ingest.NewDeduplicator(),     // optional, content-hash based
    Validator: ...,                         // optional, Schema-based rejection
    Incremental: ...,                       // optional, only changed docs
    Progress:  ingest.NewProgress(),        // optional counters + callbacks
    Concurrency: 4,                         // 0/1 = sequential
})
result, err := p.Run(ctx)   // *Result{Loaded, Uploaded, Skipped, Failed, Duration}
```

## Parts

- `Deduplicator` — content-hash dedup; `LoadDeduplicator(path)` persists
  seen hashes across runs.
- `Incremental` — per-source content fingerprints so re-runs only ingest
  changed documents; state persisted via `NewIncremental(path)`.
- `Validator` / `Schema` — reject documents violating a schema (e.g.
  minimum length, required fields).
- `Progress` — thread-safe counters with `OnDocument` / `OnPhase`
  callbacks and `Summary()`.
- `RunBatch(ctx, opts, refs)` — runs the pipeline over many sources.
- `Transform` — `func(*loader.Document) (*loader.Document, error)` applied
  per document before upload.

`ContentHash` is the SHA-256 hash used by dedup/incremental, exported for
build-your-own bookkeeping.
