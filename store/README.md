# store

The high-level knowledge store: chunking, embedding, indexing, and
persistence behind one interface.

```go
type Store interface {
    Upload(ctx context.Context, doc *core.Document, content string) error
    Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error)
    SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error)
    GetChunk(id string) (*core.Chunk, bool)
    DeleteChunk(ctx context.Context, id string) error
    DeleteDocument(ctx context.Context, docID string) error
    Count() int
    Namespaces() []string
    Close() error
}
```

## Implementations

| Store | Constructor | Persistence |
|-------|-------------|-------------|
| `MemoryStore` | `NewMemoryStore(cfg)` | in-memory (per-namespace indexes) |
| `SQLiteStore` | `NewSQLiteStore(cfg, dbPath)` | SQLite via `modernc.org/sqlite` (pure Go), WAL mode, HNSW for 1K+ chunks |
| `MultiModalStore` | `NewMultiModalStore(embedder)` | text + image items |

`Config` selects namespace, embedder, `ChunkerFactory`, and (SQLite)
schema `Migrations` applied via `Migrator`.

## Graph stores

- `MemoryGraphStore` / `SQLiteGraphStore` implement `GraphStore`
  (entity/relation extraction, path finding, transitive closure,
  common neighbors) — the graph layer over the store.
- `GraphPersistence` + `LoadFromDB` give SQLite graphs full round-trips.

## Resilience & operations

- `ResilientStore` — backup/restore/checkpoint operations;
  `RestoreSQLite(src, dest)` for atomic restore.
- `HealthCheck` / `HealthHandler`, `DiagnosticsReport`, integrity checks
  (WAL status, foreign keys).
- `Resilience` helpers (WAL auto-checkpointing, backup rotation) in
  `resilience.go`.
