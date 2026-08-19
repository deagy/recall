# index

Storage indexes: in-memory vector + keyword indexes, ANN search, metadata
filtering, and quantized variants. This is the retrieval engine under
`store`.

## Indexes

| Index | Constructor | Notes |
|-------|-------------|-------|
| `MemoryIndex` | `NewMemoryIndex(ns, dim)` | Brute-force cosine similarity + internal BM25; the workhorse. |
| `HNSW` | `NewHNSW(dim, cfg)` | Hierarchical Navigable Small World ANN graph; auto-enabled by stores at 1K+ chunks (`HNSWThreshold`). |
| `HybridIndex` | `NewHybridIndex(ns, dim, fusion)` | Vector + BM25 with a pluggable `fuse.Fusion`. |
| `MetadataIndex` | `NewMetadataIndex()` | Pure metadata filtering. |
| `MultiVectorIndex` | `NewMultiVectorIndex(ns, dim)` | Multiple vectors per item with aggregation modes. |
| `QuantizedIndex` / `PQIndex` | `NewQuantizedIndex` / `NewPQIndex` | Scalar / product quantization for memory reduction. |

## Search options & filters

`SearchOptions` (`TopK`, `MinScore`, `Filters`, `Hybrid`, `BM25Weight`,
`Fusion`, `EfSearch`) parameterizes every search. Filters combine
conjunctively:

- `TermFilter` — metadata term equality
- `RangeFilter` — numeric range
- `DateRangeFilter` — time range

`SortedIDs` renders a set of IDs deterministically (used in tests/goldens).
