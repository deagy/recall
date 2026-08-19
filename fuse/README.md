# fuse

Score fusion for combining ranked result lists — the combining step of
hybrid (vector + BM25) search.

## Fusions

| Fusion | Constructor | Behavior |
|--------|-------------|----------|
| `WeightedFusion` | `NewWeightedFusion(alpha)` | `alpha * vectorScore + (1-alpha) * keywordScore` on normalized scores. |
| `RRFFusion` | `NewRRFFusion(k)` | Reciprocal Rank Fusion: rank-based, scale-free, robust when the two score distributions differ. |

Both implement the `Fusion` interface, which `index.SearchOptions.Fusion`
accepts so callers can swap strategies per query.

## Used by

`index` (hybrid search) and `store.SearchHybrid` (default: weighted at
0.5/0.5, overridable with `index.SearchOptions.Fusion`).
