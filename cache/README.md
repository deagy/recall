# cache

In-memory LRU caches with TTL for the expensive results in a RAG pipeline:
embeddings, query results, and graph traversals.

## API

- `Cache` — the core interface (`Get`, `Set`, `Delete`, `Len`, `Stats`).
- `NewLRUCache(config)` / `DefaultCacheConfig()` — LRU implementation with
  optional TTL and a size cap.
- `NewEmbeddingCache(config)` + `GenerateEmbeddingKey(text)` — cache keyed
  on the hashed input text; wraps with
  `embedder.NewCachingEmbedder(inner, cache, ttl)` to memoize an embedder.
- `NewQueryCache(config)` + `GenerateQueryKey(query, filters)` — keyed on the
  query plus a canonicalized filter representation.
- `NewGraphCache(config)` + `GenerateGraphTraversalKey(...)` — keyed on
  entity + traversal type + depth.
- `CacheManager` — manages multiple named caches; `CacheStats` reports
  hits/misses/evictions.

## Used by

`embedder` (embedding memoization) and application code in front of
`pipeline` / `graph`.
