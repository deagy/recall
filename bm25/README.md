# bm25

A pure-Go Okapi BM25 keyword ranking function used for the keyword leg of
hybrid search. No external dependencies.

## API

```go
bm := bm25.New(bm25.DefaultConfig())
bm.AddDocument(chunkID, content)
results := bm.Search("query text", topK) // []SearchResult{ID, Score}
bm.RemoveDocument(chunkID)
```

`Config` exposes the standard BM25 parameters (`K1`, `B`, tokenization
case-folding).

## Used by

`index.MemoryIndex` / `index.HybridIndex` maintain a BM25 index alongside
the vector index; `store.SearchHybrid` fuses the two score lists (see
`fuse`).
