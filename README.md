# Recall — Go Knowledge Store for RAG

A Go library for building Retrieval-Augmented Generation (RAG) applications. Recall provides structured storage, embedding-based similarity search, metadata filtering, document chunking, persistent SQLite storage, HNSW ANN indexing, and RAG pipeline context assembly — all with zero CGO required.

## Features

- **Document Chunking** — Pluggable chunking strategies (fixed-size, recursive paragraph/sentence splitting)
- **Embedding Abstraction** — Dependency-injected embedders (bring your own: OpenAI, local models, or mock)
- **Vector Similarity Search** — Cosine similarity with brute-force or HNSW ANN indexing
- **Metadata Filtering** — Term, range, date range, and custom filters
- **Hybrid Search** — BM25 keyword + vector score fusion with WeightedFusion or RRF
- **SQLite Persistence** — Persistent storage with `modernc.org/sqlite` (pure Go, no CGO)
- **HNSW ANN Index** — Approximate nearest neighbor search for 100K+ chunks
- **RAG Pipeline** — Context assembly, prompt templating, token management
- **Knowledge Graph** — Entity/relation extraction, graph traversal (BFS/DFS), transitive closure, path finding, common-neighbor inference
- **Multi-namespace** — Isolated knowledge spaces within a single store
- **Zero CGO** — Pure Go standard library only for core; SQLite via pure Go driver

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/deagy/recall/chunker"
    "github.com/deagy/recall/core"
    "github.com/deagy/recall/embedder"
    "github.com/deagy/recall/index"
    "github.com/deagy/recall/store"
)

func main() {
    ctx := context.Background()

    // Create a store with mock embedder (384 dimensions)
    s, err := store.NewMemoryStore(store.Config{
        Namespace:      "my-knowledge",
        Embedder:       embedder.NewMockEmbedder(384),
        ChunkerFactory: chunker.NewFixed,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer s.Close()

    // Upload a document
    doc := core.NewDocument("doc-1", "Go Programming", "https://go.dev")
    err = s.Upload(ctx, doc, `
        Go is a statically typed, compiled programming language designed at Google.
        It is syntactically similar to C but with memory safety, garbage collection,
        and structural typing. Go compiles quickly to machine code yet has the
        convenience of a dynamically typed language.
    `)
    if err != nil {
        log.Fatal(err)
    }

    // Search for relevant chunks
    results, err := s.Search(ctx, "Go garbage collection memory", index.DefaultSearchOptions(5))
    if err != nil {
        log.Fatal(err)
    }

    for i, r := range results {
        fmt.Printf("[%d] score=%.4f: %s...\n", i+1, r.Score, r.Chunk.Content[:80])
    }

    // Hybrid search: combine vector similarity with BM25 keyword ranking
    hybridOpts := index.DefaultSearchOptions(5)
    hybridOpts.BM25Weight = 0.5 // 50% vector, 50% BM25
    hybridResults, err := s.SearchHybrid(ctx, "Go garbage collection", hybridOpts)
    if err != nil {
        log.Fatal(err)
    }
    for i, r := range hybridResults {
        fmt.Printf("[H%d] fused=%.4f: %s...\n", i+1, r.Score, r.Chunk.Content[:80])
    }
}
```

## Architecture

```
recall/
├── core/           # Data types: Chunk, Document, Value, errors
├── chunker/        # Text chunking: Fixed, Recursive strategies
├── embedder/       # Embedding interface + Mock implementation
├── index/          # Storage index: Memory (brute-force + HNSW), filters
├── store/          # High-level store: Memory + SQLite backends
├── pipeline/       # RAG pipeline: context assembly, templates, queries
├── graph/          # Knowledge graph: entities, relations, traversal, inference
├── bm25/           # BM25 keyword ranking function
├── fuse/           # Score fusion: WeightedFusion, RRFFusion
├── query/          # Query engine (planned)
└── example/        # Usage examples
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding dims | Variable (`[]float32`) | Supports any model (Ada-002: 1536, bge-small: 384, etc.) |
| Similarity | Cosine | Standard for text embeddings |
| Chunking | Pluggable interface | Different docs need different strategies |
| Embedder | Dependency injection | Users bring their own model |
| ANN | Brute-force (Phase 1), HNSW (Phase 6) | Simple first, scale later |
| Dependencies | Zero for core | Easy adoption, no supply chain risk |

## Current Status

- [x] Phase 1: Core data model + chunking
- [x] Phase 2: Embedding + in-memory index
- [x] Phase 3: Query engine with filters
- [x] Phase 4: Hybrid search (BM25 + vector fusion)
- [x] Phase 5: SQLite persistence (modernc.org/sqlite, pure Go)
- [x] Phase 6: HNSW ANN index (auto-enabled for 1K+ chunks)
- [x] Phase 7: RAG pipeline (context assembly, prompt templates, token management)
- [x] Phase 8: Knowledge graph (entity/relation extraction, BFS/DFS traversal, transitive closure, path finding, common-neighbor inference)
- [x] Phase 9: Graph-based RAG (GraphStore interface, entity/relation extraction from text, graph-augmented retrieval)
- [ ] Phase 10: Multi-hop reasoning engine (automated path generation, relationship inference chains)

## Testing

```bash
go test ./... -v
```

## License

MIT
