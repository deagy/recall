# Recall — Go Knowledge Store for RAG

A Go library for building Retrieval-Augmented Generation (RAG) applications. Recall provides structured storage, embedding-based similarity search, metadata filtering, document chunking, persistent SQLite storage, HNSW ANN indexing, and RAG pipeline context assembly — all with zero CGO required.

## Features

- **Document Chunking** — Pluggable chunking strategies (fixed-size, recursive paragraph/sentence splitting)
- **Embedding Abstraction** — Dependency-injected embedders (bring your own: OpenAI, Cohere, Ollama, local ONNX models, or mock)
- **Vector Similarity Search** — Cosine similarity with brute-force or HNSW ANN indexing
- **Metadata Filtering** — Term, range, date range, and custom filters
- **Hybrid Search** — BM25 keyword + vector score fusion with WeightedFusion or RRF
- **SQLite Persistence** — Persistent storage with `modernc.org/sqlite` (pure Go, no CGO)
- **HNSW ANN Index** — Approximate nearest neighbor search for large collections (automatic brute-force below the activation threshold)
- **RAG Pipeline** — Context assembly, prompt templating, token management
- **Knowledge Graph** — Entity/relation extraction, graph traversal (BFS/DFS), transitive closure, path finding, common-neighbor inference
- **Multi-hop Reasoning** — Pluggable inference rules (transitive, symmetric, anti-symmetric), depth-limited path exploration, confidence propagation, natural language query → graph reasoning
- **Namespaces** — Isolated knowledge spaces (one store instance per namespace; a document can override the store's namespace via `core.Document.Namespace`; search spans all namespaces in a store)
- **Distributed Storage** — Consistent hashing, automatic sharding, scatter-gather search, replication strategies (primary-replica, quorum, all-nodes)
- **Semantic Chunking** — Similarity-based text splitting, streaming processing, chunk quality metrics, adaptive sizing
- **Graph Embeddings** — TransE-based entity/relation embeddings, link prediction, entity similarity search, knowledge graph completion, model persistence via `Save`/`Load`
- **Intelligent Caching** — LRU eviction, TTL-based expiration, query result caching, embedding caching, graph traversal caching, multi-level caching (L1/L2), cache warming
- **Local ONNX Embeddings** — Pure-Go ONNX inference runtime (`embedder/onnx`) runs sentence-transformer ONNX exports with no CGO and no network; `embedder.OnnxEmbedder` accepts a tokenizer function and drops into the `embedder.Pipeline` failover chain. Ships fully-offline bundled WordPiece tokenizers for `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, and `nomic-embed-text-v1.5` (embedded BERT-uncased vocab via `embedder.BundledTokenizer`), parallel `EmbedBatch` execution (`BatchConcurrency`), and on-disk model download/caching (`embedder.ModelCache` + `LoadHFModel`)
- **Document Loaders** — `loader` package reads text, markdown (heading sections), CSV (column mapping), JSON (nested field paths), HTML (visible-text extraction), PDF (plain-text extraction), DOCX (OOXML paragraph extraction), and whole directories into a uniform `Document` representation ready for upload
- **Source Connectors** — `connector` package fetches documents from the web (rate-limited), git repositories, S3-compatible buckets (self-contained SigV4, MinIO-friendly), GitHub repos/issues, and SQL tables
- **Ingestion Pipeline** — `ingest` package orchestrates load → dedup → validation → transform → upload with thread-safe progress callbacks, parallel batch ingestion across sources, and incremental (delta) re-ingestion via persisted content hashes
- **Advanced Indexing** — 8-bit scalar quantization (`index.ScalarQuantizer`/`QuantizedIndex`, 4x memory reduction), product quantization (`index.ProductQuantizer`/`PQIndex`, k-means++ codebooks with ADC search), combined vector+BM25 `index.HybridIndex` with pluggable fusion, inverted `index.MetadataIndex` pre-filter, and `index.MultiVectorIndex` for multiple embeddings per chunk (MaxSim / mean / top-mean)
- **Query Enhancement** — LLM-powered `query.Rewriter`, `query.HyDE` (hypothetical document embeddings), `query.StepBack` (abstraction prompting), `query.SubQueryDecomposer` (LLM-first, heuristic fallback), and `query.Multilingual` (script-based language detection + pluggable translation for multi-query retrieval)
- **Advanced Chunking** — `chunker.ParentChildChunker` (child retrieval with parent context expansion), `chunker.DocumentAwareChunker` (strict document-boundary respect), `chunker.AdaptiveChunker` (content-driven chunk sizing)
- **Multi-Modal** — `embedder.MultiModalEmbedder` (shared text+image vector space), `store.MultiModalStore` (cross-modal search: text queries find images and vice versa), `pipeline.MultiModalPipeline` (mixed text/image RAG context with optional LLM answer)
- **Reranking** — `reranker` package improves top-k precision with a two-stage coarse→fine stage: `CrossEncoderReranker` (pure-Go ONNX cross-encoder), `SparseReranker` (BM25 re-scoring), `LLMReranker` (LLM-as-judge over an injected `llm.Backend`), `EnsembleReranker` (fuses several rerankers via `fuse`), `LTRanker` (pointwise learning-to-rank with `Fit`), `AdaptiveLTRanker` (feedback-driven adaptation with auto-refit at a configurable threshold), and `Experiment` (A/B testing: NDCG@K, MRR@K, Precision@K, win rate, Welch t-test). Wire any of them into a pipeline with `RAGPipeline.WithReranker(...)` plus `WithCoarseTopK`/`WithRerankTopK`; rerank scores and ranks are attributed on each `index.SearchResult`
- **REST API Service** — `api` package exposes Recall over HTTP using only the standard library: `POST /upload`, `GET /search`, `POST /hybrid-search`, `POST /rag`, `GET /graph/{entity}`, `POST /graph/reason`, plus `/healthz`/`/readyz`/`/diagnostics` and an embedded OpenAPI 3.0 spec at `GET /openapi.json`; optional authentication via static API keys (`X-API-Key`/Bearer) or HS256 JWTs (stdlib-verified), CORS, and body-size limits; runnable standalone via `cmd/recall-server` with graceful shutdown
- **Configuration & Deployment** — `config` package loads JSON/YAML configs with env-var overrides (`RECALL__SECTION__KEY`), validation, and mtime-poll hot reload (`config.Watcher`); `deploy/` ships a multi-stage pure-Go (CGO_ENABLED=0) distroless Dockerfile, docker-compose, and Kubernetes manifests (Deployment + Service + HPA with health probes)
- **Command-Line Interface (CLI)** — `cmd/recall` is a cobra-based toolkit that runs in two modes: **local** (in-process against the configured SQLite/memory store) and **server** (HTTP client of a running recall-server via `--server`/`cli.url`, backed by the typed `client` package). Commands: `upload` (files + recursive directories), `search`, `hybrid-search`, `rag`, `graph` (+ `graph list`), `reason` (NL query or `--from`/`--to` paths), `store info|migrate|backup|restore` (online VACUUM INTO backup, atomic restore, versioned SQL migrations), `cluster status` (node /diagnostics probes with exit-code health gating), and `eval` (+ `eval compare` regression gate, exit 2 on regressions). Output as table, JSON, or YAML (`-o`); configuration via `--config`, `$HOME/.recall.yaml` (.yml/.json), and `RECALL__SECTION__KEY` env overrides
- **Zero CGO** — Pure Go standard library only for core; SQLite via pure Go driver. The zero-CGO constraint applies to library code and its dependencies — a test build with `CGO_ENABLED=1` is explicitly allowed where a C compiler is present, since Go's race detector (`go test -race`) links the toolchain's own C runtime (tsan) and does not make the library itself cgo-dependent

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

## Command-Line Interface (CLI)

`cmd/recall` is the Recall command-line toolkit. Build it with:

```bash
go build -o recall ./cmd/recall
```

Commands run in two modes:

- **local** (default) — in-process against the store from your configuration
  (`store.backend: sqlite` + `store.path` for persistence, or an in-memory store)
- **server** — when a server URL is given (`--server` flag or `cli.url` in the
  config), the data commands act as an HTTP client for a running recall-server

```bash
# Ingest files or directories (text, markdown, CSV, JSON, HTML, PDF, DOCX)
recall upload ./docs

# Vector / hybrid search
recall search "how does the index work" --top-k 5
recall hybrid-search "indexing" --bm25-weight 0.7

# RAG: retrieve, assemble context, render the prompt (with citations)
recall rag "Why is chunking important?" --top-k 5

# Knowledge graph + multi-hop reasoning
recall graph "Alice"
recall reason --from alice --to berlin --max-hops 4

# Store maintenance (SQLite backend)
recall store info
recall store migrate migrations.sql
recall store backup recall.db.backup
recall store restore recall.db.backup --force

# Distributed cluster health (exit 1 when a node is down)
recall cluster status --node http://node1:9000 --node http://node2:9000

# Retrieval evaluation (Precision/Recall/MRR/NDCG@K) + CI regression gate
recall eval dataset.json --save report.json
recall eval compare baseline.json report.json   # exit 2 on regression

# Talk to a running recall-server instead of the local store
recall --server http://localhost:8080 search "indexing" -o json
```

Every command renders `table` (default), `json`, or `yaml` via `-o/--output`.
Configuration resolves in order: `--config` flag, `$HOME/.recall.yaml`
(also `.yml`/`.json`), built-in defaults; `RECALL__SECTION__KEY` environment
variables (e.g. `RECALL__CLI__API_KEY`) override file values. See `deploy/config/`
for example config files and the `cli` section (url, api_key, timeout, output,
cluster_nodes).

## Semantic Chunking

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/deagy/recall/chunker"
    "github.com/deagy/recall/core"
    "github.com/deagy/recall/embedder"
)

func main() {
    ctx := context.Background()

    // Create a semantic chunker with mock embedder
    embedder := embedder.NewMockEmbedder(384)
    cfg := chunker.DefaultSemanticConfig()
    cfg.Threshold = 0.7
    cfg.MinChunkSize = 100
    cfg.MaxChunkSize = 2000

    semanticChunker := chunker.NewSemantic(embedder, cfg)

    doc := core.NewDocument("doc-1", "Semantic Chunking Example", "")
    content := `
        Go is a statically typed, compiled programming language designed at Google.
        It is syntactically similar to C but with memory safety, garbage collection,
        and structural typing. Go compiles quickly to machine code yet has the
        convenience of a dynamically typed language.

        Python is a high-level, general-purpose programming language. Its design
        philosophy emphasizes code readability with the use of significant indentation.
        Python is dynamically typed and garbage-collected.

        Rust is a multi-paradigm, systems programming language focused on safety,
        especially safe concurrency. Rust is syntactically similar to C++, but is
        designed to provide better memory safety while maintaining high performance.
    `

    chunks, err := semanticChunker.Chunk(doc, content)
    if err != nil {
        log.Fatal(err)
    }

    for i, chunk := range chunks {
        fmt.Printf("[%d] size=%d: %s...\\n", i+1, len(chunk.Content), chunk.Content[:80])
    }
}
```

### How It Works

**Semantic Similarity**: The chunker embeds each sentence and computes cosine similarity between adjacent sentences. When similarity drops below a threshold, a split point is created.

**Configurable Threshold**: Lower thresholds create larger chunks (more tolerant of topic changes), while higher thresholds create smaller, more focused chunks.

**Adaptive Sizing**: Minimum and maximum chunk sizes ensure chunks are neither too small to be useful nor too large for effective retrieval.

**Streaming Processing**: For large documents, the StreamingChunker processes content incrementally, emitting chunks as they become available.

## Distributed Storage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/deagy/recall/chunker"
    "github.com/deagy/recall/core"
    "github.com/deagy/recall/distributed"
    "github.com/deagy/recall/embedder"
    "github.com/deagy/recall/index"
)

func main() {
    ctx := context.Background()

    // Create a distributed store with mock embedder
    ds := distributed.NewDistributedStore(
        distributed.DefaultClusterConfig(),
        embedder.NewMockEmbedder(384),
        chunker.NewFixed,
        "distributed-knowledge",
    )

    // Add nodes to the cluster
    ds.AddNode(&distributed.Node{
        ID:      "node-1",
        Address: "localhost:8080",
    })
    ds.AddNode(&distributed.Node{
        ID:      "node-2",
        Address: "localhost:8081",
    })
    ds.AddNode(&distributed.Node{
        ID:      "node-3",
        Address: "localhost:8082",
    })

    // Upload a document (automatically sharded across nodes)
    doc := core.NewDocument("doc-1", "Distributed Systems", "https://en.wikipedia.org/wiki/Distributed_computing")
    err := ds.Upload(ctx, doc, `
        Distributed computing is a field of computer science that studies distributed systems.
        A distributed system is a system whose components are located on different networked computers.
        All the computers work together to achieve some common goal.
        Distributed systems hide the fact that hardware and software components are distributed from the user.
    `)
    if err != nil {
        log.Fatal(err)
    }

    // Search across all shards using scatter-gather
    results, err := ds.Search(ctx, "distributed systems computing", index.DefaultSearchOptions(5))
    if err != nil {
        log.Fatal(err)
    }

    for i, r := range results {
        fmt.Printf("[%d] score=%.4f: %s...\\n", i+1, r.Score, r.Chunk.Content[:80])
    }
}
```

### How It Works

**Consistent Hashing**: Data is distributed across nodes using a consistent hash ring with virtual nodes. This ensures minimal data movement when nodes are added or removed.

**Sharding**: Documents are automatically split into chunks and distributed across shards. Each shard is assigned to a node based on consistent hashing.

**Scatter-Gather Search**: When a query is issued, it is fanned out to all active shards in parallel. Results are collected, merged, sorted by relevance, and the top-K results are returned.

**Replication**: Data can be replicated across multiple nodes using configurable strategies:
- **Primary-Replica**: Data is written to a primary node and replicated to N-1 replica nodes
- **Quorum**: Data is written to a quorum of nodes (majority)
- **All-Nodes**: Data is replicated to all online nodes

## Graph Embeddings

Learn dense vector representations for knowledge graph entities and relations using the TransE algorithm. Enable link prediction, entity similarity search, and knowledge graph completion.

```go
package main

import (
    "fmt"
    "log"

    "github.com/deagy/recall/graph"
)

func main() {
    // Create a knowledge graph
    kg := graph.NewKnowledgeGraph()

    // Add entities
    kg.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
    kg.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
    kg.AddEntity(graph.NewEntity("charlie", "Charlie", graph.EntityPerson))
    kg.AddEntity(graph.NewEntity("acme", "Acme Corp", graph.EntityOrganizer))

    // Add relations
    kg.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
    kg.AddRelation(graph.NewRelation("bob", "charlie", "knows", 0.8))
    kg.AddRelation(graph.NewRelation("alice", "acme", "works_at", 0.95))
    kg.AddRelation(graph.NewRelation("bob", "acme", "works_at", 0.85))

    fmt.Println(kg)

    // Create training triples from the graph
    triples := make([]*graph.Triple, 0)
    for _, r := range kg.Relations() {
        triples = append(triples, &graph.Triple{
            Head:     r.From,
            Relation: r.Type,
            Tail:     r.To,
        })
    }

    // Train TransE embeddings
    store := graph.NewEmbeddingStore(64)
    model := graph.NewTransE(store)

    opts := graph.DefaultTrainOptions()
    opts.Dimension = 64
    opts.Epochs = 100
    opts.BatchSize = 32
    opts.NegativeSamples = 5

    err := model.Train(triples, opts)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Trained embeddings: %d entities, %d relations\\n",
        store.EntityCount(), store.RelationCount())

    // Get entity embeddings
    aliceEmb, err := model.EmbedEntity("alice")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Alice embedding dimension: %d\\n", len(aliceEmb))

    // Get relation embeddings
    knowsEmb, err := model.EmbedRelation("knows")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Knows relation embedding dimension: %d\\n", len(knowsEmb))

    // Compute entity similarity
    sim, err := graph.EntityPairSimilarity(model, "alice", "bob")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Similarity between Alice and Bob: %.4f\\n", sim)

    // Compute relation similarity
    relSim, err := graph.RelationPairSimilarity(model, "knows", "works_at")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Similarity between 'knows' and 'works_at': %.4f\\n", relSim)

    // Link prediction
    lp := graph.NewLinkPrediction(model)
    results, err := lp.PredictTail("alice", "knows", 5)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Predicted tails for (alice, knows): %d results\\n", len(results))

    // Evaluate link prediction metrics
    metrics := graph.NewEvaluateMetrics()
    metrics.AddResult(1, []int{1, 3, 5, 10})
    metrics.AddResult(2, []int{1, 3, 5, 10})
    metrics.Finalize()
    fmt.Println(metrics.String())
}
```

### How It Works

**TransE Algorithm**: TransE represents entities and relations as vectors in a low-dimensional space. The key insight is that for a valid triple (h, r, t), the relation vector r should approximately satisfy: h + r ≈ t.

**Margin-Based Loss**: The training objective uses a margin-based ranking loss:
```
L = Σ max(0, margin - score(pos) + score(neg))
```
where `score(pos) = ||h + r - t||` and `score(neg) = ||h' + r - t'||` for corrupted triples.

**Negative Sampling**: For each positive triple, negative samples are generated by corrupting either the head or tail entity. This makes training efficient even for large graphs.

**Link Prediction**: To predict missing links, we score all possible (head, relation, tail) triples and rank them by their TransE score. Lower scores indicate more likely triples.

**Entity Similarity**: Entity embeddings capture semantic similarity. Entities that appear in similar relational contexts will have similar embeddings, enabling similarity-based retrieval.

## Intelligent Caching

Reduce latency and cost for repeated queries through intelligent caching with LRU eviction, TTL-based expiration, and multi-level caching.

```go
package main

import (
    "fmt"
    "log"
    "time"

    "github.com/deagy/recall/cache"
)

func main() {
    // Create an LRU cache with 1000 entry capacity
    lru := cache.NewLRUCache(cache.DefaultCacheConfig())

    // Store a query result
    queryResult := map[string]interface{}{
        "results": []string{"result1", "result2"},
        "count":   2,
    }
    lru.Set("query:123", queryResult, 5*time.Minute)

    // Retrieve the cached result
    val, ok := lru.Get("query:123")
    if ok {
        fmt.Printf("Cache hit: %v\\n", val)
    }

    // Query result cache
    qc := cache.NewQueryCache(1000)
    qc.Set("what is Go?", nil, &cache.QueryResult{
        Query:   "what is Go?",
        Results: []interface{}{"Go is a programming language"},
    }, 10*time.Minute)

    // Embedding cache
    ec := cache.NewEmbeddingCache(5000)
    embedding := []float32{0.1, 0.2, 0.3}
    ec.Set("text to embed", embedding, 1*time.Hour)

    // Multi-level cache (L1: fast in-memory, L2: larger storage)
    l1 := cache.NewLRUCache(cache.DefaultCacheConfig())
    l2 := cache.NewLRUCache(cache.DefaultCacheConfig())
    mlc := cache.NewMultiLevelCache(l1, l2)

    // Set in both levels
    mlc.Set("important data", "critical value", 30*time.Minute)

    // Get from L1 first, then L2 (with automatic promotion)
    val, ok = mlc.Get("important data")
    if ok {
        fmt.Printf("Multi-level cache hit: %v\\n", val)
    }

    // Cache warming: pre-populate cache with popular queries
    warmer := cache.NewCacheWarmer(lru)
    warmer.AddRequest(cache.WarmRequest{
        Query:  "popular query 1",
        Result: "cached result 1",
        TTL:    1 * time.Hour,
    })
    warmer.AddRequest(cache.WarmRequest{
        Query:  "popular query 2",
        Result: "cached result 2",
        TTL:    1 * time.Hour,
    })
    warmer.Warm()

    fmt.Printf("Warm stats: %s\\n", warmer.Stats())

    // Cache manager for multiple caches
    manager := cache.NewCacheManager(cache.DefaultCacheConfig())
    queryCache := manager.GetCache("queries")
    embeddingCache := manager.GetCache("embeddings")

    // Get aggregated stats
    stats := manager.Stats()
    fmt.Printf("Cache manager stats: %v\\n", stats)

    _ = log.Println // suppress unused import
}
```

### How It Works

**LRU Eviction**: The Least Recently Used (LRU) cache automatically evicts the least recently accessed entries when the cache reaches its capacity. This ensures that frequently accessed data stays in memory.

**TTL-Based Expiration**: Cache entries can have a time-to-live (TTL) after which they automatically expire. This prevents stale data from being served.

**Query Result Cache**: Caches the results of expensive queries. When the same query is issued again, the cached result is returned immediately, avoiding redundant computation.

**Embedding Cache**: Caches embedding vectors to avoid redundant computation. Since embeddings are often expensive to compute, caching them can significantly reduce latency.

**Graph Traversal Cache**: Caches the results of graph traversals (e.g., "find all friends of X"). When the same traversal is requested again, the cached result is returned.

**Multi-Level Caching**: Uses two levels of caching:
- **L1**: Fast, small in-memory cache (e.g., LRU with 1000 entries)
- **L2**: Slower, larger storage (e.g., LRU with 100,000 entries or disk-based)

When a value is retrieved from L2, it is automatically promoted to L1 for faster future access.

**Cache Warming**: Pre-populates the cache with popular queries before the application starts serving traffic. This eliminates the "cold start" problem and ensures fast response times from the beginning.

**Cache Manager**: Manages multiple cache instances with a unified interface. Useful for organizing caches by purpose (queries, embeddings, graph traversals, etc.).

## Architecture

```
recall/
├── core/           # Data types: Chunk, Document, Value, errors
├── chunker/        # Text chunking: Fixed, Recursive, Semantic, Streaming strategies
├── embedder/       # Embedding interface + Mock, OpenAI, Cohere, Ollama, ONNX (local) + pipeline
├── loader/         # Document loaders (text, markdown, CSV, JSON, HTML, PDF, DOCX, directory)
├── connector/      # Source connectors (web, git, S3+SigV4, GitHub, database)
├── ingest/         # Ingestion pipeline (dedup, validation, progress, batch, incremental)
├── llm/            # LLM backends (pluggable) + resilience decorators (retry, breaker, rate limit, fallback)
├── cache/          # Intelligent caching: LRU, TTL, query/embedding/graph caching, multi-level
├── index/          # Storage index: Memory (brute-force + HNSW), filters
├── store/          # High-level store: Memory + SQLite backends, GraphStore
├── pipeline/       # RAG pipeline: context assembly, templates, queries
├── graph/          # Knowledge graph: entities, relations, traversal, inference, embeddings
├── reasoning/      # Multi-hop reasoning: inference rules, path exploration, confidence propagation
├── bm25/           # BM25 keyword ranking function
├── fuse/           # Score fusion: WeightedFusion, RRFFusion
├── reranker/       # Fine ranking: cross-encoder (ONNX), sparse (BM25), LLM-judge, ensemble, pointwise LTR
├── query/          # Query engine (planned)
├── distributed/    # Distributed storage: consistent hashing, sharding, scatter-gather search, replication
├── metrics/        # Observability: counters/gauges/histograms, Prometheus export, structured logging, store instrumentation
├── tracing/        # OTel-compatible tracing: spans, W3C traceparent, span processors, store instrumentation
├── analytics/      # Query analytics: query log, popular queries, drop-off detection, sinks
├── feedback/       # Relevance feedback: Rocchio query expansion (vector + lexical), expand-and-retrieve
├── eval/           # Evaluation: Precision/Recall/MRR/NDCG@K, RAG answer quality, benchmark suite, reports
├── hitl/           # Human-in-the-loop: review queue, annotations, active-learning (uncertainty) prioritization
├── testutil/       # Test helpers: fixture store, scripted LLM/embedder mocks, golden files, benchmark harness
├── api/            # REST API service: stdlib HTTP server, auth (API keys + HS256 JWT), embedded OpenAPI spec
├── config/         # Service configuration: JSON/YAML load, env overrides, validation, hot-reload watcher
├── app/            # Service assembly: config → embedder/store/pipeline/graph/reasoner/API server (shared by server & CLI)
├── client/         # Typed HTTP client for the recall-server REST API (CLI server-mode transport)
├── cmd/            # recall-server: standalone service entrypoint; recall: CLI (upload/search/rag/graph/reason/store/cluster/eval)
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

## Security Guidance

Recall is a library: the security boundary of your application **is** the host
process. The notes below apply when you run the `recall-server` service
(Phase 27) or embed the SDK behind your own service layer.

### Authentication
- **Library mode** — there is no built-in authentication by design; your
  application's trust model (gateway, service mesh, middleware) is
  responsible for authorization.
- **Service mode** — enable the `auth` config section. Use static API keys
  (`api_keys`) for single-tenant deployments, and **namespace-scoped keys**
  (`scoped_keys`) when one store is shared across teams or projects. A
  scoped key can only upload into, search, run RAG over, and traverse the
  graph within its allowed namespaces:

  ```json
  "auth": {
    "enabled": true,
    "api_keys": ["admin-key"],
    "scoped_keys": [
      { "key": "team-a-key", "namespaces": ["team-a"] }
    ]
  }
  ```

  Scope is enforced on every data endpoint (`/upload`, `/search`,
  `/hybrid-search`, `/rag`, `/graph/*`): disallowed upload targets receive
  `403`, search/RAG retrieval is restricted to in-scope chunks, and
  out-of-scope graph entities are reported as *not found* so their
  existence is not leaked. Keys in `api_keys` remain unrestricted, so an
  admin key can sit alongside scoped keys. JWT auth (`jwt_secret`, HS256)
  is also available; JWT claims are not yet mapped to namespace scopes.

### Encryption
- **At rest** — the practical path for the pure-Go SQLite driver is
  filesystem-level encryption (LUKS, BitLocker, or an encrypted cloud
  volume; on Kubernetes, an encrypted volume for the database plus etcd
  encryption for etcd-managed state). In-database page encryption
  (e.g. SQLCipher) is **not supported**: it requires CGO, which Recall's
  zero-CGO constraint rules out.
- **In transit** — terminate TLS at the load balancer or reverse proxy
  (the standard approach; the bundled `deploy/` assets assume it). The
  server can also terminate TLS in-process via stdlib `net/http`
  (`ListenAndServeTLS`) if you prefer.

### Secrets
- Never commit config files containing keys or JWT secrets. Use
  environment overrides (`RECALL__AUTH__API_KEYS=...`,
  `RECALL__AUTH__JWT_SECRET=...`) or the K8s Secret in
  `deploy/kubernetes/recall.yaml`. Scoped keys go in the config file —
  the env-var format does not express per-key namespace lists.
- Rotate by changing the config and restarting (or letting the config
  watcher hot-reload it). There is no per-key revocation list.

### Threat-model notes
- Namespace scoping is **not** a substitute for tenant isolation at the
  process level: all namespaces share one SQLite file, so one process can
  read them all. For hard isolation, run one server (and one database)
  per tenant.
- Chunks that were never stamped with a namespace (e.g. uploaded by an
  older version) are invisible to scoped credentials — scope checks fail
  closed rather than open.

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
- [x] Phase 10: Multi-hop reasoning engine (inference rules, depth-limited path exploration, confidence propagation, natural language query reasoning)
- [x] Phase 13: Advanced query processing (intent detection, entity extraction, query expansion, adaptive retrieval)
- [x] Phase 14: LLM integration (pluggable backends, streaming, LLM-assisted extraction)
- [x] Phase 15: Distributed storage (consistent hashing, sharding, scatter-gather search, replication)
- [x] Phase 16: Streaming & semantic chunking (similarity-based splitting, incremental processing, chunk quality metrics)
- [x] Phase 17: Graph embeddings (TransE algorithm, link prediction, entity/relation similarity, knowledge graph completion)
- [x] Phase 18: Intelligent caching (LRU eviction, TTL expiration, query/embedding/graph traversal caching, multi-level L1/L2, cache warming)
- [x] Phase 11: Pluggable NER + relation pattern extraction (HeuristicNER with stopword filtering, PatternRelationExtractor)
- [x] Phase 12: Performance & robustness (context cancellation, SQLite HNSW mirroring, entity extraction heuristics)
- [x] Phase 20: Real embedding providers — OpenAI, Cohere, Ollama HTTP providers with retry/backoff; `embedder.Pipeline` failover + `CachingEmbedder`; pure-Go ONNX runtime (`embedder/onnx/`) with `embedder.OnnxEmbedder` for local sentence-transformer inference (no CGO, no network). Phase 20.2 complete: bundled fully-offline WordPiece tokenizers for `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text-v1.5` (embedded BERT-uncased vocab, zero network); CPU optimizations (float32 `MatMul` fast path + `Model.BatchRun` parallel worker pool with `BatchConcurrency`); and model download/caching (`embedder.ModelCache` + `LoadHFModel` with configurable base URL for mirrors/offline)
- [x] Phase 21: Document ingestion — `loader` (text, markdown, CSV, JSON, directory, HTML, PDF, DOCX), `connector` (web, git, S3 with self-contained SigV4, GitHub, database), and `ingest` (pipeline, dedup, validation, progress, batch, incremental)
- [x] Phase 26: Advanced retrieval — SQ8/PQ quantized indexes, hybrid + metadata + multi-vector indexes, LLM query enhancement (rewrite/HyDE/step-back/sub-query/multilingual), parent-child/document-aware/adaptive chunking, and multi-modal embedding/store/pipeline
- [x] Phase 22: Reranking — `reranker` package (cross-encoder on the pure-Go ONNX runtime, BM25 sparse re-scoring, LLM-as-judge, ensemble fusion, pointwise learning-to-rank) wired into `pipeline.RAGPipeline` as an optional two-stage coarse→fine stage via `WithReranker`/`WithCoarseTopK`/`WithRerankTopK`, with rerank score attribution on `SearchResult`
- [x] Phase 22.3: Learning-to-Rank — `reranker.AdaptiveLTRanker` for feedback-driven reranker adaptation (labeled feedback buffers and auto-refits at a configurable threshold, thread-safe) and `reranker.Experiment` A/B testing framework (per-variant NDCG@K / MRR@K / Precision@K, pairwise win rate, Welch t-test with p-value)
- [x] Phase 25.1: LLM Backend Resilience — composable decorators around `llm.Backend`: `RetryBackend` (exponential backoff + jitter, pluggable retryable predicate), `CircuitBreakerBackend` (closed/open/half-open with cooldown + single probe), `RateLimitBackend` (token-bucket, blocking, context-aware), `FallbackBackend` (ordered failover), and `Middleware` (chains any of them via `MiddlewareFunc`)
- [x] Phase 25.2: Store Resilience — production reliability for the SQLite store: `Checkpoint`/`StartAutoCheckpoint` (WAL), `Backup` (online `VACUUM INTO`) + `RestoreSQLite`, schema `Migration`/`Migrator` (versioned, transactional, auto-applied), and `IntegrityCheck`/`Repair` (corruption detection + FTS rebuild), exposed via the `ResilientStore` interface
- [x] Phase 25.3: Distributed Resilience — `NodeHealth` (periodic probing, offline/recovery), `AutoRebalancer` (ring tracks active nodes via `Cluster.RebalanceActive`), fault tolerance (`Cluster.Health()` healthy/degraded/down, `Quorum`/`ReplicateOp` quorum writes), and `Consensus` (deterministic leader election with a leadership `Term`)
- [x] Phase 25.4: Context Management — `SmartContextWindow` (priority-based chunk inclusion, `RAGPipeline.WithSmartContext`), `ContextCompressor` + `ExtractiveSummarizer` (summarize long contexts), `TrackCitations`/`RenderCitations` (`RAGPipeline.WithCitations` → `RAGResponse.Citations`), and `HallucinationDetector` (lexical claim-support + `HallucinationRate`)
- [x] Phase 23.1: Metrics & Observability foundation — new stdlib-only `metrics` package: thread-safe `Registry` with `Counter`/`Gauge`/`Histogram` (fixed buckets + bounded reservoir for p50/p95/p99), Prometheus text export (`Registry.RenderPrometheus` + `Registry.HTTPHandler()` for a `/metrics` endpoint), structured `Logger` (JSON or key=value) with correlation-ID propagation, ready-made bundles (`StoreMetrics`, `EmbeddingMetrics`, `CacheMetrics`, `GraphMetrics`), and `InstrumentedStore` — a drop-in `store.Store` wrapper that records search/upload latency, throughput, error rate, and store size
- [x] Phase 23.2: Tracing — new stdlib-only, OpenTelemetry-compatible `tracing` package: 128-bit trace / 64-bit span IDs, spans with kinds/attributes/events/status and context propagation, W3C `traceparent` inject/parse + `StartRemoteSpan` for cross-node correlation, pluggable `SpanProcessor`s (`InMemoryProcessor` grouped by trace, `ConsoleProcessor`), and `InstrumentedTracingStore` — a drop-in `store.Store` wrapper that records the upload → search → retrieve path as parent/child spans with metadata tags. The OTel SDK is intentionally not bundled (zero-dependency/zero-CGO); `SpanProcessor` is the bridge point
- [x] Phase 23.3: Health & Diagnostics — `store.HealthCheck()` (connectivity + size + namespaces + SQLite structural integrity → `store.HealthReport`), `distributed.ShardDistribution()` (per-node shard stats), and HTTP diagnostics endpoints: `store.HealthHandler` and `distributed.HealthHandler` serve `/healthz` (200 when healthy, 503 otherwise) and `/diagnostics` (JSON snapshot)
- [x] Phase 23.4: Query Analytics — new stdlib-only `analytics` package: bounded, thread-safe `QueryLog` (ring buffer) recording query latency/results, `PopularQueries` (trending detection), `DropOff` (queries with no good results), pluggable `Sink`s (`FileSink` NDJSON, `HTTPSink` POST; message-queue via the same interface), and `InstrumentedAnalyticsStore` — a drop-in `store.Store` wrapper. **Phase 23 (Observability & Monitoring) is now complete.**
- [x] Phase 24.1: Relevance Feedback — new stdlib-only `feedback` package: `Label`/`Feedback` + thread-safe `Collector` (a "training store" with `ToMetadata()` for persistence), the classic `Rocchio` query-expansion algorithm in both **vector** form (`Q' = αQ + β·mean(relevant) − γ·mean(irrelevant)`) and **lexical** form (expanded query string), and `RelevanceFeedback.ExpandAndRetrieve` (retrieve → mark relevant → Rocchio → re-search → boost relevant) using caller-side `VectorSearcher`/`ChunkGetter`/`Embedder` interfaces
- [x] Phase 24.2: Evaluation Framework — new stdlib-only `eval` package: retrieval metrics (Precision@K, Recall@K, MRR, NDCG@K, graded or binary relevance), `RAGEval` answer quality (faithfulness/relevance/correctness) via a pluggable `Judge` (deterministic `OverlapJudge` included), `Dataset` (JSON load/save), `BenchmarkSuite` (`Run`/`RunWithAnswers` + `Compare` for baseline regression), and `Report` (JSON/Markdown)
- [x] Phase 24.3: Human-in-the-Loop — new stdlib-only `hitl` package: thread-safe `ReviewQueue` (de-duplicated, highest-uncertainty-first, approve/reject lifecycle), `Annotation`/`AnnotationStore` (relevance/correction/feedback, indexed by chunk + ID), and `ActiveLearning` (least-confidence `UncertaintyFromScores`, top-1/top-2 `Margin`, and `Select` to enqueue the most uncertain candidates for review)
- [x] Phase 24.4: Automated Testing — new `testutil` package (import from `_test.go` only): `FixtureStore` (preloaded deterministic `MemoryStore` with predictable chunk IDs), `MockLLM` (scripted `llm.Backend` with streaming + call tracking), `MockEmbedder`/`DeterministicEmbed` (deterministic vectors), `Golden`/`GoldenJSON` (golden-file comparison with `UpdateGolden` refresh mode) and `BenchmarkHarness` (warmup + correct timer handling); CI workflow gains an overall coverage regression gate (≥ 80%); also fixed a pre-existing data race in the ingest worker pool
- [x] Phase 27.1: REST API — new stdlib-only `api` package: `Server`/`Handler` on Go 1.22+ `ServeMux` routing with `POST /upload`, `GET /search`, `POST /hybrid-search`, `POST /rag`, `GET /graph/{entity}` (ID + label fallback), `POST /graph/reason` (NL reasoning + path exploration), and operational `/healthz`, `/readyz`, `/diagnostics`; OpenAPI 3.0 spec embedded via `go:embed` and served at `GET /openapi.json`; pluggable `Authenticator` (API keys via `X-API-Key`/Bearer, HS256 JWTs verified with stdlib `crypto/hmac`, or `Composite`), CORS, and per-request body limits; `cmd/recall-server` is the standalone entrypoint (config-driven store/embedder/pipeline/graph wiring, graceful SIGINT/SIGTERM shutdown, `-health-probe` mode for curl-less containers)
- [x] Phase 27.3: Configuration — new `config` package: JSON/YAML file loading (by extension) with defaults and multi-problem `Validate()`, environment overrides via `RECALL__SECTION__KEY` (double-underscore nesting; malformed values ignored), and `config.Watcher` hot reload (mtime/size polling, validated reload through a callback, invalid edits skipped and reported via `LastError`). **Phase 27.2 (gRPC) is intentionally deferred** — it requires protobuf codegen and the `grpc` dependency, and the roadmap marks it as future work
- [x] Phase 27.4: Deployment — `deploy/Dockerfile` (multi-stage, `CGO_ENABLED=0` pure-Go build on distroless/nonroot, in-image `HEALTHCHECK` via the binary's `-health-probe` mode), `deploy/docker-compose.yml` (single node: mounted config + SQLite volume + env overrides, with a documented multi-node template), and `deploy/kubernetes/recall.yaml` (ConfigMap, Secret, Deployment with `/readyz`/`/healthz` probes and resource limits, Service, HPA). Example configs in `deploy/config/` are regression-tested by the `config` package
- [x] Phase 28.1 (partial): Security — **namespace-scoped API keys** (`api.ScopedAPIKeyAuth` + `auth.scoped_keys`): per-key namespace restrictions enforced on every data endpoint (uploads `403` outside scope; search/hybrid/RAG retrieval restricted via a `core.MetadataKeyNamespace` filter; graph entities, relations, and reasoning paths filtered with out-of-scope entities reported as 404 — fail closed). Stores now stamp `core.MetadataKeyNamespace` on every chunk at upload, and `pipeline.RAGPipeline` gained race-safe `Clone()`/`WithSearchFilters()` for request-scoped retrieval. While adding scoping, two pre-existing bugs were fixed: memory-store hybrid search bypassed metadata filters for keyword-only (BM25) matches, and the SQLite store corrupted typed metadata values (`core.String` & co.) after a DB round-trip (legacy rows now unwrap transparently). The remainder of Phase 28 (RBAC, row-level security, at-rest/in-transit encryption, audit logging, vault, compliance) is **deferred** until a multi-tenant hosted service is needed — see the [Security Guidance](#security-guidance) section for the operational guidance that covers library and single-tenant service use in the meantime
- [x] Phase 29: CLI Tool — new `cmd/recall` cobra binary with two execution modes: **local** (in-process against the configured SQLite/memory store) and **server** (typed HTTP client of a running recall-server via `--server` or `cli.url`). Data commands: `upload` (files + recursive directories through `loader`), `search`, `hybrid-search`, `rag` (assembled context + rendered prompt with citations), `graph <entity>`/`graph list`, `reason` (NL reasoning or `--from`/`--to` path exploration). Management commands: `store info` (health, namespaces, schema version, integrity), `store migrate` (versioned `-- recall-migration:` SQL files, transactional + idempotent), `store backup` (online `VACUUM INTO`), `store restore` (atomic temp-file + rename with `--force` guard), `cluster status` (probes node `/diagnostics`, exit 1 when a node is down/unreachable). Evaluation commands: `eval` (Precision/Recall/MRR/NDCG@K over an `eval.Dataset`, `--save` JSON report) and `eval compare` (tolerance-gated regression check, exit 2 on regression — a CI gate). All commands render table/JSON/YAML (`-o`); configuration resolves `--config` → `$HOME/.recall.yaml` (.yml/.json) → defaults with `RECALL__SECTION__KEY` env overrides, and gains a `cli` section (url, api_key, timeout, output, cluster_nodes). Service assembly moved into the new `app` package so recall-server and the CLI always wire components the same way; the typed REST client lives in the new `client` package

## Roadmap

The full development roadmap is documented in [ROADMAP.md](./ROADMAP.md). Current priorities:

| Priority | Phase | Focus | Estimated Effort |
|----------|-------|-------|-----------------|
| **1 — Foundation** | 19 | Test coverage hardening (all packages → ≥80%) | ~4 weeks |
| | 21 | Document ingestion pipeline (PDF, DOCX, HTML, connectors) | |
| | 26 | Advanced retrieval (PQ/SQ indexing, HyDE, parent-child chunking) | |
| **2 — Production** | 20 | Real embedding providers (OpenAI, Cohere, local ONNX) | ~10 weeks |
| | 22 | Cross-encoder reranking for retrieval quality | |
| | 25 | Resilience (retry, circuit breaker, rate limiting, backup) | |
| **3 — Growth** | 23 | Observability (metrics, tracing, health checks) | ~10 weeks |
| | 24 | Feedback loop & evaluation framework (NDCG, relevance feedback) | |
| **4 — Ecosystem** | 27 | REST API & service layer (Docker, K8s) | ~12 weeks |
| | 28 | Security (auth, RBAC, encryption, audit logging) | |
| | 29 | ~~CLI tool~~ ✅ (`recall search`, `recall upload`, etc.) | |
| | 30–32 | Web UI, SDK wrappers (Python/TypeScript), project hygiene | |

**Total estimated effort: ~36–38 weeks for full roadmap.**

### Quick Wins
1. **Test coverage hardening** — `llm/` at 40.2%, `store/` at 49.9% need immediate attention
2. **Project hygiene** — CHANGELOG, CONTRIBUTING, CI/CD, golangci-lint
3. ~~**CLI tool**~~ — ✅ done (`cmd/recall`, see [Command-Line Interface](#command-line-interface-cli))

### Current Coverage Gaps

| Package | Coverage | Target |
|---------|----------|--------|
| `llm/` | 40.2% | ≥80% |
| `store/` | 49.9% | ≥80% |
| `distributed/` | 55.8% | ≥80% |
| `core/` | 67.7% | ≥80% |
| `index/` | 70.9% | ≥80% |

## Testing

```bash
go test ./... -v
```

## License

MIT

## Advanced Usage

### LLM Integration

```go
import (
    "github.com/deagy/recall/llm"
)

// Use OpenAI backend
client := llm.NewOpenAIClient("your-api-key", "")

// Or use Ollama for local models
// client := llm.NewOllamaClient("http://localhost:11434")

// Send a chat request
resp, err := client.Chat(ctx, &llm.ChatRequest{
    Messages: []llm.Message{
        {Role: "system", Content: "You are a helpful assistant."},
        {Role: "user", Content: "What is Go programming language?"},
    },
    Model:     "gpt-4",
    MaxTokens: 1000,
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(resp.Message.Content)

// Use streaming
client.ChatStream(ctx, &llm.ChatRequest{
    Messages: []llm.Message{
        {Role: "user", Content: "Tell me about Go"},
    },
    Stream: true,
}, func(chunk *llm.StreamChunk) error {
    fmt.Print(chunk.Delta.Content)
    return nil
})

// Extract entities using LLM
extractor := llm.NewLLMExtractor(client, "gpt-4")
entities, err := extractor.ExtractEntities(ctx, "Go was created by Google in 2007", "chunk-1")
if err != nil {
    log.Fatal(err)
}
```
### Advanced Query Processing

```go
import (
    "github.com/deagy/recall/graph"
    "github.com/deagy/recall/query"
)

// Create a parser with optional knowledge graph
g := graph.NewKnowledgeGraph()
parser := query.NewDefaultParser(g)

// Parse a query
parsed, err := parser.Parse(ctx, "How does Go compare to Python?")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Intent: %s\n", parsed.Intent)
fmt.Printf("Entities: %v\n", parsed.Entities)
fmt.Printf("Confidence: %.2f\n", parsed.Confidence)

// Expand the query using synonyms and graph relations
expander := query.NewGraphExpander(g)
expanded, err := expander.Expand(ctx, parsed)
if err != nil {
    log.Fatal(err)
}

// Use adaptive retrieval
retriever := query.NewAdaptiveRetriever(store, parser, expander)
results, err := retriever.Retrieve(ctx, "How does Go compare to Python?", 10)
if err != nil {
    log.Fatal(err)
}
```
### Hybrid Search with Custom Fusion

```go
import (
    "github.com/deagy/recall/fuse"
    "github.com/deagy/recall/index"
)

// Create a custom fusion strategy
fusion := fuse.NewWeightedFusion(0.7) // 70% vector, 30% BM25

// Use hybrid search with custom fusion
hybridOpts := index.DefaultSearchOptions(10)
hybridOpts.BM25Weight = 0.3
hybridOpts.Fusion = fusion

results, err := s.SearchHybrid(ctx, "query", hybridOpts)
```

### Knowledge Graph Operations

```go
import (
    "github.com/deagy/recall/graph"
)

// Create a knowledge graph
g := graph.NewKnowledgeGraph()

// Add entities
g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
g.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
g.AddEntity(graph.NewEntity("go", "Go", graph.EntityConcept))

// Add relations
g.AddRelation(graph.NewRelation("alice", "go", "uses", 0.9))
g.AddRelation(graph.NewRelation("bob", "go", "uses", 0.8))

// Find paths	path := g.FindPath("alice", "bob")
if path != nil {
    fmt.Printf("Path: %s\n", path.String())
}

// Get neighbors
neighbors := g.Neighbors("alice")
for _, n := range neighbors {
    fmt.Printf("Alice knows: %s\n", n.Name)
}

// Transitive closure
closure := g.TransitiveClosure("alice")
fmt.Printf("Alice can reach %d entities\n", len(closure))
```

### Multi-Hop Reasoning

```go
import (
    "github.com/deagy/recall/reasoning"
)

// Create reasoning engine
engine := reasoning.NewEngine(
    reasoning.Config{
        MaxDepth: 5,
        MinConfidence: 0.5,
    },
)

// Add inference rules
engine.AddRule(reasoning.NewTransitiveRule())
engine.AddRule(reasoning.NewSymmetricRule())

// Explore paths
paths, err := engine.ExplorePaths(g, "alice", "bob", 3)
if err != nil {
    log.Fatal(err)
}

for _, path := range paths {
    fmt.Printf("Path: %s (confidence: %.2f)\n", path.String(), path.Confidence)
}
```

## Running Benchmarks

```bash
# Run all benchmarks
go test ./... -bench=. -benchmem

# Run specific package benchmarks
go test ./index/ -bench=. -benchmem
go test ./bm25/ -bench=. -benchmem
go test ./chunker/ -bench=. -benchmem
go test ./graph/ -bench=. -benchmem

# Run with profiling
go test ./... -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof
```

## Performance Characteristics

| Operation | Performance | Notes |
|-----------|-------------|-------|
| HNSW Add | ~876 ns/op | Scales with dimension |
| HNSW Search | ~69 μs/op (1K docs) | ~713 μs/op (10K docs) |
| BM25 AddDocument | ~2.6 μs/op | Includes tokenization |
| BM25 Search | ~6 μs/op (1K corpus) | Fast keyword search |
| Fixed Chunking | ~81 ns/op (small) | ~28 μs/op (large) |
| Recursive Chunking | ~44 ns/op (small) | ~18 μs/op (large) |
| Graph AddEntity | ~144 ns/op | Fast entity addition |
| Graph FindPath | ~1 μs/op | BFS traversal |
| Store Upload | ~3.7 μs/op | Includes chunking + embedding |
| Store Search | ~2.6 μs/op | Fast similarity search |

## License

MIT

## Multi-hop Reasoning

```go
import (
    "github.com/deagy/recall/graph"
    "github.com/deagy/recall/reasoning"
)

// Create a knowledge graph
g := graph.NewKnowledgeGraph()
g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
g.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
g.AddEntity(graph.NewEntity("go", "Go", graph.EntityConcept))
g.AddEntity(graph.NewEntity("google", "Google", graph.EntityOrganization))

// Add relations
g.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
g.AddRelation(graph.NewRelation("bob", "go", "uses", 0.8))
g.AddRelation(graph.NewRelation("go", "google", "created_by", 0.7))

// Create reasoning engine
engine := reasoning.NewEngine(g, reasoning.DefaultConfig())

// Explore paths between entities
paths := engine.ExplorePaths("alice", "google")
for _, path := range paths {
    fmt.Printf("Path: %s\n", path.String())
}

// Infer relations using rules
inferred := engine.InferRelations()
for _, ir := range inferred {
    fmt.Printf("Inferred: %s\n", ir.String())
}

// Reason about natural language queries
results := engine.Reason("Who created Go?", 3)
for _, r := range results {
    fmt.Printf("Answer: %s -> %s (confidence: %.2f)\n", r.From, r.To, r.Confidence)
}
```

### Custom Inference Rules

```go
// Create custom inverse rule
inverseRule := &reasoning.InverseRule{
    Mappings: map[string]string{
        "works_at": "works_for",
        "created_by": "created",
    },
    MinWeight: 0.5,
}

// Create composition rule
compositionRule := &reasoning.CompositionRule{
    Rules: map[string]string{
        "located_in|works_at": "works_in",
    },
    MinWeight: 0.5,
}

// Create engine with custom rules
engine := reasoning.NewEngine(g, reasoning.Config{
    MaxDepth:      5,
    MinConfidence: 0.3,
    Rules: []reasoning.InferenceRule{
        inverseRule,
        compositionRule,
    },
})
```

### Confidence Propagation

```go
// Use product aggregator (default)
productAgg := &reasoning.ProductAggregator{Decay: 0.9}

// Use minimum aggregator
minAgg := &reasoning.MinAggregator{Decay: 0.9}

// Use average aggregator
avgAgg := &reasoning.AverageAggregator{Decay: 0.9}

// Extract entities from text
extractor := reasoning.NewEntityExtractor()
entities := extractor.ExtractEntities("Alice works at Google in New York")

// Expand query with synonyms
expanded := extractor.ExpandQuery("Go")
// Returns: ["go", "golang", "gopher"]
```

## Running Benchmarks

```bash
# Run all benchmarks
go test ./... -bench=. -benchmem

# Run specific package benchmarks
go test ./reasoning/ -bench=. -benchmem

# Run with profiling
go test ./... -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof
```
