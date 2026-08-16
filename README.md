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
- **Multi-hop Reasoning** — Pluggable inference rules (transitive, symmetric, anti-symmetric), depth-limited path exploration, confidence propagation, natural language query → graph reasoning
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
├── store/          # High-level store: Memory + SQLite backends, GraphStore
├── pipeline/       # RAG pipeline: context assembly, templates, queries
├── graph/          # Knowledge graph: entities, relations, traversal, inference
├── reasoning/      # Multi-hop reasoning: inference rules, path exploration, confidence propagation
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
- [x] Phase 10: Multi-hop reasoning engine (inference rules, depth-limited path exploration, confidence propagation, natural language query reasoning)
- [x] Phase 13: Advanced query processing (intent detection, entity extraction, query expansion, adaptive retrieval)
- [x] Phase 14: LLM integration (pluggable backends, streaming, LLM-assisted extraction)
- [x] Phase 11: Pluggable NER + relation pattern extraction (HeuristicNER with stopword filtering, PatternRelationExtractor)
- [x] Phase 12: Performance & robustness (context cancellation, SQLite HNSW mirroring, entity extraction heuristics)

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
