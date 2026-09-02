# Recall — Advanced Features Roadmap

This document outlines the plan for advanced features beyond the current Phase 12 completion. These features will enhance Recall's capabilities in query processing, LLM integration, scalability, and intelligence.

---

## Phase 13: Advanced Query Processing

**Goal:** Transform natural language queries into optimized graph traversals and retrieval strategies.

### Features

#### 13.1 Query Parser & Intent Detection
- **QueryParser** interface for analyzing query structure
- Intent classification: factual, comparative, temporal, causal
- Entity extraction with confidence scores
- Query decomposition for complex multi-part questions

```go
type QueryParser interface {
    Parse(ctx context.Context, query string) (*ParsedQuery, error)
}

type ParsedQuery struct {
    Original   string
    Intent     Intent
    Entities   []ExtractedEntity
    Relations  []ExtractedRelation
    SubQueries []string  // for decomposed queries
}
```

#### 13.2 Query Expansion & Synonyms
- Automatic query expansion using knowledge graph relations
- Synonym injection from entity metadata
- Temporal normalization (e.g., "last year" → specific date range)
- Negation handling ("people who are NOT X")

#### 13.3 Multi-Modal Query Support
- Support for structured filters alongside natural language
- Hybrid search optimization (vector + BM25 + graph traversal)
- Query rewriting based on retrieval feedback

#### 13.4 Adaptive Retrieval Strategies
Learn from user corrections, vary `topK` with query complexity rather than
fixing it, and fall back to keyword or graph traversal when vector search
returns nothing useful.

### Implementation Plan

1. **Week 1-2**: QueryParser interface + intent detection
   - Create `query/parser.go` with intent classification
   - Add entity extraction using existing HeuristicNER
   - Unit tests for each intent type

2. **Week 3**: Query expansion engine
   - Create `query/expander.go`
   - Integrate with knowledge graph for synonym injection
   - Benchmark expansion overhead

3. **Week 4**: Adaptive retrieval
   - Modify RAGPipeline to accept ParsedQuery
   - Implement relevance feedback loop
   - Add query-specific topK configuration

### Expected Outcomes
- 20-30% improvement in retrieval relevance for complex queries
- Support for 5+ query intents (factual, comparative, temporal, causal, procedural)
- Query expansion adds <1ms overhead

---

## Phase 14: Real LLM Integration

**Goal:** Pluggable LLM backend for entity extraction, relation inference, and answer generation.

### Features

#### 14.1 LLM Backend Interface
```go
type LLMBackend interface {
    Generate(ctx context.Context, prompt string, opts GenerateOptions) (*LLMResponse, error)
    Embed(ctx context.Context, text string) ([]float32, error)
    Dimension() int
    Close() error
}

type GenerateOptions struct {
    MaxTokens   int
    Temperature float64
    TopP        float64
    StopSequences []string
}

type LLMResponse struct {
    Text    string
    Tokens  int
    Latency time.Duration
    Model   string
}
```

#### 14.2 OpenAI Integration
- Support for GPT-4, GPT-3.5-Turbo
- Function calling for structured extraction
- Embedding API integration (text-embedding-ada-002, text-embedding-3-small)
- Rate limiting and retry logic

#### 14.3 Local Model Support
- Ollama integration (Llama, Mistral, etc.)
- vLLM compatibility
- ONNX Runtime for local inference
- Model quantization support (GGUF, GPTQ)

#### 14.4 Streaming Responses
- SSE (Server-Sent Events) streaming support
- Token-by-token processing for latency-sensitive apps
- Streaming entity extraction

#### 14.5 LLM-Assisted Extraction
- Replace heuristic NER with LLM-powered extraction
- Relation extraction with few-shot examples
- Confidence scoring from LLM logits

### Implementation Plan

1. **Week 1**: LLMBackend interface + OpenAI client
   - Create `llm/openai.go`
   - Implement Generate, Embed, Close
   - Add API key configuration and rate limiting
   - Tests with mock HTTP server

2. **Week 2**: Ollama integration
   - Create `llm/ollama.go`
   - Support for local model inference
   - Model listing and selection

3. **Week 3**: Streaming support
   - Add `GenerateStream` method to interface
   - SSE handling for OpenAI and Ollama
   - Token callback mechanism

4. **Week 4**: LLM-assisted extraction
   - Integrate with graph/extract.go
   - Few-shot prompt templates
   - Benchmark vs heuristic extraction

### Expected Outcomes
- Support for 3+ LLM backends (OpenAI, Ollama, local)
- Entity extraction accuracy: 85%+ F1 on standard datasets
- Streaming latency: <100ms per token
- Graceful degradation when LLM unavailable

---

## Phase 15: Distributed Storage

**Goal:** Scale Recall beyond single-node deployments with sharding, replication, and consensus.

### Features

#### 15.1 Storage Sharding
```go
type ShardConfig struct {
    NumShards    int
    ShardingKey  func(doc *core.Document) string
    Replication  int  // replication factor
}

type ShardedStore interface {
    Store
    GetShard(docID string) Shard
    Rebalance() error
}
```

- Hash-based sharding (consistent hashing)
- Automatic shard assignment
- Shard migration support

#### 15.2 Replication
- Leader-follower replication
- Read/write quorum configuration
- Conflict resolution (last-writer-wins, custom)
- Cross-region replication

#### 15.3 Consensus Protocol (Optional)
- Raft-based consensus for multi-master setups
- Eventual consistency guarantees
- Partition tolerance

#### 15.4 Distributed Search
- Scatter-gather query execution
- Local BM25 + global vector search
- Result merging and re-ranking

#### 15.5 Storage Backend Abstraction
```go
type StorageBackend interface {
    Open(ctx context.Context, config map[string]string) error
    Put(key string, value []byte) error
    Get(key string) ([]byte, error)
    Delete(key string) error
    Scan(prefix string) ([]string, error)
    Close() error
}
```

- SQLite (existing)
- BoltDB (embedded)
- Redis (external)
- S3-compatible object storage

### Implementation Plan

1. **Week 1-2**: ShardedStore interface + consistent hashing
   - Create `store/sharded.go`
   - Implement consistent hashing with virtual nodes
   - Unit tests for shard assignment

2. **Week 3**: Replication
   - Create `store/replica.go`
   - Leader-follower sync
   - Quorum read/write

3. **Week 4**: Distributed search
   - Modify Search to support multiple shards
   - Implement scatter-gather
   - Benchmark distributed vs single-node

### Expected Outcomes
- Support for 1000+ chunks per shard
- Linear scalability with shard count
- <10ms latency overhead for replication
- Support for 5+ storage backends

---

## Phase 16: Streaming & Semantic Chunking

**Goal:** Move beyond fixed-size chunking to semantic, streaming-aware chunking strategies.

### Features

#### 16.1 Semantic Chunker
```go
type SemanticChunker struct {
    Embedder    embedder.Embedder
    Threshold   float64  // similarity threshold for chunk boundaries
    MaxTokens   int
    MinTokens   int
}

func (s *SemanticChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error)
```

- Split at semantic boundaries (similarity drops)
- Use embedding similarity to detect topic shifts
- Configurable threshold for granularity

#### 16.2 Sentence-Boundary Chunker
- Split at sentence boundaries (using NLP tokenizer)
- Merge sentences until max tokens reached
- Preserve grammatical structure

#### 16.3 Sliding Window with Semantic Overlap
- Configurable overlap based on semantic similarity
- Ensure chunk boundaries don't split related content
- Adaptive overlap based on document structure

#### 16.4 Streaming Chunker
- Process documents incrementally (useful for live feeds)
- Stateless chunking (no full document required)
- Buffer management for memory-constrained environments

```go
type StreamingChunker interface {
    Feed(token string) ([]*core.Chunk, bool)  // returns chunks and whether stream is complete
    Flush() ([]*core.Chunk, error)            // flush remaining buffer
    Reset()                                   // reset state
}
```

#### 16.5 Document Structure Awareness
- Markdown-aware chunking (preserve headers, lists)
- Code-aware chunking (preserve function boundaries)
- HTML/XML structure preservation

### Implementation Plan

1. **Week 1**: Semantic chunker
   - Create `chunker/semantic.go`
   - Implement similarity-based splitting
   - Benchmark vs fixed-size chunking

2. **Week 2**: Sentence-boundary chunker
   - Integrate with NLP tokenizer (e.g., `github.com/niemeyer/pretty` or custom)
   - Implement sentence merging logic
   - Tests for grammatical correctness

3. **Week 3**: Streaming chunker
   - Create `chunker/streaming.go`
   - Implement buffer management
   - Tests for incremental processing

4. **Week 4**: Structure-aware chunking
   - Add Markdown parser
   - Implement code-aware chunking
   - Benchmark structure preservation

### Expected Outcomes
- 15-25% improvement in retrieval relevance with semantic chunking
- Streaming chunker supports 10K+ tokens/sec throughput
- Structure-aware chunking preserves 95%+ of document hierarchy

---

## Phase 17: Graph Embeddings & Link Prediction

**Goal:** Use the graph's structure for link prediction, entity recommendation, and to widen a retrieval beyond what vector similarity alone returns.

### Features

#### 17.1 Node2Vec Embeddings
```go
type Node2Vec struct {
    Graph      *graph.KnowledgeGraph
    Dimension  int
    Walks      int
    WalkLength int
    p, q       float64  // return and inout parameters
}

func (n *Node2Vec) Train(ctx context.Context) error
func (n *Node2Vec) Embed(entityID string) ([]float32, error)
func (n *Node2Vec) SimilarEntities(entityID string, topK int) []*graph.Entity
```

- Random walk-based graph embeddings
- Configurable walk length and number
- p/q parameters for biased exploration

#### 17.2 TransE Embeddings
- Translation-based entity embeddings
- Learn relation vectors
- Support for link prediction

```go
type TransE struct {
    Graph        *graph.KnowledgeGraph
    Dimension    int
    Margin       float64
    LearningRate float64
}

func (t *TransE) Train(ctx context.Context, epochs int) error
func (t *TransE) PredictRelation(head, tail string) float64
func (t *TransE) PredictEntity(head string, relation string) []*graph.Entity
```

#### 17.3 Link Prediction
- Score potential missing edges
- Recommend new relations
- Anomaly detection (unexpected relations)

```go
type LinkPredictor struct {
    Embeddings graph.EmbeddingStore
}

func (lp *LinkPredictor) Score(head, tail string) float64
func (lp *LinkPredictor) Predict(head string, topK int) []*graph.Entity
```

#### 17.4 Entity Recommendation
- "People you may know" style recommendations
- Based on graph structure + embeddings
- Confidence scoring

### Implementation Plan

1. **Week 1-2**: Node2Vec implementation
   - Create `graph/embedding.go`
   - Implement random walks
   - Skip-gram training (simplified)
   - Unit tests for embedding quality

2. **Week 3**: TransE implementation
   - Extend embedding.go with TransE
   - Implement loss function and optimization
   - Link prediction tests

3. **Week 4**: Link prediction API
   - Create `graph/link_predictor.go`
   - Integrate with existing graph
   - Benchmark prediction accuracy

### Expected Outcomes
- Node2Vec embeddings: 70%+ accuracy on link prediction (standard datasets)
- TransE: Support for 10K+ entities
- Link prediction latency: <1ms per query
- Entity recommendation: Top-10 recall >60%

---

## Phase 18: Caching Layer

**Goal:** Reduce latency and cost for repeated queries through intelligent caching.

### Features

#### 18.1 Query Result Cache
```go
type Cache interface {
    Get(key string) (interface{}, bool)
    Set(key string, value interface{}, ttl time.Duration)
    Delete(key string)
    Clear()
    Stats() CacheStats
}

type CacheStats struct {
    Hits   int
    Misses int
    Size   int
}
```

- LRU (Least Recently Used) eviction
- TTL-based expiration
- Cache key generation (query hash + filters)

#### 18.2 Embedding Cache
- Cache embedding vectors to avoid redundant computation
- Share embeddings across documents
- Automatic invalidation on document update

#### 18.3 Graph Traversal Cache
- Cache common graph traversals (e.g., "find all friends of X")
- Invalidate on graph mutations
- Prefetch popular queries

#### 18.4 Multi-Level Caching
```go
type MultiLevelCache struct {
    L1 Cache  // In-memory (fast, small)
    L2 Cache  // Disk-based (slower, larger)
}
```

- L1: In-memory LRU (Go map-based)
- L2: Disk-based (SQLite or BoltDB)
- Automatic promotion/demotion

#### 18.5 Cache Warming
- Pre-populate cache with popular queries
- Background warming on startup
- Analytics-driven cache warming

### Implementation Plan

1. **Week 1**: Cache interface + LRU implementation
   - Create `cache/lru.go`
   - Implement thread-safe LRU
   - Unit tests for eviction

2. **Week 2**: Query result cache
   - Create `cache/query.go`
   - Integrate with RAGPipeline
   - Benchmark hit/miss performance

3. **Week 3**: Multi-level caching
   - Create `cache/multi.go`
   - Implement L1/L2 coordination
   - Tests for promotion/demotion

4. **Week 4**: Cache warming + analytics
   - Add cache statistics tracking
   - Implement warming strategies
   - Dashboard integration (optional)

### Expected Outcomes
- 50-80% reduction in query latency for repeated queries
- Cache hit rate: 60%+ for typical workloads
- Memory overhead: <10% of total store size
- Sub-millisecond cache lookup

---

## Implementation Priorities

| Phase | Feature | Priority | Effort | Impact |
|-------|---------|----------|--------|--------|
| 13 | Advanced Query Processing | High | 4 weeks | High |
| 14 | Real LLM Integration | High | 4 weeks | High |
| 18 | Caching Layer | Medium | 4 weeks | Medium |
| 16 | Streaming & Semantic Chunking | Medium | 4 weeks | Medium |
| 17 | Graph Embeddings | Low | 4 weeks | High (research) |
| 15 | Distributed Storage | Low | 4 weeks | High (scale) |

### Recommended Order
1. **Phase 13** (Query Processing) — Immediate value, builds on existing code
2. **Phase 18** (Caching) — Quick wins, reduces latency/cost
3. **Phase 14** (LLM Integration) — Strategic, enables advanced features
4. **Phase 16** (Chunking) — Improves retrieval quality
5. **Phase 17** (Graph Embeddings) — Research-heavy, long-term value
6. **Phase 15** (Distributed) — Infrastructure, needed for scale

---

## Dependencies & Prerequisites

Phase 13 requires Phase 11 (NER) and Phase 10 (Reasoning); Phase 14 requires
Phase 13 for query parsing, and Phase 11; Phase 15 requires Phase 5 (SQLite)
and Phase 6 (HNSW).
- **Phase 16**: Requires Phase 1 (Chunking) and Phase 2 (Embeddings)
- **Phase 17**: Requires Phase 8 (Graph) and Phase 14 (LLM for embeddings)
- **Phase 18**: Requires all previous phases

---

## Success Metrics

| Feature | Metric | Target |
|---------|--------|--------|
| Query Processing | Retrieval relevance (NDCG) | +20% vs baseline |
| LLM Integration | Entity extraction F1 | 85%+ |
| Caching | Query latency reduction | 50-80% |
| Semantic Chunking | Retrieval relevance (NDCG) | +15% vs fixed |
| Graph Embeddings | Link prediction accuracy | 70%+ |
| Distributed | Throughput (queries/sec) | Linear with shards |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API costs | High | Caching, rate limiting, local fallback |
| Graph embedding training time | Medium | Sampling, incremental training |
| Distributed complexity | High | Start with sharding only, add replication later |
| Cache invalidation bugs | Medium | Tests that exercise eviction under concurrent writes; cache-hit metrics |
| Semantic chunking overhead | Low | Benchmark, optimize similarity computation |

---

## Next Steps

1. **Immediate**: Start Phase 13 (Query Processing) — highest ROI
2. **Sprint 2**: Phase 18 (Caching) — quick wins
3. **Sprint 3-4**: Phase 14 (LLM Integration) — strategic investment
4. **Ongoing**: Phase 16 (Chunking) — parallel work
5. **Research**: Phase 17 (Graph Embeddings) — explore feasibility
6. **Future**: Phase 15 (Distributed) — when scale requires it

---

## References

- [Node2Vec paper](https://arxiv.org/abs/1607.00653)
- [TransE paper](https://papers.nips.cc/paper/5535-learning-entity-and-relation-embeddings-for-knowledge-graph-completion.pdf)
- [Consistent Hashing](https://en.wikipedia.org/wiki/Consistent_hashing)
- [Raft Consensus](https://raft.github.io/)
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
