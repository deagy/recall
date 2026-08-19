# reranker

Fine-ranking (reranking) of coarse retrieval results. All rerankers
implement the `Reranker` interface and plug into
`pipeline.RAGPipeline.WithReranker` or `index.SearchOptions`.

## Rerankers

| Reranker | Constructor | Mechanism |
|----------|-------------|-----------|
| `SparseReranker` | `NewSparseReranker()` | BM25-based rescoring (fast, no model) |
| `CrossEncoderReranker` | `NewCrossEncoderReranker(cfg)` / `NewCrossEncoderFile(path, cfg)` | ONNX cross-encoder (local, zero CGO); pluggable tokenizer |
| `LLMReranker` | `NewLLMReranker(cfg)` | LLM-judge scoring via an `llm.Backend` |
| `LTRanker` | `NewLTRanker(cfg)` | pointwise learning-to-rank over `DefaultFeatures` |
| `AdaptiveLTRanker` | `NewAdaptiveLTRanker(cfg)` | LTR with adaptive feature weights |
| `EnsembleReranker` | `NewEnsembleReranker(rerankers...)` | score-fuses several rerankers |

## Training data & experimentation

- `RelevanceSample` / `MarkRelevantIDs` / `MarkTopRelevant` — capture
  labeled examples for LTR training.
- `NewABTest(cfg)` / `Experiment` — run A/B comparisons between reranker
  variants (`VariantMetrics`, NDCG at `DefaultNDCGCutoff`).
