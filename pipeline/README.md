# pipeline

The RAG workflow: **retrieve → assemble context → render prompt**, plus the
context-management machinery around it.

## RAG pipeline

```go
p := pipeline.NewRAGPipeline(store, nil /* DefaultTemplate */)
p = p.WithTopK(5).WithMinScore(0.2).WithMaxTokens(2048).
    WithCitations().WithSmartContext().
    WithReranker(reranker).WithSearchFilters(termFilter)

resp, err := p.Query(ctx, question)       // vector retrieval
resp, err = p.QueryHybrid(ctx, question)  // hybrid retrieval
// resp.Answer = rendered prompt; resp.Context = assembled context;
// resp.Sources / resp.Citations / resp.Tokens
```

`Template` / `DefaultTemplate()` control the prompt shape (system
instruction, context numbering, question placement). `Clone()` derives a
request-specific pipeline from a shared one without data races.

## Context management

- `ContextWindow` / `SmartContextWindow` — token-budgeted context assembly
  (smart: by relevance score, not just order).
- `ContextCompressor` — summarizes/compacts overflowing context via a
  pluggable `Summarizer`.
- `HallucinationDetector` — flags answers whose content is not supported by
  the retrieved context.
- `MultiModalPipeline` — image+text RAG over `store.MultiModalStore`.
- `WithReranker` / `WithCoarseTopK` / `WithRerankTopK` — two-stage
  retrieve-then-rerank (see `reranker`).
