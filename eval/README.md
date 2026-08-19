# eval

Retrieval and RAG evaluation: datasets with ground truth, ranking metrics,
regression comparison, and (optionally) LLM-judged answer quality.

## Retrieval evaluation

```go
ds := eval.NewDataset("myset")
ds.Add(eval.EvalQuery{
    Query:       "which company designed go",
    RelevantIDs: []string{"chunk-1"},        // binary relevance
    Relevance:   map[string]int{"chunk-1": 2}, // optional graded (NDCG)
})
suite := eval.NewBenchmarkSuite(ds, k)
report, err := suite.Run(ctx, retriever)   // retriever implements Retriever
```

- `Retriever` — `Retrieve(ctx, query, k) ([]string, error)`; adapt any
  store with a 10-line wrapper (see `example/e2e`).
- Metrics: `PrecisionAtK`, `RecallAtK`, `MRR`, `NDCGAtK`,
  `ComputeRetrievalMetrics`; `Report` aggregates means per dataset.
- `Compare(current, baseline, tolerance)` — regression gate used by
  `recall eval compare` (exit 2 on regression, CI-friendly).

## RAG (answer) evaluation

`RAGEval` (`NewRAGEval(judge)`) + `RunWithAnswers` scores generated
answers (faithfulness, correctness) using a pluggable `Judge` (LLM or
heuristic) over `AnswerQuality` per query.

Datasets persist as JSON: `Dataset.Save(path)` / `LoadDataset(path)`.
