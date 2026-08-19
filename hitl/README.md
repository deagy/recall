# hitl

Human-in-the-loop: uncertainty estimation, active-learning candidate
selection, and a review queue + annotation store for labeling workflows.

## Uncertainty

- `UncertaintyFromScores(scores)` — per-candidate uncertainty from
  normalized candidate scores.
- `Margin(scores)` — score-margin uncertainty (top-1 vs top-2 gap); low
  margin = model unsure = worth a human look.

## Review workflow

```go
queue := hitl.NewReviewQueue()
queue.Enqueue(item)               // *ReviewItem with candidate scores
item := queue.Next()              // next pending item (StatusPending)
queue.Decide(item, StatusApproved) // approve/reject/...

store := hitl.NewAnnotationStore()
store.Add(hitl.NewAnnotation(chunkID, hitl.AnnotationRelevance, "yes"))
```

`ActiveLearning` (`NewActiveLearning(queue, batchSize)`) pulls batches of
the most uncertain candidates for labeling, closing the loop with
`feedback` (Rocchio) to improve retrieval.
