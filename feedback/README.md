# feedback

Relevance feedback (query reformulation from user feedback) — the
interactive loop of the retrieval cycle.

## Components

- `Label` (`LabelUnlabeled`, relevant/irrelevant) and
  `Feedback` (`NewFeedback(query, labels)`) — one query plus per-chunk
  relevance labels.
- `Collector` — accumulates feedback across a session;
  `ErrNoFeedback` when nothing usable has been collected.
- `RelevanceFeedback` (`NewRelevanceFeedback(searcher, getter, embedder)`)
  — turns collected feedback into a reformulated query.

## Algorithms

- **Rocchio (vector)**: `Rocchio(query, relevant, irrelevant, params)` —
  shifts the query vector toward relevant and away from irrelevant chunk
  embeddings (`RocchioParams` / `DefaultRocchioParams`).
- **Rocchio (terms)**: `RocchioTerms(...)` — adjusts the keyword side
  (`TermRocchioParams`), returning the reformulated query string and the
  term weights.
- `BoostRelevant(results, relevant)` — reorders existing results to lift
  feedback-positive chunks.
- Helpers: `CosineSimilarity`, `L2Normalize`, `MeanVectors`.

Interfaces `VectorSearcher` / `ChunkGetter` are defined so the package
stays decoupled from `store`.
