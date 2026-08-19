# testutil

Shared test helpers used across recall's own test suites (also useful when
building on recall):

- `NewMockEmbedder(dim)` — alias of `embedder.NewMockEmbedder`
  (deterministic vectors).
- `DeterministicEmbed(e, text)` — the canonical embedding of a string under
  a mock embedder (stable across runs).
- `ChunkID(docID, i)` — the chunk ID format stores generate, for
  hand-building expected results.
- `FixtureDoc` / `NewFixtureStore(docs...)` — build a store pre-loaded with
  fixture documents.
- `MockLLM` (`NewMockLLM(responses...)`) — scripted LLM responses.
- `Golden(t, path, got)` / `GoldenJSON` — golden-file assertions; set
  `testutil.UpdateGolden = true` (usually via an `-update` flag) to rewrite
  goldens. `GoldenDiff` renders unified diffs for failure messages.
- `NewHarness(b)` — a micro-benchmark harness for benchmark tests.
- `DefaultFixtureDim` — default fixture embedding dimension.
