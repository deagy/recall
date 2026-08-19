# query

Query-time intelligence: parsing, rewriting, expansion, and adaptive
retrieval strategies that run **before** search.

## Parsing & filtering

- `QueryParser` / `DefaultParser` — parse a query into a `ParsedQuery`
  (intent, entities, filters); `NewDefaultParser(g)` uses the knowledge
  graph for entity resolution.
- `Filter` / `TermFilter` — convert parsed constraints into
  `index.Filter`s.

## Rewrite strategies (LLM-backed)

| Strategy | Constructor | Effect |
|----------|-------------|--------|
| `Rewriter` | `NewRewriter(backend)` | cleans/clarifies the raw query |
| `HyDE` | `NewHyDE(backend)` | hypothetical document embeddings (retrieves against a generated answer) |
| `StepBack` | `NewStepBack(backend)` | abstract to the broader question first |
| `SubQueryDecomposer` | `NewSubQueryDecomposer()` | split a compound question into sub-queries |
| `Multilingual` | `NewMultilingual(translator, targets...)` | translate the query across languages (`Translator`, `LLMTranslator`, `DetectLanguage`) |

## Expansion & adaptive retrieval

- `Expander` / `GraphExpander` — expand a query with graph-derived related
  terms/entities.
- `AdaptiveRetriever` (`NewAdaptiveRetriever(store, parser, expander)`) —
  parses the query, picks a strategy, expands, and retrieves in one call.
