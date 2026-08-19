# analytics

Query analytics: record what users search for, how well it went, and where
results drop off — the signal for tuning retrieval.

## Components

- `QueryRecord` — one search event (query, result IDs, scores, timing,
  outcome signals).
- `QueryLog` (`NewQueryLog(maxLen)`) — bounded in-memory query log with
  `QueryCount` aggregation and `DropOffQuery` analysis (queries that
  returned no/weak results).
- `Sink` interface + implementations:
  - `FileSink` (`NewFileSink(path)`) — append NDJSON to a file.
  - `HTTPSink` (`NewHTTPSink(url, client)`) — POST records to an
    endpoint.
- `InstrumentedAnalyticsStore`
  (`NewInstrumentedAnalyticsStore(store, log)`) — wraps a `store.Store`
  so every search is automatically recorded.
