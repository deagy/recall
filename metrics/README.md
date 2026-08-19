# metrics

A small dependency-free metrics toolkit: counters, gauges, histograms in a
`Registry`, with structured (JSON or text) logging output.

## API

```go
reg := metrics.NewRegistry()
counter := reg.Counter("recall.upload.total")
counter.Inc()

g := reg.Gauge("recall.store.chunks")
g.Set(n)

h := reg.Histogram("recall.search.latency")
h.Observe(elapsed)

log := metrics.NewLogger(os.Stderr, metrics.LevelInfo, /*json*/ true)
```

Field helpers (`metrics.Int("n", 3)`, `metrics.Error("err", err)`, ...)
build structured log fields.

## Built-in instruments

- `StoreMetrics` / `NewInstrumentedStore(store, reg)` — wraps any
  `store.Store`, instrumenting upload/search latency + counts.
- `EmbeddingMetrics` — embed call counts, batch sizes, latency.
- `GraphMetrics` — entity/relation counts and operation timings.
