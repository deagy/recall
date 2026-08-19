# tracing

Lightweight OpenTelemetry-compatible tracing (W3C Trace Context) with no
external dependencies: spans in `context.Context`, traceparent
propagation, and pluggable span processors.

## API

```go
tracer := tracing.DefaultTracer()            // or NewTracer(processors...)
ctx, span := tracing.StartSpan(ctx, "search")
span.SetAttributes(map[string]any{"topk": 10})
span.End()
```

- `TraceID` / `SpanID` / `TraceFlags` — OTel-compatible identifiers.
- `TraceParent` + `ParseTraceParent` / `StartRemoteSpan` — continue a
  trace across process boundaries (client → server).
- `SpanProcessor` interface — export spans anywhere (stdout, OTLP bridge,
  in-memory for tests).
- `SpanKind` (internal/client/server/producer/consumer), `SpanStatus`,
  `WithAttributes` / `WithKind` / `WithParent` / `WithTraceID` options.

Use it with `client` (injects traceparent) and `api`/`store` (extracts and
continues) to trace a RAG request end to end.
