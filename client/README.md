# client

The typed REST client for a running recall-server. It is the transport
behind the `recall` CLI's server mode and the natural way for applications
to talk to the service.

```go
c, err := client.New(client.Config{
    BaseURL: "http://localhost:8080",
    APIKey:  os.Getenv("RECALL_API_KEY"), // optional; sent as Bearer + X-API-Key
    Timeout: 30 * time.Second,
})
```

## Methods

| Method | Endpoint |
|--------|----------|
| `Upload(ctx, UploadRequest)` | `POST /upload` — document upload (ID auto-generated when empty) |
| `Search(ctx, query, SearchOptions)` | vector search |
| `HybridSearch(ctx, query, SearchOptions)` | hybrid (BM25+vector) search |
| `RAG(ctx, query, hybrid)` | `POST /rag` — rendered RAG prompt + sources + citations |
| `GraphEntity(ctx, id)` | entity lookup |
| `Reason(ctx, ReasonRequest)` | path exploration / NL reasoning |
| `Health(ctx)` / `Diagnostics(ctx)` | `GET /healthz` / `GET /diagnostics` |

All responses are typed structs mirroring the OpenAPI contract; the client
is safe for concurrent use.
