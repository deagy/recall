# api

The REST API server: HTTP handlers for store, RAG, graph, and reasoning,
plus authentication middleware. `openapi.json` documents the contract.

## Server

```go
srv, err := api.NewServer(api.Config{
    Store:    store,
    Pipeline: ragPipeline,
    Graph:    graphStore,
    Reasoner: reasoner,
    // Host, Port, timeouts, Auth, ExemptPaths...
})
srv.ListenAndServe()      // or mount srv.Handler() in your own mux
srv.Shutdown(ctx)         // graceful
```

Endpoints (summary — see `openapi.json`): `/healthz`, `/diagnostics`,
`POST /upload`, `GET/POST /search` (vector + hybrid), `POST /rag`,
graph entity/relation routes, `POST /reason`, chunk delete routes.

## Authentication

- `Authenticator` interface + `RequireAuth` middleware.
- `APIKeyAuth` (`NewAPIKeyAuth(keys...)`, `NewAPIKey()`) — static API
  keys, checked as `Authorization: Bearer` or `X-API-Key`.
- `ScopedAPIKeyAuth` / `KeySpec` — per-key namespace scoping
  (`NamespaceScoper`, `RequestNamespaces`).
- `JWTAuth` (`NewJWTAuth(cfg)`) — JWT bearer auth.
- `Composite` — OR-combine multiple authenticators.

Errors use a stable JSON envelope (`Error`, `ErrCode*` codes).
