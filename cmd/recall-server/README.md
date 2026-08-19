# cmd/recall-server

The standalone recall REST API server.

## Build & run

```sh
go build -o recall-server ./cmd/recall-server

# dev defaults: in-memory store, mock embedder, 127.0.0.1:8080
./recall-server

# production: config file (YAML/JSON), env overrides via RECALL__SECTION__KEY
./recall-server -config /etc/recall/recall.yaml

# container health check (no curl needed)
./recall-server -health-probe -probe-url http://127.0.0.1:8080/healthz
```

The server is assembled through the `app` package (`BuildAPIServer`), so
it exposes the same components the CLI and library do: SQLite store, RAG
pipeline, knowledge graph, and reasoning engine behind the `api` package's
REST endpoints (see `openapi.json` in the `api` package). Graceful shutdown
on SIGINT/SIGTERM; TLS via config. See `config/README.md` for the full
schema and `api/README.md` for auth.
