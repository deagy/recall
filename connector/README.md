# connector

Fetches documents from **remote** sources into `loader.Document` values —
the remote counterpart of `loader` (local files). All connectors implement:

```go
type Connector interface {
    Fetch(ctx context.Context, ref string) ([]*loader.Document, error)
}
```

## Connectors

| Connector | Source | `ref` |
|-----------|--------|-------|
| `WebConnector` | Web pages / RSS | URL; rate-limited (min gap between requests) |
| `GitConnector` | Git repositories | repo URL (+ optional clone depth, extra args) |
| `S3Connector` | AWS S3 | bucket/prefix; SigV4 signing, pure Go |
| `GitHubConnector` | GitHub repos/issues | owner/repo; API-base configurable for GHES |
| `DatabaseConnector` | SQL databases | table/row source for `database/sql` |

## Used by

`ingest.Pipeline` takes a `Connector` (mutually exclusive with a `Loader`)
and uploads everything `Fetch` returns into the configured store.
