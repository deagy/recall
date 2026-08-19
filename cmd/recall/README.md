# cmd/recall

The `recall` command-line client (cobra-based). It runs in two modes:

- **local** — in-process, against a SQLite or memory store configured by
  the flags/config;
- **server** — typed REST client of a running recall-server
  (`--server` flag or `cli.url` in config), using the `client` package.

## Build & run

```sh
go build -o recall ./cmd/recall
./recall --help
```

## Command groups

| Group | Commands |
|-------|----------|
| Data | `upload` (files + recursive dirs), `search`, `hybrid-search`, `rag` |
| Graph | `graph`, `graph list` — entities and relations |
| Reasoning | `reason` — NL query or `--from`/`--to` path exploration |
| Store | `store info`, `store migrate`, `store backup` (online VACUUM INTO), `store restore` (atomic rename) |
| Cluster | `cluster status` — node diagnostics (exit 1 on down nodes) |
| Eval | `eval` (P/R/MRR/NDCG@K, `--save`), `eval compare` (regression gate, exit 2) |

Output formats: table, JSON, YAML (`-o/--output` or `cli.output`).
Configuration: `--config` → `$HOME/.recall.yaml` → defaults, with
`RECALL__SECTION__KEY` environment overrides (see `config`).
