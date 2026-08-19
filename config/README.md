# config

Configuration for recall-server and the CLI: typed schema, file loading
(YAML/JSON), environment overrides, and validation.

## Schema

```yaml
server:  { host, port, max_upload_bytes, read/write/idle timeouts, allow_cors }
store:   { backend: memory|sqlite, path, namespace,
           embedder: { type: mock|openai|cohere|ollama|onnx, model, dimension, api_key_env },
           chunking: { ... } }
auth:    { api_keys / jwt settings }
cli:     { url, api_key, timeout, output, cluster_nodes }
```

Type constants: `BackendMemory` / `BackendSQLite`, `EmbedderMock` /
`EmbedderOpenAI` / `EmbedderCohere` / `EmbedderOllama` / `EmbedderONNX`.

## Loading & precedence

```go
cfg, err := config.Load("recall.yaml")  // .yaml, .yml, or .json
cfg.ApplyEnv("")                        // RECALL__SECTION__KEY overrides file values
err = cfg.Validate()                    // fail fast on bad config
cfg.WithDefaults()                      // dev defaults (in-memory store, mock embedder)
```

Precedence: **environment > file > defaults**.

## Security note

API keys are **never stored in config files**: `EmbedderConfig.APIKeyEnv`
names the environment variable holding the key, and `app.BuildEmbedder`
reads it from the environment at build time. See SECURITY.md.
