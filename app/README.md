# app

Shared service assembly: builds the full component stack (store → RAG
pipeline → knowledge graph → reasoning engine → API server) from a
`config.Config`. `recall-server` and the `recall` CLI's server mode wire
components **identically** through this package, so behavior never drifts
between the two.

## API

```go
// Store + pipeline + graph + reasoner:
svc, cleanup, err := app.BuildService(cfg)      // *Service{Store, Pipeline, Graph, Reasoner}

// Full HTTP server (returns *api.Server):
srv, cleanup, err := app.BuildAPIServer(cfg)
```

`BuildEmbedder(e config.EmbedderConfig)` maps the config's embedder type to
a concrete embedder; API keys come from the environment variable named by
`EmbedderConfig.APIKeyEnv` (keys are never read from config files).
`BuildStore` opens the configured `memory` or `sqlite` store.

`cleanup()` closes everything the build opened — always `defer` it.
