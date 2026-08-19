# llm

LLM backends behind one interface, plus production resilience middleware.
Recall treats the LLM as an injected dependency — pick a backend, wrap it
with the middleware you need.

```go
type Backend interface {
    Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    ChatStream(ctx context.Context, req *ChatRequest, fn func(*StreamChunk) error) error
}
```

## Backends

| Backend | Constructor |
|---------|-------------|
| `OpenAIClient` | `NewOpenAIClient(apiKey, baseURL)` — chat + streaming + JSON response format |
| `OllamaClient` | `NewOllamaClient(baseURL)` — local Ollama |
| `MockBackend` | `NewMockBackend()` — deterministic scripted responses for tests/examples |

`ResponseFormat` / `JSONSchema` request structured JSON output;
`ChatRequest` validates model/temperature at construction.

## Resilience middleware

Composable wrappers over any `Backend`:

- `NewRetryBackend(inner, cfg)` — retries with backoff/jitter on
  transient errors.
- `NewRateLimitBackend(inner, cfg)` — token-bucket rate limiting.
- `NewCircuitBreakerBackend(inner, cfg)` — open/half-open/closed states
  (`ErrCircuitOpen` when tripped).
- `NewFallbackBackend(primary, fallbacks...)` — fail over to the next
  backend.
- `NewMiddleware(core, fns...)` + `Build()` — compose any of the above in
  a chosen order.

## Structured extraction

`Extractor` interface + `LLMExtractor` (`NewLLMExtractor(backend, model)`)
extract entities/relations from text via JSON-schema chat;
`MockExtractor` for tests.
