package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"time"
)

// Well-known local embedding models available through Ollama, with their
// output dimensions.
const (
	// ModelAllMiniLML6V2 is a fast, general-purpose sentence model (384 dims).
	ModelAllMiniLML6V2 = "all-minilm-l6-v2"

	// ModelNomicEmbedText is a high-quality general embedding model (768 dims).
	ModelNomicEmbedText = "nomic-embed-text"

	// ModelBGESmallENV15 is a compact BGE model (384 dims).
	ModelBGESmallENV15 = "bge-small-en-v1.5"
)

// ollamaKnownDimensions maps well-known Ollama embedding models to their
// native output dimension.
var ollamaKnownDimensions = map[string]int{
	ModelAllMiniLML6V2:  384,
	ModelNomicEmbedText: 768,
	ModelBGESmallENV15:  384,
}

const (
	defaultOllamaURL     = "http://localhost:11434"
	defaultOllamaTimeout = 60 * time.Second // local inference can be slow
	defaultOllamaBatch   = 32
)

// OllamaConfig configures an OllamaEmbedder.
type OllamaConfig struct {
	// Model is the Ollama embedding model name (required), e.g.
	// "all-minilm-l6-v2" or any model pulled via `ollama pull`.
	Model string

	// BaseURL is the Ollama server URL.
	// Defaults to http://localhost:11434.
	BaseURL string

	// Dimension optionally fixes the expected output dimension (used to
	// validate responses). For models not in ollamaKnownDimensions this is
	// required — or call DetectDimension before using the embedder with a
	// store, since stores pin the dimension at construction time.
	Dimension int

	// BatchSize is the maximum number of texts per API request.
	// EmbedBatch splits larger inputs automatically. Defaults to 32.
	BatchSize int

	// Retry configures retry and backoff behavior.
	// Zero value uses DefaultRetryConfig.
	Retry RetryConfig

	// HTTPClient overrides the HTTP client. Defaults to a client with a
	// 60s timeout.
	HTTPClient *http.Client

	// Timeout is the request timeout when HTTPClient is nil.
	// Defaults to 60s.
	Timeout time.Duration
}

// OllamaEmbedder embeds text using a local model served by an Ollama
// instance. This is the zero-CGO path for local embedding models: Ollama
// handles model download, caching, and CPU inference.
type OllamaEmbedder struct {
	cfg    OllamaConfig
	client *http.Client
	retry  RetryConfig

	// dim holds the resolved dimension (atomic). It is zero until a known
	// model, an explicit config value, or a detection/first request
	// establishes it.
	dim int32
}

// NewOllamaEmbedder creates a new OllamaEmbedder.
func NewOllamaEmbedder(cfg OllamaConfig) (*OllamaEmbedder, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("ollama embedder: Model is required")
	}
	if cfg.Dimension < 0 {
		return nil, fmt.Errorf("ollama embedder: Dimension must be >= 0, got %d", cfg.Dimension)
	}
	dim := cfg.Dimension
	if dim == 0 {
		dim = ollamaKnownDimensions[cfg.Model] // 0 for unknown models
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultOllamaURL
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = defaultOllamaBatch
	}
	client := cfg.HTTPClient
	if client == nil {
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = defaultOllamaTimeout
		}
		client = &http.Client{Timeout: timeout}
	}

	e := &OllamaEmbedder{
		cfg:    cfg,
		client: client,
		retry:  cfg.Retry,
	}
	e.dim = int32(dim)
	return e, nil
}

// Dimension returns the embedding dimension of this embedder. It may return
// 0 for models of unknown dimension until DetectDimension or the first
// Embed call has resolved it.
func (e *OllamaEmbedder) Dimension() int {
	return int(atomic.LoadInt32(&e.dim))
}

// DetectDimension embeds a sample text and returns the resulting vector
// dimension, updating the embedder's known dimension in the process. Use it
// before constructing a store with a model whose dimension is not well-known.
func (e *OllamaEmbedder) DetectDimension(ctx context.Context, sample string) (int, error) {
	vec, err := e.Embed(ctx, sample)
	if err != nil {
		return 0, err
	}
	return len(vec), nil
}

// Embed converts a single text string into an embedding vector.
func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch converts multiple texts into embedding vectors, automatically
// splitting the input into batches.
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.cfg.BatchSize {
		end := start + e.cfg.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := e.embedRequest(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// embedRequest sends one request for the given (already size-bounded) slice
// of texts and validates the response, resolving the dimension lazily when
// it was unknown at construction time.
func (e *OllamaEmbedder) embedRequest(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := ollamaEmbedRequest{Model: e.cfg.Model, Input: texts}

	var resp ollamaEmbedResponse
	err := retry(ctx, e.retry, func() error {
		var decodeErr error
		resp, decodeErr = e.doHTTP(ctx, reqBody)
		if decodeErr != nil {
			return decodeErr
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Embeddings) != len(texts) {
		return nil, &apiError{message: fmt.Sprintf("ollama: expected %d embeddings, got %d", len(texts), len(resp.Embeddings))}
	}

	// The first successful response establishes the dimension when it was
	// unknown at construction time; an explicitly configured dimension is
	// authoritative and used for validation.
	if atomic.LoadInt32(&e.dim) == 0 {
		atomic.StoreInt32(&e.dim, int32(len(resp.Embeddings[0])))
	}
	expected := int(atomic.LoadInt32(&e.dim))
	if expected > 0 {
		for i, vec := range resp.Embeddings {
			if len(vec) != expected {
				return nil, &apiError{message: fmt.Sprintf("ollama: embedding %d has %d dimensions, expected %d", i, len(vec), expected)}
			}
		}
	}
	return resp.Embeddings, nil
}

// doHTTP performs a single POST to the Ollama /api/embed endpoint.
func (e *OllamaEmbedder) doHTTP(ctx context.Context, body ollamaEmbedRequest) (ollamaEmbedResponse, error) {
	var out ollamaEmbedResponse

	data, err := json.Marshal(body)
	if err != nil {
		return out, fmt.Errorf("ollama: failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.BaseURL+"/api/embed", bytes.NewReader(data))
	if err != nil {
		return out, fmt.Errorf("ollama: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return out, fmt.Errorf("ollama: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return out, fmt.Errorf("ollama: failed to read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return out, &apiError{status: resp.StatusCode, message: string(respBody)}
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return out, &apiError{message: fmt.Sprintf("ollama: invalid response JSON: %v", err)}
	}
	return out, nil
}

// ollamaEmbedRequest represents an Ollama /api/embed request.
type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ollamaEmbedResponse represents an Ollama /api/embed response.
type ollamaEmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}
