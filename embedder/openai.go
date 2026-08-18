package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Well-known OpenAI embedding model names.
const (
	// ModelTextEmbedding3Small is the default OpenAI embedding model (1536 dims).
	ModelTextEmbedding3Small = "text-embedding-3-small"

	// ModelTextEmbedding3Large is the highest-quality OpenAI embedding model (3072 dims).
	ModelTextEmbedding3Large = "text-embedding-3-large"

	// ModelTextEmbeddingAda002 is the previous-generation OpenAI embedding model (1536 dims).
	ModelTextEmbeddingAda002 = "text-embedding-ada-002"
)

// openaiKnownDimensions maps well-known OpenAI embedding models to their
// native output dimension.
var openaiKnownDimensions = map[string]int{
	ModelTextEmbedding3Small: 1536,
	ModelTextEmbedding3Large: 3072,
	ModelTextEmbeddingAda002: 1536,
}

const (
	defaultOpenAIBatchSize = 100
	maxOpenAIBatchSize     = 2048
	defaultOpenAIURL       = "https://api.openai.com/v1"
	defaultOpenAITimeout   = 30 * time.Second
)

// OpenAIConfig configures an OpenAIEmbedder.
type OpenAIConfig struct {
	// APIKey is the OpenAI API key (required).
	APIKey string

	// Model is the embedding model name (required).
	Model string

	// BaseURL overrides the API endpoint (useful for proxies and tests).
	// Defaults to https://api.openai.com/v1.
	BaseURL string

	// Dimension optionally requests a reduced output dimension (the
	// Matryoshka "dimension" parameter) and is used to validate responses.
	// It must not exceed the model's native dimension. Zero means "use the
	// model's native dimension".
	Dimension int

	// BatchSize is the maximum number of texts per API request.
	// EmbedBatch splits larger inputs automatically.
	// Defaults to 100 and is capped at 2048.
	BatchSize int

	// Retry configures retry and backoff behavior.
	// Zero value uses DefaultRetryConfig.
	Retry RetryConfig

	// HTTPClient overrides the HTTP client. Defaults to a client with a
	// 30s timeout.
	HTTPClient *http.Client

	// Timeout is the request timeout when HTTPClient is nil.
	// Defaults to 30s.
	Timeout time.Duration
}

// OpenAIEmbedder embeds text using the OpenAI embeddings API.
type OpenAIEmbedder struct {
	cfg    OpenAIConfig
	client *http.Client
	retry  RetryConfig
	dim    int
}

// NewOpenAIEmbedder creates a new OpenAIEmbedder, validating the
// configuration (API key, model, and dimension constraints).
func NewOpenAIEmbedder(cfg OpenAIConfig) (*OpenAIEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openai embedder: APIKey is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("openai embedder: Model is required")
	}
	if cfg.Dimension < 0 {
		return nil, fmt.Errorf("openai embedder: Dimension must be >= 0, got %d", cfg.Dimension)
	}

	native, known := openaiKnownDimensions[cfg.Model]
	if !known {
		if cfg.Dimension <= 0 {
			return nil, fmt.Errorf("openai embedder: unknown model %q — set Dimension explicitly", cfg.Model)
		}
		native = cfg.Dimension
	}
	if cfg.Dimension > native {
		return nil, fmt.Errorf("openai embedder: Dimension %d exceeds the native dimension %d of %s", cfg.Dimension, native, cfg.Model)
	}
	dim := cfg.Dimension
	if dim == 0 {
		dim = native
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultOpenAIURL
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = defaultOpenAIBatchSize
	}
	if cfg.BatchSize > maxOpenAIBatchSize {
		cfg.BatchSize = maxOpenAIBatchSize
	}
	client := cfg.HTTPClient
	if client == nil {
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = defaultOpenAITimeout
		}
		client = &http.Client{Timeout: timeout}
	}

	return &OpenAIEmbedder{
		cfg:    cfg,
		client: client,
		retry:  cfg.Retry,
		dim:    dim,
	}, nil
}

// Dimension returns the embedding dimension of this embedder.
func (e *OpenAIEmbedder) Dimension() int {
	return e.dim
}

// Embed converts a single text string into an embedding vector.
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch converts multiple texts into embedding vectors, automatically
// splitting the input into API-sized batches.
func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

// embedRequest sends one API request for the given (already size-bounded)
// slice of texts and validates the response.
func (e *OpenAIEmbedder) embedRequest(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := openaiEmbedRequest{
		Model: e.cfg.Model,
		Input: texts,
	}
	if e.cfg.Dimension > 0 {
		reqBody.Dimension = &e.cfg.Dimension
	}

	var resp openaiEmbedResponse
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

	if len(resp.Data) != len(texts) {
		return nil, &apiError{message: fmt.Sprintf("openai: expected %d embeddings, got %d", len(texts), len(resp.Data))}
	}
	out := make([][]float32, len(texts))
	for i, item := range resp.Data {
		if item.Index != i {
			return nil, &apiError{message: fmt.Sprintf("openai: out-of-order embedding at position %d (index %d)", i, item.Index)}
		}
		if len(item.Embedding) != e.dim {
			return nil, &apiError{message: fmt.Sprintf("openai: embedding %d has %d dimensions, expected %d", i, len(item.Embedding), e.dim)}
		}
		out[i] = item.Embedding
	}
	return out, nil
}

// doHTTP performs a single POST to the OpenAI embeddings endpoint.
func (e *OpenAIEmbedder) doHTTP(ctx context.Context, body openaiEmbedRequest) (openaiEmbedResponse, error) {
	var out openaiEmbedResponse

	data, err := json.Marshal(body)
	if err != nil {
		return out, fmt.Errorf("openai: failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.BaseURL+"/embeddings", bytes.NewReader(data))
	if err != nil {
		return out, fmt.Errorf("openai: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return out, fmt.Errorf("openai: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return out, fmt.Errorf("openai: failed to read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return out, &apiError{status: resp.StatusCode, message: string(respBody)}
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return out, &apiError{message: fmt.Sprintf("openai: invalid response JSON: %v", err)}
	}
	return out, nil
}

// openaiEmbedRequest represents an OpenAI embeddings API request.
type openaiEmbedRequest struct {
	Model     string   `json:"model"`
	Input     []string `json:"input"`
	Dimension *int     `json:"dimension,omitempty"`
}

// openaiEmbedResponse represents an OpenAI embeddings API response.
type openaiEmbedResponse struct {
	Object string            `json:"object"`
	Data   []openaiEmbedItem `json:"data"`
	Model  string            `json:"model"`
	Usage  openaiEmbedUsage  `json:"usage"`
}

// openaiEmbedItem is a single embedding in an OpenAI response.
type openaiEmbedItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// openaiEmbedUsage contains token usage statistics.
type openaiEmbedUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
