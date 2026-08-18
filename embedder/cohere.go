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

// Well-known Cohere embedding model names (both 1024 dimensions).
const (
	// ModelEmbedEnglishV3 is Cohere's English embedding model.
	ModelEmbedEnglishV3 = "embed-english-v3.0"

	// ModelEmbedMultilingualV3 is Cohere's multilingual embedding model.
	ModelEmbedMultilingualV3 = "embed-multilingual-v3.0"
)

// cohereKnownDimensions maps well-known Cohere embedding models to their
// native output dimension.
var cohereKnownDimensions = map[string]int{
	ModelEmbedEnglishV3:      1024,
	ModelEmbedMultilingualV3: 1024,
}

// Cohere input types (the "input_type" request field).
const (
	// InputTypeSearchDocument marks text being embedded for indexing.
	InputTypeSearchDocument = "search_document"

	// InputTypeSearchQuery marks text being embedded as a search query.
	InputTypeSearchQuery = "search_query"

	// InputTypeClassification marks text being embedded for classification.
	InputTypeClassification = "classification"
)

// Cohere truncation strategies (the "truncate" request field).
const (
	// TruncationNone disables truncation (the default).
	TruncationNone = "NONE"

	// TruncationStart truncates the beginning of the input.
	TruncationStart = "START"

	// TruncationEnd truncates the end of the input.
	TruncationEnd = "END"
)

const (
	defaultCohereURL     = "https://api.cohere.ai"
	defaultCohereTimeout = 30 * time.Second
	maxCohereBatchSize   = 96
)

// CohereConfig configures a CohereEmbedder.
type CohereConfig struct {
	// APIKey is the Cohere API key (required).
	APIKey string

	// Model is the embedding model name (required).
	Model string

	// BaseURL overrides the API endpoint (useful for proxies and tests).
	// Defaults to https://api.cohere.ai.
	BaseURL string

	// InputType is the embedding input type.
	// Defaults to InputTypeSearchDocument.
	InputType string

	// Truncation is the truncation strategy ("NONE", "START", "END").
	// Empty string omits the field (Cohere treats it as "NONE").
	Truncation string

	// Dimension optionally overrides the expected output dimension and is
	// used to validate responses. Zero means "use the model's native
	// dimension".
	Dimension int

	// BatchSize is the maximum number of texts per API request.
	// EmbedBatch splits larger inputs automatically.
	// Defaults to 96 (the Cohere API maximum).
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

// CohereEmbedder embeds text using the Cohere embeddings API.
type CohereEmbedder struct {
	cfg    CohereConfig
	client *http.Client
	retry  RetryConfig
	dim    int
}

// NewCohereEmbedder creates a new CohereEmbedder, validating the
// configuration (API key, model, input type, and truncation strategy).
func NewCohereEmbedder(cfg CohereConfig) (*CohereEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("cohere embedder: APIKey is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("cohere embedder: Model is required")
	}
	if cfg.InputType == "" {
		cfg.InputType = InputTypeSearchDocument
	}
	switch cfg.InputType {
	case InputTypeSearchDocument, InputTypeSearchQuery, InputTypeClassification:
	default:
		return nil, fmt.Errorf("cohere embedder: invalid InputType %q", cfg.InputType)
	}
	if cfg.Truncation != "" && cfg.Truncation != TruncationNone && cfg.Truncation != TruncationStart && cfg.Truncation != TruncationEnd {
		return nil, fmt.Errorf("cohere embedder: invalid Truncation %q (want \"NONE\", \"START\", or \"END\")", cfg.Truncation)
	}

	native, known := cohereKnownDimensions[cfg.Model]
	if !known {
		if cfg.Dimension <= 0 {
			return nil, fmt.Errorf("cohere embedder: unknown model %q — set Dimension explicitly", cfg.Model)
		}
		native = cfg.Dimension
	}
	if cfg.Dimension > native {
		return nil, fmt.Errorf("cohere embedder: Dimension %d exceeds the native dimension %d of %s", cfg.Dimension, native, cfg.Model)
	}
	dim := cfg.Dimension
	if dim == 0 {
		dim = native
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultCohereURL
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = maxCohereBatchSize
	}
	if cfg.BatchSize > maxCohereBatchSize {
		cfg.BatchSize = maxCohereBatchSize
	}
	client := cfg.HTTPClient
	if client == nil {
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = defaultCohereTimeout
		}
		client = &http.Client{Timeout: timeout}
	}

	return &CohereEmbedder{
		cfg:    cfg,
		client: client,
		retry:  cfg.Retry,
		dim:    dim,
	}, nil
}

// Dimension returns the embedding dimension of this embedder.
func (e *CohereEmbedder) Dimension() int {
	return e.dim
}

// Embed converts a single text string into an embedding vector.
func (e *CohereEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch converts multiple texts into embedding vectors, automatically
// splitting the input into API-sized batches.
func (e *CohereEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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
func (e *CohereEmbedder) embedRequest(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := cohereEmbedRequest{
		Model:      e.cfg.Model,
		Texts:      texts,
		InputType:  e.cfg.InputType,
		Truncation: e.cfg.Truncation,
	}

	var resp cohereEmbedResponse
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
		return nil, &apiError{message: fmt.Sprintf("cohere: expected %d embeddings, got %d", len(texts), len(resp.Embeddings))}
	}
	out := make([][]float32, len(texts))
	for i, vec := range resp.Embeddings {
		if len(vec) != e.dim {
			return nil, &apiError{message: fmt.Sprintf("cohere: embedding %d has %d dimensions, expected %d", i, len(vec), e.dim)}
		}
		out[i] = vec
	}
	return out, nil
}

// doHTTP performs a single POST to the Cohere /v1/embed endpoint.
func (e *CohereEmbedder) doHTTP(ctx context.Context, body cohereEmbedRequest) (cohereEmbedResponse, error) {
	var out cohereEmbedResponse

	data, err := json.Marshal(body)
	if err != nil {
		return out, fmt.Errorf("cohere: failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.BaseURL+"/v1/embed", bytes.NewReader(data))
	if err != nil {
		return out, fmt.Errorf("cohere: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)
	req.Header.Set("cohere-version", "2022-12-06")

	resp, err := e.client.Do(req)
	if err != nil {
		return out, fmt.Errorf("cohere: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return out, fmt.Errorf("cohere: failed to read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return out, &apiError{status: resp.StatusCode, message: string(respBody)}
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return out, &apiError{message: fmt.Sprintf("cohere: invalid response JSON: %v", err)}
	}
	return out, nil
}

// cohereEmbedRequest represents a Cohere /v1/embed request.
type cohereEmbedRequest struct {
	Model      string   `json:"model"`
	Texts      []string `json:"texts"`
	InputType  string   `json:"input_type"`
	Truncation string   `json:"truncate,omitempty"`
}

// cohereEmbedResponse represents a Cohere /v1/embed response.
type cohereEmbedResponse struct {
	ID         string      `json:"id"`
	Texts      []string    `json:"texts"`
	Embeddings [][]float32 `json:"embeddings"`
}
