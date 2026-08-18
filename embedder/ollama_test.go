package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newOllamaTestServer starts a fake Ollama /api/embed endpoint.
func newOllamaTestServer(t *testing.T, handler func(req ollamaEmbedRequest) (interface{}, int)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed to decode request: %v", err)
			return
		}
		body, status := handler(req)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		if err := json.NewEncoder(w).Encode(body); err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
}

var ollamaFastRetry = RetryConfig{MaxAttempts: 3, InitialBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond}

func TestNewOllamaEmbedder_Validation(t *testing.T) {
	_, err := NewOllamaEmbedder(OllamaConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Model")

	_, err = NewOllamaEmbedder(OllamaConfig{Model: ModelNomicEmbedText, Dimension: -1})
	require.Error(t, err)
	assert.Contains(t, err.Error(), ">= 0")
}

func TestNewOllamaEmbedder_KnownDimensions(t *testing.T) {
	tests := []struct {
		model string
		dim   int
	}{
		{ModelAllMiniLML6V2, 384},
		{ModelNomicEmbedText, 768},
		{ModelBGESmallENV15, 384},
		{"some-custom-model", 0},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			e, err := NewOllamaEmbedder(OllamaConfig{Model: tt.model})
			require.NoError(t, err)
			assert.Equal(t, tt.dim, e.Dimension())
		})
	}
}

func TestOllamaEmbedder_Embed(t *testing.T) {
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		if req.Model != ModelNomicEmbedText {
			t.Errorf("unexpected model %q", req.Model)
		}
		return ollamaEmbedResponse{Model: req.Model, Embeddings: [][]float32{vecOf(768, 0.1)}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: ModelNomicEmbedText, BaseURL: srv.URL, Retry: ollamaFastRetry})
	require.NoError(t, err)
	assert.Equal(t, 768, e.Dimension(), "known model dimension should be resolved at construction")

	vec, err := e.Embed(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, vecOf(768, 0.1), vec)
}

func TestOllamaEmbedder_DimensionDetection(t *testing.T) {
	// Server returns 5-dimensional vectors for an unknown model.
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		emb := make([][]float32, len(req.Input))
		for i := range emb {
			emb[i] = vecOf(5, 0.5)
		}
		return ollamaEmbedResponse{Embeddings: emb}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: "my-fine-tuned-model", BaseURL: srv.URL, Retry: ollamaFastRetry})
	require.NoError(t, err)
	assert.Equal(t, 0, e.Dimension(), "unknown model should start with undetected dimension")

	dim, err := e.DetectDimension(context.Background(), "sample text")
	require.NoError(t, err)
	assert.Equal(t, 5, dim)
	assert.Equal(t, 5, e.Dimension(), "dimension should be resolved after detection")
}

func TestOllamaEmbedder_LazyDetectionWithoutDetectCall(t *testing.T) {
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		emb := make([][]float32, len(req.Input))
		for i := range emb {
			emb[i] = vecOf(4, 0.25)
		}
		return ollamaEmbedResponse{Embeddings: emb}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: "unknown-model", BaseURL: srv.URL, Retry: ollamaFastRetry})
	require.NoError(t, err)
	assert.Equal(t, 0, e.Dimension())

	_, err = e.Embed(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, 4, e.Dimension(), "dimension should be resolved by the first request")
}

func TestOllamaEmbedder_ExplicitDimensionMismatch(t *testing.T) {
	// Client expects 5 dims, server returns 4: the explicit configuration
	// must win and the mismatch must be reported.
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		emb := make([][]float32, len(req.Input))
		for i := range emb {
			emb[i] = vecOf(4, 0.25)
		}
		return ollamaEmbedResponse{Embeddings: emb}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: "unknown-model", Dimension: 5, BaseURL: srv.URL, Retry: ollamaFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hi")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 5")
	assert.Equal(t, 5, e.Dimension(), "explicit dimension must not be overwritten")
}

func TestOllamaEmbedder_Batching(t *testing.T) {
	var mu sync.Mutex
	var sizes []int
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		mu.Lock()
		sizes = append(sizes, len(req.Input))
		mu.Unlock()
		emb := make([][]float32, len(req.Input))
		for i := range emb {
			emb[i] = vecOf(2, 1)
		}
		return ollamaEmbedResponse{Embeddings: emb}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: ModelAllMiniLML6V2, Dimension: 2, BaseURL: srv.URL, BatchSize: 2, Retry: ollamaFastRetry})
	require.NoError(t, err)

	texts := []string{"a", "b", "c", "d", "e"}
	vecs, err := e.EmbedBatch(context.Background(), texts)
	require.NoError(t, err)
	require.Len(t, vecs, 5)

	mu.Lock()
	assert.Equal(t, []int{2, 2, 1}, sizes)
	mu.Unlock()
}

func TestOllamaEmbedder_ModelNotFound(t *testing.T) {
	var attempts int32
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		atomic.AddInt32(&attempts, 1)
		return map[string]string{"error": "model not found"}, http.StatusNotFound
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: "nope", BaseURL: srv.URL, Retry: ollamaFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hi")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "404")
	assert.Equal(t, int32(1), atomic.LoadInt32(&attempts), "404 must not be retried")
}

func TestOllamaEmbedder_RetryOn500(t *testing.T) {
	var attempts int32
	srv := newOllamaTestServer(t, func(req ollamaEmbedRequest) (interface{}, int) {
		if atomic.AddInt32(&attempts, 1) == 1 {
			return map[string]string{"error": "inference failed"}, http.StatusInternalServerError
		}
		return ollamaEmbedResponse{Embeddings: [][]float32{vecOf(2, 0.5)}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOllamaEmbedder(OllamaConfig{Model: ModelAllMiniLML6V2, Dimension: 2, BaseURL: srv.URL, Retry: ollamaFastRetry})
	require.NoError(t, err)

	vec, err := e.Embed(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, vecOf(2, 0.5), vec)
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts))
}
