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

// newCohereTestServer starts a fake Cohere /v1/embed endpoint.
func newCohereTestServer(t *testing.T, handler func(req cohereEmbedRequest) (interface{}, int)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embed" {
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("unexpected Authorization header %q", got)
		}
		if got := r.Header.Get("cohere-version"); got == "" {
			t.Errorf("missing cohere-version header")
		}
		var req cohereEmbedRequest
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

var cohereFastRetry = RetryConfig{MaxAttempts: 3, InitialBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond}

func TestNewCohereEmbedder_Validation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     CohereConfig
		wantErr string
	}{
		{"missing key", CohereConfig{Model: ModelEmbedEnglishV3}, "APIKey"},
		{"missing model", CohereConfig{APIKey: "k"}, "Model"},
		{"invalid input type", CohereConfig{APIKey: "k", Model: ModelEmbedEnglishV3, InputType: "bogus"}, "invalid InputType"},
		{"invalid truncation", CohereConfig{APIKey: "k", Model: ModelEmbedEnglishV3, Truncation: "MIDDLE"}, "invalid Truncation"},
		{"unknown model without dimension", CohereConfig{APIKey: "k", Model: "mystery"}, "set Dimension explicitly"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewCohereEmbedder(tt.cfg)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

func TestNewCohereEmbedder_Dimensions(t *testing.T) {
	e, err := NewCohereEmbedder(CohereConfig{APIKey: "k", Model: ModelEmbedEnglishV3})
	require.NoError(t, err)
	assert.Equal(t, 1024, e.Dimension())

	e, err = NewCohereEmbedder(CohereConfig{APIKey: "k", Model: ModelEmbedMultilingualV3})
	require.NoError(t, err)
	assert.Equal(t, 1024, e.Dimension())

	e, err = NewCohereEmbedder(CohereConfig{APIKey: "k", Model: "custom", Dimension: 128})
	require.NoError(t, err)
	assert.Equal(t, 128, e.Dimension())
}

func TestCohereEmbedder_Embed_RequestFields(t *testing.T) {
	var got cohereEmbedRequest
	srv := newCohereTestServer(t, func(req cohereEmbedRequest) (interface{}, int) {
		got = req
		return cohereEmbedResponse{Embeddings: [][]float32{vecOf(3, 0.25)}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewCohereEmbedder(CohereConfig{
		APIKey: "test-key", Model: ModelEmbedEnglishV3, Dimension: 3,
		InputType: InputTypeSearchQuery, Truncation: TruncationEnd,
		BaseURL: srv.URL, Retry: cohereFastRetry,
	})
	require.NoError(t, err)

	vec, err := e.Embed(context.Background(), "what is RAG?")
	require.NoError(t, err)
	assert.Equal(t, vecOf(3, 0.25), vec)
	assert.Equal(t, ModelEmbedEnglishV3, got.Model)
	assert.Equal(t, InputTypeSearchQuery, got.InputType)
	assert.Equal(t, TruncationEnd, got.Truncation)
	assert.Equal(t, []string{"what is RAG?"}, got.Texts)
}

func TestCohereEmbedder_DefaultInputType(t *testing.T) {
	var got cohereEmbedRequest
	srv := newCohereTestServer(t, func(req cohereEmbedRequest) (interface{}, int) {
		got = req
		return cohereEmbedResponse{Embeddings: [][]float32{vecOf(3, 0.25)}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewCohereEmbedder(CohereConfig{APIKey: "test-key", Model: ModelEmbedEnglishV3, Dimension: 3, BaseURL: srv.URL, Retry: cohereFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, InputTypeSearchDocument, got.InputType)
	assert.Empty(t, got.Truncation)
}

func TestCohereEmbedder_EmbedBatch_Batching(t *testing.T) {
	var mu sync.Mutex
	var sizes []int
	srv := newCohereTestServer(t, func(req cohereEmbedRequest) (interface{}, int) {
		mu.Lock()
		sizes = append(sizes, len(req.Texts))
		mu.Unlock()
		emb := make([][]float32, len(req.Texts))
		for i := range emb {
			emb[i] = vecOf(3, float32(i))
		}
		return cohereEmbedResponse{Embeddings: emb}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewCohereEmbedder(CohereConfig{APIKey: "test-key", Model: ModelEmbedEnglishV3, Dimension: 3, BaseURL: srv.URL, BatchSize: 4, Retry: cohereFastRetry})
	require.NoError(t, err)

	texts := make([]string, 10)
	for i := range texts {
		texts[i] = string(rune('a' + i))
	}
	vecs, err := e.EmbedBatch(context.Background(), texts)
	require.NoError(t, err)
	require.Len(t, vecs, 10)

	mu.Lock()
	assert.Equal(t, []int{4, 4, 2}, sizes)
	mu.Unlock()

	// First vector of the second batch (global index 4) must carry the
	// local index 0 from that batch.
	assert.Equal(t, vecOf(3, 0), vecs[4])
}

func TestCohereEmbedder_RetryOn500(t *testing.T) {
	var attempts int32
	srv := newCohereTestServer(t, func(req cohereEmbedRequest) (interface{}, int) {
		if atomic.AddInt32(&attempts, 1) == 1 {
			return map[string]string{"message": "server error"}, http.StatusInternalServerError
		}
		return cohereEmbedResponse{Embeddings: [][]float32{vecOf(3, 0.5)}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewCohereEmbedder(CohereConfig{APIKey: "test-key", Model: ModelEmbedEnglishV3, Dimension: 3, BaseURL: srv.URL, Retry: cohereFastRetry})
	require.NoError(t, err)

	vec, err := e.Embed(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, vecOf(3, 0.5), vec)
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts))
}

func TestCohereEmbedder_DimensionMismatch(t *testing.T) {
	srv := newCohereTestServer(t, func(req cohereEmbedRequest) (interface{}, int) {
		return cohereEmbedResponse{Embeddings: [][]float32{vecOf(2, 0.5)}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewCohereEmbedder(CohereConfig{APIKey: "test-key", Model: ModelEmbedEnglishV3, Dimension: 3, BaseURL: srv.URL, Retry: cohereFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dimensions")
}

func TestCohereEmbedder_CountMismatch(t *testing.T) {
	srv := newCohereTestServer(t, func(req cohereEmbedRequest) (interface{}, int) {
		return cohereEmbedResponse{Embeddings: [][]float32{}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewCohereEmbedder(CohereConfig{APIKey: "test-key", Model: ModelEmbedEnglishV3, BaseURL: srv.URL, Retry: cohereFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 1 embeddings, got 0")
}
