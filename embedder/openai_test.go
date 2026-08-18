package embedder

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// vecOf returns a vector of length n filled with v.
func vecOf(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// newOpenAITestServer starts a fake OpenAI embeddings endpoint. The handler
// receives the decoded request and returns a JSON-serializable response
// body and HTTP status.
func newOpenAITestServer(t *testing.T, handler func(req openaiEmbedRequest) (interface{}, int)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("unexpected Authorization header %q", got)
			http.Error(w, "bad auth", http.StatusUnauthorized)
			return
		}
		var req openaiEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed to decode request: %v", err)
			http.Error(w, "bad json", http.StatusBadRequest)
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

// openAIFastRetry is a RetryConfig sized for fast tests.
var openAIFastRetry = RetryConfig{MaxAttempts: 3, InitialBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond}

func TestNewOpenAIEmbedder_Validation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     OpenAIConfig
		wantErr string
	}{
		{"missing key", OpenAIConfig{Model: ModelTextEmbedding3Small}, "APIKey"},
		{"missing model", OpenAIConfig{APIKey: "k"}, "Model"},
		{"unknown model without dimension", OpenAIConfig{APIKey: "k", Model: "mystery-model"}, "set Dimension explicitly"},
		{"dimension exceeds native", OpenAIConfig{APIKey: "k", Model: ModelTextEmbeddingAda002, Dimension: 2048}, "exceeds"},
		{"negative dimension", OpenAIConfig{APIKey: "k", Model: ModelTextEmbedding3Small, Dimension: -1}, ">= 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewOpenAIEmbedder(tt.cfg)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

func TestNewOpenAIEmbedder_Dimensions(t *testing.T) {
	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "k", Model: ModelTextEmbedding3Small})
	require.NoError(t, err)
	assert.Equal(t, 1536, e.Dimension())

	e, err = NewOpenAIEmbedder(OpenAIConfig{APIKey: "k", Model: ModelTextEmbedding3Large})
	require.NoError(t, err)
	assert.Equal(t, 3072, e.Dimension())

	// Matryoshka truncation to a smaller dimension.
	e, err = NewOpenAIEmbedder(OpenAIConfig{APIKey: "k", Model: ModelTextEmbedding3Small, Dimension: 256})
	require.NoError(t, err)
	assert.Equal(t, 256, e.Dimension())

	// Unknown model with an explicit dimension.
	e, err = NewOpenAIEmbedder(OpenAIConfig{APIKey: "k", Model: "custom-model", Dimension: 64})
	require.NoError(t, err)
	assert.Equal(t, 64, e.Dimension())
}

func TestOpenAIEmbedder_Embed(t *testing.T) {
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		if req.Model != ModelTextEmbedding3Small || req.Dimension == nil || *req.Dimension != 3 {
			t.Errorf("unexpected request: model=%q dimension=%v", req.Model, req.Dimension)
		}
		return openaiEmbedResponse{Data: []openaiEmbedItem{{Object: "embedding", Index: 0, Embedding: vecOf(3, 0.5)}}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	vec, err := e.Embed(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, vecOf(3, 0.5), vec)
}

func TestOpenAIEmbedder_EmbedBatch_OrderingAndBatching(t *testing.T) {
	var mu sync.Mutex
	var sizes []int
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		mu.Lock()
		sizes = append(sizes, len(req.Input))
		mu.Unlock()
		items := make([]openaiEmbedItem, len(req.Input))
		for i, text := range req.Input {
			sum := 0
			for _, c := range text {
				sum += int(c)
			}
			items[i] = openaiEmbedItem{Object: "embedding", Index: i, Embedding: []float32{float32(sum), 0, 0}}
		}
		return openaiEmbedResponse{Data: items}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, BatchSize: 100, Retry: openAIFastRetry})
	require.NoError(t, err)

	texts := make([]string, 250)
	for i := range texts {
		texts[i] = fmt.Sprintf("doc-%d", i)
	}
	vecs, err := e.EmbedBatch(context.Background(), texts)
	require.NoError(t, err)
	require.Len(t, vecs, 250)

	mu.Lock()
	assert.Equal(t, []int{100, 100, 50}, sizes)
	mu.Unlock()

	// Ordering must be preserved across batch boundaries.
	for _, i := range []int{0, 99, 100, 249} {
		sum := 0
		for _, c := range texts[i] {
			sum += int(c)
		}
		assert.Equal(t, float32(sum), vecs[i][0], "ordering broken at index %d", i)
	}
}

func TestOpenAIEmbedder_EmbedBatch_Empty(t *testing.T) {
	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small})
	require.NoError(t, err)
	vecs, err := e.EmbedBatch(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, vecs)
}

func TestOpenAIEmbedder_RetryOn429(t *testing.T) {
	var attempts int32
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		if atomic.AddInt32(&attempts, 1) <= 2 {
			return map[string]string{"error": "rate limited"}, http.StatusTooManyRequests
		}
		return openaiEmbedResponse{Data: []openaiEmbedItem{{Index: 0, Embedding: vecOf(3, 0.1)}}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	vec, err := e.Embed(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, vecOf(3, 0.1), vec)
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
}

func TestOpenAIEmbedder_RetryExhausted(t *testing.T) {
	var attempts int32
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		atomic.AddInt32(&attempts, 1)
		return map[string]string{"error": "rate limited"}, http.StatusTooManyRequests
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "429")
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
}

func TestOpenAIEmbedder_NoRetryOn401(t *testing.T) {
	var attempts int32
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		atomic.AddInt32(&attempts, 1)
		return map[string]string{"error": "invalid api key"}, http.StatusUnauthorized
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "401")
	assert.Equal(t, int32(1), atomic.LoadInt32(&attempts))
}

func TestOpenAIEmbedder_DimensionMismatch(t *testing.T) {
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		return openaiEmbedResponse{Data: []openaiEmbedItem{{Index: 0, Embedding: vecOf(2, 0.5)}}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dimensions")
}

func TestOpenAIEmbedder_OutOfOrderResponse(t *testing.T) {
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		// Deliberately reversed: index 1 first.
		return openaiEmbedResponse{Data: []openaiEmbedItem{
			{Index: 1, Embedding: vecOf(3, 0.2)},
			{Index: 0, Embedding: vecOf(3, 0.1)},
		}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	_, err = e.EmbedBatch(context.Background(), []string{"a", "b"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out-of-order")
}

func TestOpenAIEmbedder_CountMismatch(t *testing.T) {
	srv := newOpenAITestServer(t, func(req openaiEmbedRequest) (interface{}, int) {
		return openaiEmbedResponse{Data: []openaiEmbedItem{{Index: 0, Embedding: vecOf(3, 0.1)}}}, http.StatusOK
	})
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, Dimension: 3, BaseURL: srv.URL, Retry: openAIFastRetry})
	require.NoError(t, err)

	_, err = e.EmbedBatch(context.Background(), []string{"a", "b"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 2 embeddings, got 1")
}

func TestOpenAIEmbedder_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "{not-json")
	}))
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, BaseURL: srv.URL, Retry: RetryConfig{MaxAttempts: 2, InitialBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond}})
	require.NoError(t, err)

	_, err = e.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid response JSON")
}

func TestOpenAIEmbedder_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()

	e, err := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key", Model: ModelTextEmbedding3Small, BaseURL: srv.URL, Retry: RetryConfig{MaxAttempts: 1}})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = e.EmbedBatch(ctx, []string{"hello"})
	require.ErrorIs(t, err, context.Canceled)
}
