package embedder

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var fastRetry = RetryConfig{MaxAttempts: 3, InitialBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond}

func TestDefaultRetryConfig(t *testing.T) {
	cfg := DefaultRetryConfig()
	assert.Equal(t, 3, cfg.MaxAttempts)
	assert.Equal(t, 500*time.Millisecond, cfg.InitialBackoff)
	assert.Equal(t, 8*time.Second, cfg.MaxBackoff)
}

func TestRetryConfig_Normalize(t *testing.T) {
	var zero RetryConfig
	zero.normalize()
	assert.Equal(t, 3, zero.MaxAttempts)
	assert.Equal(t, 500*time.Millisecond, zero.InitialBackoff)
	assert.Equal(t, 8*time.Second, zero.MaxBackoff)

	rc := RetryConfig{MaxAttempts: 5, InitialBackoff: 10 * time.Second, MaxBackoff: time.Second}
	rc.normalize()
	assert.Equal(t, 5, rc.MaxAttempts)
	assert.GreaterOrEqual(t, rc.MaxBackoff, rc.InitialBackoff, "MaxBackoff must be raised to InitialBackoff")
}

func TestIsRetryableStatus(t *testing.T) {
	tests := []struct {
		status int
		want   bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusUnauthorized, false},
		{http.StatusNotFound, false},
		{http.StatusTooManyRequests, true},
		{http.StatusInternalServerError, true},
		{http.StatusBadGateway, true},
		{http.StatusServiceUnavailable, true},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("status_%d", tt.status), func(t *testing.T) {
			assert.Equal(t, tt.want, isRetryableStatus(tt.status))
		})
	}
}

func TestRetry_FirstAttemptSuccess(t *testing.T) {
	var attempts int32
	err := retry(context.Background(), fastRetry, func() error {
		atomic.AddInt32(&attempts, 1)
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(&attempts))
}

func TestRetry_RetryableAPIError(t *testing.T) {
	var attempts int32
	err := retry(context.Background(), fastRetry, func() error {
		if atomic.AddInt32(&attempts, 1) <= 2 {
			return &apiError{status: http.StatusTooManyRequests, message: "slow down"}
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
}

func TestRetry_NonRetryableAPIError(t *testing.T) {
	wantErr := &apiError{status: http.StatusBadRequest, message: "bad request"}
	var attempts int32
	err := retry(context.Background(), fastRetry, func() error {
		atomic.AddInt32(&attempts, 1)
		return wantErr
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "status 400")
	assert.Equal(t, int32(1), atomic.LoadInt32(&attempts), "4xx errors must not be retried")
}

func TestRetry_WrappedAPIError(t *testing.T) {
	var attempts int32
	err := retry(context.Background(), fastRetry, func() error {
		if atomic.AddInt32(&attempts, 1) == 1 {
			return fmt.Errorf("outer wrapper: %w", &apiError{status: http.StatusServiceUnavailable, message: "unavailable"})
		}
		return nil
	})
	require.NoError(t, err, "wrapped apiError should be detected and retried")
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts))
}

func TestRetry_NetworkErrorRetried(t *testing.T) {
	transportErr := errors.New("connection refused")
	var attempts int32
	err := retry(context.Background(), fastRetry, func() error {
		if atomic.AddInt32(&attempts, 1) == 1 {
			return fmt.Errorf("request failed: %w", transportErr)
		}
		return nil
	})
	require.NoError(t, err, "transport errors should be retried")
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts))
}

func TestRetry_Exhaustion(t *testing.T) {
	var attempts int32
	err := retry(context.Background(), fastRetry, func() error {
		atomic.AddInt32(&attempts, 1)
		return &apiError{status: http.StatusServiceUnavailable, message: "down"}
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "status 503")
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
}

func TestRetry_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cfg := RetryConfig{MaxAttempts: 5, InitialBackoff: 10 * time.Millisecond, MaxBackoff: 20 * time.Millisecond}
	var attempts int32

	go func() {
		time.Sleep(time.Millisecond)
		cancel()
	}()

	err := retry(ctx, cfg, func() error {
		atomic.AddInt32(&attempts, 1)
		return &apiError{status: http.StatusTooManyRequests, message: "limited"}
	})
	require.ErrorIs(t, err, context.Canceled)
	assert.Less(t, atomic.LoadInt32(&attempts), int32(5), "cancellation must stop the retry loop early")
}

func TestAPIError_Error(t *testing.T) {
	assert.Equal(t, "embedding API error (status 429): slow down", (&apiError{status: 429, message: "slow down"}).Error())
	assert.Equal(t, "embedding API error: malformed body", (&apiError{message: "malformed body"}).Error())
}
