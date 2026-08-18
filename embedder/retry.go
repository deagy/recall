package embedder

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"time"
)

// RetryConfig controls retry behavior for embedder HTTP requests.
type RetryConfig struct {
	// MaxAttempts is the total number of attempts, including the first.
	// Zero uses the default (3).
	MaxAttempts int

	// InitialBackoff is the base delay before the first retry.
	// Zero uses the default (500ms).
	InitialBackoff time.Duration

	// MaxBackoff caps the exponential backoff delay.
	// Zero uses the default (8s).
	MaxBackoff time.Duration
}

// DefaultRetryConfig returns sensible retry defaults.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    3,
		InitialBackoff: 500 * time.Millisecond,
		MaxBackoff:     8 * time.Second,
	}
}

// normalize fills zero fields with defaults.
func (rc *RetryConfig) normalize() {
	def := DefaultRetryConfig()
	if rc.MaxAttempts <= 0 {
		rc.MaxAttempts = def.MaxAttempts
	}
	if rc.InitialBackoff <= 0 {
		rc.InitialBackoff = def.InitialBackoff
	}
	if rc.MaxBackoff <= 0 {
		rc.MaxBackoff = def.MaxBackoff
	}
	if rc.InitialBackoff > rc.MaxBackoff {
		rc.MaxBackoff = rc.InitialBackoff
	}
}

// apiError represents a failed interaction with an embedding API.
// A zero status denotes a client-side failure (e.g. malformed response or
// validation error) which is never retried.
type apiError struct {
	status  int
	message string
}

func (e *apiError) Error() string {
	if e.status == 0 {
		return fmt.Sprintf("embedding API error: %s", e.message)
	}
	return fmt.Sprintf("embedding API error (status %d): %s", e.status, e.message)
}

// isRetryable reports whether the failure warrants a retry: rate limits
// (429) and server errors (5xx) do; other 4xx responses do not.
func (e *apiError) isRetryable() bool {
	return isRetryableStatus(e.status)
}

// isRetryableStatus reports whether an HTTP status code warrants a retry.
func isRetryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= http.StatusInternalServerError
}

// retry calls attempt up to rc.MaxAttempts times, sleeping with exponential
// backoff plus jitter between retryable failures.
//
// Retryability rules: an *apiError is retried only when its status is
// retryable; any other error (transport failures, timeouts) is treated as a
// transient network problem and retried. Context cancellation is always
// honored immediately.
func retry(ctx context.Context, rc RetryConfig, attempt func() error) error {
	rc.normalize()

	var err error
	for i := 0; i < rc.MaxAttempts; i++ {
		if err = attempt(); err == nil {
			return nil
		}

		var apiErr *apiError
		if asAPIError(err, &apiErr) {
			if !apiErr.isRetryable() {
				return err
			}
		}
		if i == rc.MaxAttempts-1 {
			break
		}

		backoff := rc.InitialBackoff << i
		if backoff > rc.MaxBackoff || backoff <= 0 {
			backoff = rc.MaxBackoff
		}
		// Full jitter in [backoff/2, backoff] keeps retries spread out
		// while bounding the worst-case delay.
		backoff = backoff/2 + time.Duration(rand.Int63n(int64(backoff/2)+1))

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
	}
	return err
}

// asAPIError reports whether err is (or wraps) an *apiError, and if so
// assigns it to target.
func asAPIError(err error, target **apiError) bool {
	for err != nil {
		if apiErr, ok := err.(*apiError); ok {
			*target = apiErr
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}
