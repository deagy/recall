package llm

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// This file implements composable resilience decorators for llm.Backend:
// retry with backoff, fallback across providers, and a Middleware composer
// that chains any number of decorators (including the circuit breaker and
// rate limiter defined in their own files).

// defaultSleep is the production, context-aware sleep used by decorators that
// back off. Tests replace it with an instant no-op.
func defaultSleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// isContextErr reports whether err is a context cancellation/deadline error,
// which should never be retried.
func isContextErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// RetryConfig controls RetryBackend behavior.
type RetryConfig struct {
	// MaxAttempts is the total number of attempts, including the first.
	// Zero uses the default (3).
	MaxAttempts int

	// InitialBackoff is the base delay before the first retry. Zero uses the
	// default (500ms).
	InitialBackoff time.Duration

	// MaxBackoff caps the exponential backoff delay. Zero uses the default
	// (8s).
	MaxBackoff time.Duration

	// Retryable optionally decides which errors are retried. When nil, every
	// error except context cancellation/deadline is retried.
	Retryable func(error) bool
}

// DefaultRetryConfig returns sensible retry defaults for LLM calls.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    3,
		InitialBackoff: 500 * time.Millisecond,
		MaxBackoff:     8 * time.Second,
	}
}

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

// shouldRetry reports whether err is worth another attempt.
func (rc *RetryConfig) shouldRetry(err error) bool {
	if rc.Retryable != nil {
		return rc.Retryable(err)
	}
	return !isContextErr(err)
}

// RetryBackend wraps a Backend and retries failed calls with exponential
// backoff and jitter, honoring context cancellation.
type RetryBackend struct {
	inner Backend
	cfg   RetryConfig
	sleep func(ctx context.Context, d time.Duration) error
}

var _ Backend = (*RetryBackend)(nil)

// NewRetryBackend wraps inner with retry behavior from cfg. Zero-valued cfg
// fields fall back to DefaultRetryConfig.
func NewRetryBackend(inner Backend, cfg RetryConfig) *RetryBackend {
	if inner == nil {
		panic("llm: NewRetryBackend requires a non-nil backend")
	}
	cfg.normalize()
	return &RetryBackend{inner: inner, cfg: cfg, sleep: defaultSleep}
}

// Backend returns the wrapped backend, enabling unwrapping by decorators.
func (b *RetryBackend) Backend() Backend { return b.inner }

// Chat sends a request, retrying on retryable errors.
func (b *RetryBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	var (
		resp *ChatResponse
		err  error
	)
	for i := 0; i < b.cfg.MaxAttempts; i++ {
		if resp, err = b.inner.Chat(ctx, req); err == nil {
			return resp, nil
		}
		if !b.cfg.shouldRetry(err) || i == b.cfg.MaxAttempts-1 {
			break
		}
		if serr := b.sleep(ctx, b.backoff(i)); serr != nil {
			return nil, err // context done while waiting; return the original error
		}
	}
	return nil, err
}

// ChatStream sends a streaming request, retrying on retryable errors. A
// stream is retried only if it fails before delivering a single chunk; once
// output has started the error is not retried (the consumer may have already
// consumed partial content).
func (b *RetryBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	var err error
	for i := 0; i < b.cfg.MaxAttempts; i++ {
		delivered := 0
		err = b.inner.ChatStream(ctx, req, func(chunk *StreamChunk) error {
			delivered++
			return fn(chunk)
		})
		if err == nil {
			return nil
		}
		if delivered > 0 || !b.cfg.shouldRetry(err) || i == b.cfg.MaxAttempts-1 {
			break
		}
		if serr := b.sleep(ctx, b.backoff(i)); serr != nil {
			break
		}
	}
	return err
}

// backoff returns the jittered delay before retry i (0-based).
func (b *RetryBackend) backoff(i int) time.Duration {
	d := b.cfg.InitialBackoff << i
	if d > b.cfg.MaxBackoff || d <= 0 {
		d = b.cfg.MaxBackoff
	}
	return d/2 + time.Duration(rand.Int63n(int64(d/2)+1))
}

// FallbackBackend wraps an ordered list of backends (primary first) and
// returns the first successful result, enabling failover across providers.
type FallbackBackend struct {
	backends []Backend
}

var _ Backend = (*FallbackBackend)(nil)

// NewFallbackBackend creates a FallbackBackend from a primary backend and
// zero or more fallbacks, tried in order. Nil fallbacks are dropped.
func NewFallbackBackend(primary Backend, fallbacks ...Backend) *FallbackBackend {
	if primary == nil {
		panic("llm: NewFallbackBackend requires a non-nil primary backend")
	}
	list := make([]Backend, 0, len(fallbacks)+1)
	list = append(list, primary)
	for _, f := range fallbacks {
		if f != nil {
			list = append(list, f)
		}
	}
	return &FallbackBackend{backends: list}
}

// Backends returns the configured backends in failover order.
func (b *FallbackBackend) Backends() []Backend { return b.backends }

// Chat tries each backend in order until one succeeds.
func (b *FallbackBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	var lastErr error
	for _, be := range b.backends {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		resp, err := be.Chat(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("llm: all %d fallback backends failed; last error: %w", len(b.backends), lastErr)
}

// ChatStream tries each backend in order until one succeeds.
func (b *FallbackBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	var lastErr error
	for _, be := range b.backends {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err := be.ChatStream(ctx, req, fn); err == nil {
			return nil
		} else {
			lastErr = err
		}
	}
	return fmt.Errorf("llm: all %d fallback backends failed; last error: %w", len(b.backends), lastErr)
}

// MiddlewareFunc decorates a backend, returning a new backend that wraps the
// given one. Compose multiple decorators to build a resilient backend chain.
type MiddlewareFunc func(Backend) Backend

// Middleware composes MiddlewareFuncs around a core backend.
//
// The functions are applied so that fns[0] becomes the outermost layer (the
// first thing a caller hits) and the last fn wraps the core directly. For
// example, NewMiddleware(core, RetryMiddleware(...), RateLimitMiddleware(...)).Build()
// yields retry(rateLimit(core)): every call is retried, and each individual
// attempt is rate-limited.
type Middleware struct {
	core Backend
	fns  []MiddlewareFunc
}

// NewMiddleware creates a Middleware around core with the given decorators.
func NewMiddleware(core Backend, fns ...MiddlewareFunc) *Middleware {
	if core == nil {
		panic("llm: NewMiddleware requires a non-nil core backend")
	}
	return &Middleware{core: core, fns: fns}
}

// Build applies all decorators to the core backend and returns the composed
// result.
func (m *Middleware) Build() Backend {
	b := m.core
	for i := len(m.fns) - 1; i >= 0; i-- {
		b = m.fns[i](b)
	}
	return b
}

// RetryMiddleware returns a MiddlewareFunc that wraps a backend in a
// RetryBackend with the given config.
func RetryMiddleware(cfg RetryConfig) MiddlewareFunc {
	return func(b Backend) Backend { return NewRetryBackend(b, cfg) }
}

// FallbackMiddleware returns a MiddlewareFunc that wraps a backend with the
// given fallbacks.
func FallbackMiddleware(fallbacks ...Backend) MiddlewareFunc {
	return func(b Backend) Backend { return NewFallbackBackend(b, fallbacks...) }
}
