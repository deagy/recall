package llm

import (
	"context"
	"sync"
	"time"
)

// RateLimitConfig controls RateLimitBackend behavior.
type RateLimitConfig struct {
	// Capacity is the maximum number of tokens in the bucket (burst size).
	// Zero uses the default (10).
	Capacity int

	// RefillRate is the number of tokens added per second. Zero uses the
	// default (10/s).
	RefillRate float64
}

// DefaultRateLimitConfig returns sensible defaults.
func DefaultRateLimitConfig() RateLimitConfig {
	return RateLimitConfig{Capacity: 10, RefillRate: 10}
}

func (c *RateLimitConfig) normalize() {
	def := DefaultRateLimitConfig()
	if c.Capacity <= 0 {
		c.Capacity = def.Capacity
	}
	if c.RefillRate <= 0 {
		c.RefillRate = def.RefillRate
	}
}

// RateLimitBackend wraps a Backend with a token-bucket rate limiter, ensuring
// calls are spaced out to stay within a sustainable request rate.
type RateLimitBackend struct {
	inner Backend
	cap   int
	rate  float64 // tokens per second

	now   func() time.Time
	sleep func(ctx context.Context, d time.Duration) error

	mu     sync.Mutex
	tokens float64
	last   time.Time
}

var _ Backend = (*RateLimitBackend)(nil)

// NewRateLimitBackend wraps inner with token-bucket rate limiting.
func NewRateLimitBackend(inner Backend, cfg RateLimitConfig) *RateLimitBackend {
	if inner == nil {
		panic("llm: NewRateLimitBackend requires a non-nil backend")
	}
	cfg.normalize()
	return &RateLimitBackend{
		inner:  inner,
		cap:    cfg.Capacity,
		rate:   cfg.RefillRate,
		now:    time.Now,
		sleep:  defaultSleep,
		tokens: float64(cfg.Capacity),
		last:   time.Now(),
	}
}

// Backend returns the wrapped backend.
func (b *RateLimitBackend) Backend() Backend { return b.inner }

// Chat acquires a token (blocking up to context deadline) then calls inner.
func (b *RateLimitBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	if err := b.acquire(ctx); err != nil {
		return nil, err
	}
	return b.inner.Chat(ctx, req)
}

// ChatStream acquires a token (blocking up to context deadline) then calls inner.
func (b *RateLimitBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	if err := b.acquire(ctx); err != nil {
		return err
	}
	return b.inner.ChatStream(ctx, req, fn)
}

// acquire waits until a token is available, refilling the bucket over time.
func (b *RateLimitBackend) acquire(ctx context.Context) error {
	for {
		b.mu.Lock()
		b.refillLocked()
		if b.tokens >= 1 {
			b.tokens--
			b.mu.Unlock()
			return nil
		}
		wait := time.Duration((1 - b.tokens) / b.rate * float64(time.Second))
		b.mu.Unlock()
		if wait <= 0 {
			continue
		}
		if err := b.sleep(ctx, wait); err != nil {
			return err
		}
	}
}

// refillLocked adds tokens based on elapsed time. Caller must hold b.mu.
func (b *RateLimitBackend) refillLocked() {
	now := b.now()
	if elapsed := now.Sub(b.last).Seconds(); elapsed > 0 {
		b.tokens += elapsed * b.rate
		if b.tokens > float64(b.cap) {
			b.tokens = float64(b.cap)
		}
		b.last = now
	}
}

// RateLimitMiddleware returns a MiddlewareFunc that wraps a backend in a
// RateLimitBackend with the given config.
func RateLimitMiddleware(cfg RateLimitConfig) MiddlewareFunc {
	return func(b Backend) Backend { return NewRateLimitBackend(b, cfg) }
}
