package llm

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrCircuitOpen is returned when a call is rejected because the circuit
// breaker is open.
var ErrCircuitOpen = errors.New("llm: circuit breaker is open")

// CircuitState describes the circuit breaker's current state.
type CircuitState int

const (
	// CircuitClosed allows all calls through.
	CircuitClosed CircuitState = iota
	// CircuitOpen rejects calls until the cooldown elapses.
	CircuitOpen
	// CircuitHalfOpen allows a single probe call to test recovery.
	CircuitHalfOpen
)

// String returns a human-readable state name.
func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "closed"
	case CircuitOpen:
		return "open"
	case CircuitHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// CircuitBreakerConfig controls CircuitBreakerBackend behavior.
type CircuitBreakerConfig struct {
	// Threshold is the number of consecutive failures that opens the circuit.
	// Zero uses the default (5).
	Threshold int

	// OpenDuration is how long the circuit stays open before allowing a
	// half-open probe. Zero uses the default (30s).
	OpenDuration time.Duration
}

// DefaultCircuitBreakerConfig returns sensible defaults.
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		Threshold:    5,
		OpenDuration: 30 * time.Second,
	}
}

func (c *CircuitBreakerConfig) normalize() {
	def := DefaultCircuitBreakerConfig()
	if c.Threshold <= 0 {
		c.Threshold = def.Threshold
	}
	if c.OpenDuration <= 0 {
		c.OpenDuration = def.OpenDuration
	}
}

// CircuitBreakerBackend wraps a Backend and stops calling it after a run of
// failures (opening the circuit), returning ErrCircuitOpen until a cooldown
// elapses, at which point a single probe is allowed to test recovery.
type CircuitBreakerBackend struct {
	inner Backend
	cfg   CircuitBreakerConfig
	now   func() time.Time

	mu       sync.Mutex
	state    CircuitState
	failures int
	openedAt time.Time
	probing  bool
}

var _ Backend = (*CircuitBreakerBackend)(nil)

// NewCircuitBreakerBackend wraps inner with circuit-breaker behavior.
func NewCircuitBreakerBackend(inner Backend, cfg CircuitBreakerConfig) *CircuitBreakerBackend {
	if inner == nil {
		panic("llm: NewCircuitBreakerBackend requires a non-nil backend")
	}
	cfg.normalize()
	return &CircuitBreakerBackend{inner: inner, cfg: cfg, now: time.Now, state: CircuitClosed}
}

// Backend returns the wrapped backend.
func (b *CircuitBreakerBackend) Backend() Backend { return b.inner }

// State returns the current circuit state (for observability/testing).
func (b *CircuitBreakerBackend) State() CircuitState {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.currentStateLocked()
}

// currentStateLocked returns the effective state, promoting an open circuit to
// half-open when its cooldown has elapsed. Caller must hold b.mu.
func (b *CircuitBreakerBackend) currentStateLocked() CircuitState {
	if b.state == CircuitOpen && b.now().Sub(b.openedAt) >= b.cfg.OpenDuration {
		b.state = CircuitHalfOpen
	}
	return b.state
}

// Chat routes the request through the circuit breaker.
func (b *CircuitBreakerBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	allow, err := b.before()
	if !allow {
		return nil, err
	}
	resp, err := b.inner.Chat(ctx, req)
	b.after(err)
	return resp, err
}

// ChatStream routes the request through the circuit breaker.
func (b *CircuitBreakerBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	allow, err := b.before()
	if !allow {
		return err
	}
	err = b.inner.ChatStream(ctx, req, fn)
	b.after(err)
	return err
}

// before decides whether a call may proceed, returning an error when the
// circuit is open.
func (b *CircuitBreakerBackend) before() (bool, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.currentStateLocked() {
	case CircuitClosed:
		return true, nil
	case CircuitOpen:
		return false, ErrCircuitOpen
	case CircuitHalfOpen:
		if b.probing {
			// Only one probe may be in flight at a time.
			return false, ErrCircuitOpen
		}
		b.probing = true
		return true, nil
	default:
		return false, ErrCircuitOpen
	}
}

// after records the outcome of a call.
func (b *CircuitBreakerBackend) after(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case CircuitHalfOpen:
		b.probing = false
		if err == nil {
			b.state = CircuitClosed
			b.failures = 0
		} else {
			b.state = CircuitOpen
			b.openedAt = b.now()
		}
	case CircuitClosed:
		if err == nil {
			b.failures = 0
		} else {
			b.failures++
			if b.failures >= b.cfg.Threshold {
				b.state = CircuitOpen
				b.openedAt = b.now()
			}
		}
	}
}

// CircuitBreakerMiddleware returns a MiddlewareFunc that wraps a backend in a
// CircuitBreakerBackend with the given config.
func CircuitBreakerMiddleware(cfg CircuitBreakerConfig) MiddlewareFunc {
	return func(b Backend) Backend { return NewCircuitBreakerBackend(b, cfg) }
}
