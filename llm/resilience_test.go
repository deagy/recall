package llm

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// scriptedBackend is a controllable Backend whose per-call outcomes are driven
// by a script: entries are either an error to return or a content string to
// return in a successful response. When the script is exhausted it succeeds.
type scriptedBackend struct {
	mu     sync.Mutex
	n      int
	script []any
}

// newScripted builds a scriptedBackend from outcome specs: an error value
// fails that call; a string succeeds with that content.
func newScripted(script ...any) *scriptedBackend {
	return &scriptedBackend{script: script}
}

// calls returns the number of Chat/ChatStream invocations so far.
func (s *scriptedBackend) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.n
}

func (s *scriptedBackend) next() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.n
	s.n++
	if i < len(s.script) {
		switch v := s.script[i].(type) {
		case error:
			return v
		case string:
			return nil
		}
	}
	return nil
}

func (s *scriptedBackend) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	if err := s.next(); err != nil {
		return nil, err
	}
	return &ChatResponse{Message: Message{Role: "assistant", Content: "ok"}}, nil
}

func (s *scriptedBackend) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	if err := s.next(); err != nil {
		return err
	}
	return fn(&StreamChunk{Delta: Message{Content: "hi "}})
}

var testNoSleep = func(ctx context.Context, d time.Duration) error { return nil }

func TestRetryBackend_SucceedsFirstTry(t *testing.T) {
	inner := newScripted("ok")
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 3})
	rb.sleep = testNoSleep
	resp, err := rb.Chat(context.Background(), &ChatRequest{})
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Fatalf("unexpected content %q", resp.Message.Content)
	}
	if c := inner.calls(); c != 1 {
		t.Fatalf("expected 1 call, got %d", c)
	}
}

func TestRetryBackend_SucceedsAfterFailures(t *testing.T) {
	inner := newScripted(errors.New("boom"), errors.New("boom"), "ok")
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 5})
	rb.sleep = testNoSleep
	if _, err := rb.Chat(context.Background(), &ChatRequest{}); err != nil {
		t.Fatalf("expected eventual success, got %v", err)
	}
	if c := inner.calls(); c != 3 {
		t.Fatalf("expected 3 calls, got %d", c)
	}
}

func TestRetryBackend_ExhaustsAttempts(t *testing.T) {
	inner := newScripted(errors.New("e1"), errors.New("e2"), errors.New("e3"))
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 3})
	rb.sleep = testNoSleep
	_, err := rb.Chat(context.Background(), &ChatRequest{})
	if err == nil {
		t.Fatal("expected error after exhausting attempts")
	}
	if c := inner.calls(); c != 3 {
		t.Fatalf("expected 3 calls, got %d", c)
	}
}

func TestRetryBackend_NonRetryableError(t *testing.T) {
	inner := newScripted(errors.New("bad request"), errors.New("bad request"))
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 5, Retryable: func(error) bool { return false }})
	rb.sleep = testNoSleep
	_, err := rb.Chat(context.Background(), &ChatRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if c := inner.calls(); c != 1 {
		t.Fatalf("expected 1 call (no retry), got %d", c)
	}
}

func TestRetryBackend_ContextErrorNotRetried(t *testing.T) {
	inner := newScripted(context.Canceled, context.Canceled)
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 5})
	rb.sleep = testNoSleep
	_, err := rb.Chat(context.Background(), &ChatRequest{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if c := inner.calls(); c != 1 {
		t.Fatalf("expected 1 call, got %d", c)
	}
}

func TestRetryBackend_DefaultsApplied(t *testing.T) {
	rb := NewRetryBackend(newScripted(), RetryConfig{})
	if rb.cfg.MaxAttempts != 3 || rb.cfg.InitialBackoff != 500*time.Millisecond || rb.cfg.MaxBackoff != 8*time.Second {
		t.Fatalf("defaults not applied: %+v", rb.cfg)
	}
}

func TestRetryBackend_BackoffGrowsAndCaps(t *testing.T) {
	rb := NewRetryBackend(newScripted(), RetryConfig{InitialBackoff: time.Second, MaxBackoff: 5 * time.Second})
	// backoff(i) is in [base<<i /2, base<<i] capped at MaxBackoff.
	if d := rb.backoff(0); d < time.Second/2 || d > time.Second {
		t.Fatalf("backoff(0)=%v out of range", d)
	}
	if d := rb.backoff(10); d > 5*time.Second {
		t.Fatalf("backoff(10)=%v exceeds cap", d)
	}
}

func TestRetryBackend_StreamRetriesBeforeDelivery(t *testing.T) {
	inner := newScripted(errors.New("s1"), errors.New("s2"), "ok")
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 5})
	rb.sleep = testNoSleep
	var got int
	if err := rb.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { got++; return nil }); err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if got != 1 {
		t.Fatalf("expected 1 delivered chunk, got %d", got)
	}
	if c := inner.calls(); c != 3 {
		t.Fatalf("expected 3 stream calls, got %d", c)
	}
}

func TestRetryBackend_StreamNoRetryAfterDelivery(t *testing.T) {
	// A stream that delivers one chunk then fails must not be retried.
	inner := &partialStream{failAfter: 1}
	rb := NewRetryBackend(inner, RetryConfig{MaxAttempts: 5})
	rb.sleep = testNoSleep
	var got int
	err := rb.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { got++; return nil })
	if err == nil {
		t.Fatal("expected error")
	}
	if c := inner.calls(); c != 1 {
		t.Fatalf("expected no retry (1 call), got %d", c)
	}
	if got != 1 {
		t.Fatalf("expected 1 chunk before failure, got %d", got)
	}
}

// partialStream delivers failAfter chunks then returns an error; it also
// records how many times ChatStream was invoked.
type partialStream struct {
	mu        sync.Mutex
	failAfter int
	n         int
}

func (p *partialStream) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	return nil, errors.New("partialStream: Chat not supported")
}

func (p *partialStream) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	p.mu.Lock()
	p.n++
	p.mu.Unlock()
	for i := 0; i < p.failAfter; i++ {
		if err := fn(&StreamChunk{Delta: Message{Content: "x "}}); err != nil {
			return err
		}
	}
	return errors.New("stream interrupted after delivery")
}

func (p *partialStream) calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.n
}

// countingFail is a Backend that always fails and counts invocations.
type countingFail struct {
	mu sync.Mutex
	n  int
}

func (c *countingFail) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	c.count()
	return nil, errors.New("always fail")
}

func (c *countingFail) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	c.count()
	return errors.New("always fail")
}

func (c *countingFail) count() {
	c.mu.Lock()
	c.n++
	c.mu.Unlock()
}

func (c *countingFail) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}

func TestCircuitBreaker_OpensAfterThreshold(t *testing.T) {
	inner := &countingFail{}
	cb := NewCircuitBreakerBackend(inner, CircuitBreakerConfig{Threshold: 3, OpenDuration: time.Minute})
	now := time.Now()
	cb.now = func() time.Time { return now }
	for i := 0; i < 3; i++ {
		if _, err := cb.Chat(context.Background(), &ChatRequest{}); err == nil {
			t.Fatalf("call %d: expected error", i)
		}
	}
	if s := cb.State(); s != CircuitOpen {
		t.Fatalf("expected open, got %s", s)
	}
	// Next call rejected without hitting inner.
	before := inner.calls()
	_, err := cb.Chat(context.Background(), &ChatRequest{})
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}
	if inner.calls() != before {
		t.Fatal("inner should not be called while circuit is open")
	}
}

func TestCircuitBreaker_HalfOpenProbeSucceeds(t *testing.T) {
	inner := newScripted() // fails until script says otherwise
	inner.script = []any{errors.New("f"), errors.New("f"), "ok"}
	cb := NewCircuitBreakerBackend(inner, CircuitBreakerConfig{Threshold: 2, OpenDuration: time.Second})
	now := time.Now()
	cb.now = func() time.Time { return now }
	// Trip the circuit (2 failures).
	for i := 0; i < 2; i++ {
		cb.Chat(context.Background(), &ChatRequest{})
	}
	if s := cb.State(); s != CircuitOpen {
		t.Fatalf("expected open, got %s", s)
	}
	// Advance past the cooldown.
	now = now.Add(2 * time.Second)
	// The half-open probe now succeeds (script index 2 = "ok").
	if _, err := cb.Chat(context.Background(), &ChatRequest{}); err != nil {
		t.Fatalf("expected probe to succeed, got %v", err)
	}
	if s := cb.State(); s != CircuitClosed {
		t.Fatalf("expected closed after successful probe, got %s", s)
	}
}

func TestCircuitBreaker_HalfOpenProbeFailsReopens(t *testing.T) {
	inner := &countingFail{}
	cb := NewCircuitBreakerBackend(inner, CircuitBreakerConfig{Threshold: 1, OpenDuration: time.Second})
	now := time.Now()
	cb.now = func() time.Time { return now }
	cb.Chat(context.Background(), &ChatRequest{}) // trips (threshold 1)
	now = now.Add(2 * time.Second)                // cooldown elapsed -> half-open
	if _, err := cb.Chat(context.Background(), &ChatRequest{}); err == nil {
		t.Fatal("expected probe to fail")
	}
	if s := cb.State(); s != CircuitOpen {
		t.Fatalf("expected open after failed probe, got %s", s)
	}
}

func TestCircuitBreaker_SuccessResetsFailures(t *testing.T) {
	inner := newScripted(errors.New("f"), "ok", errors.New("f"))
	cb := NewCircuitBreakerBackend(inner, CircuitBreakerConfig{Threshold: 2, OpenDuration: time.Minute})
	now := time.Now()
	cb.now = func() time.Time { return now }
	cb.Chat(context.Background(), &ChatRequest{}) // failure (1)
	if _, err := cb.Chat(context.Background(), &ChatRequest{}); err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	// A single further failure must not open (count reset by success).
	cb.Chat(context.Background(), &ChatRequest{})
	if s := cb.State(); s != CircuitClosed {
		t.Fatalf("expected closed, got %s", s)
	}
}

func TestCircuitBreaker_SingleConcurrentProbe(t *testing.T) {
	inner := &countingFail{}
	cb := NewCircuitBreakerBackend(inner, CircuitBreakerConfig{Threshold: 1, OpenDuration: time.Second})
	now := time.Now()
	cb.now = func() time.Time { return now }
	cb.Chat(context.Background(), &ChatRequest{}) // trips
	now = now.Add(2 * time.Second)                // half-open
	// Launch many goroutines; exactly one may be the probe (reach inner).
	const n = 16
	var wg sync.WaitGroup
	probes := make(chan struct{}, n)
	baseCalls := inner.calls()
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := cb.Chat(context.Background(), &ChatRequest{})
			// A probe reached the inner backend, which always returns a
			// non-circuit error; rejected calls get ErrCircuitOpen.
			if err != nil && !errors.Is(err, ErrCircuitOpen) {
				probes <- struct{}{}
			}
		}()
	}
	wg.Wait()
	if count := len(probes); count != 1 {
		t.Fatalf("expected exactly 1 probe, got %d", count)
	}
	if c := inner.calls() - baseCalls; c != 1 {
		t.Fatalf("expected exactly 1 probe to reach inner, got %d", c)
	}
}

func TestCircuitBreaker_Stream(t *testing.T) {
	inner := &countingFail{}
	cb := NewCircuitBreakerBackend(inner, CircuitBreakerConfig{Threshold: 2, OpenDuration: time.Minute})
	now := time.Now()
	cb.now = func() time.Time { return now }
	for i := 0; i < 2; i++ {
		cb.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { return nil })
	}
	err := cb.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { return nil })
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}
}

func TestRateLimit_AllowsBurstThenWaits(t *testing.T) {
	inner := newScripted()
	rl := NewRateLimitBackend(inner, RateLimitConfig{Capacity: 2, RefillRate: 100})
	var now time.Time = time.Now()
	rl.now = func() time.Time { return now }
	var slept time.Duration
	rl.sleep = func(ctx context.Context, d time.Duration) error {
		slept += d
		now = now.Add(d)
		return nil
	}
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		if _, err := rl.Chat(ctx, &ChatRequest{}); err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}
	if slept != 0 {
		t.Fatalf("burst should not sleep, slept %v", slept)
	}
	if _, err := rl.Chat(ctx, &ChatRequest{}); err != nil {
		t.Fatalf("call 3: %v", err)
	}
	if slept <= 0 {
		t.Fatal("expected a wait after burst exhausted")
	}
	if c := inner.calls(); c != 3 {
		t.Fatalf("expected 3 inner calls, got %d", c)
	}
}

func TestRateLimit_RefillsOverTime(t *testing.T) {
	inner := newScripted()
	rl := NewRateLimitBackend(inner, RateLimitConfig{Capacity: 1, RefillRate: 10})
	var now time.Time = time.Now()
	rl.now = func() time.Time { return now }
	rl.sleep = func(ctx context.Context, d time.Duration) error { now = now.Add(d); return nil }
	ctx := context.Background()
	rl.Chat(ctx, &ChatRequest{}) // uses the single token
	now = now.Add(2 * time.Second)
	before := now
	rl.Chat(ctx, &ChatRequest{}) // refilled, no wait
	if !now.Equal(before) {
		t.Fatal("should not need to wait after bucket refilled")
	}
}

func TestRateLimit_ContextCancelWhileWaiting(t *testing.T) {
	inner := newScripted()
	rl := NewRateLimitBackend(inner, RateLimitConfig{Capacity: 10, RefillRate: 1})
	rl.tokens = 0
	base := time.Now()
	rl.now = func() time.Time { return base }
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rl.sleep = func(ctx context.Context, d time.Duration) error { return ctx.Err() }
	if _, err := rl.Chat(ctx, &ChatRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if c := inner.calls(); c != 0 {
		t.Fatalf("inner should not be called, got %d", c)
	}
}

func TestRateLimit_StreamAcquiresToken(t *testing.T) {
	inner := newScripted()
	rl := NewRateLimitBackend(inner, RateLimitConfig{Capacity: 1, RefillRate: 100})
	rl.now = func() time.Time { return time.Now() }
	rl.sleep = func(ctx context.Context, d time.Duration) error { return nil }
	var chunks int
	if err := rl.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { chunks++; return nil }); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if chunks != 1 {
		t.Fatalf("expected 1 chunk, got %d", chunks)
	}
}

func TestFallback_PrimarySuccessNoFailover(t *testing.T) {
	primary := newScripted("primary")
	secondary := newScripted("secondary")
	fb := NewFallbackBackend(primary, secondary)
	resp, err := fb.Chat(context.Background(), &ChatRequest{})
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Fatalf("unexpected content %q", resp.Message.Content)
	}
	if c := secondary.calls(); c != 0 {
		t.Fatalf("secondary should not be called, got %d", c)
	}
}

func TestFallback_FailsOverToSecondary(t *testing.T) {
	primary := newScripted(errors.New("down"))
	secondary := newScripted("ok")
	fb := NewFallbackBackend(primary, secondary)
	if _, err := fb.Chat(context.Background(), &ChatRequest{}); err != nil {
		t.Fatalf("expected success via fallback, got %v", err)
	}
	if c := secondary.calls(); c != 1 {
		t.Fatalf("expected secondary called once, got %d", c)
	}
}

func TestFallback_AllFail(t *testing.T) {
	primary := newScripted(errors.New("p"))
	secondary := newScripted(errors.New("s"))
	fb := NewFallbackBackend(primary, secondary)
	if _, err := fb.Chat(context.Background(), &ChatRequest{}); err == nil {
		t.Fatal("expected error when all backends fail")
	}
}

func TestFallback_StreamFailover(t *testing.T) {
	primary := newScripted(errors.New("down"))
	secondary := newScripted("ok")
	fb := NewFallbackBackend(primary, secondary)
	var chunks int
	err := fb.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { chunks++; return nil })
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if chunks != 1 {
		t.Fatalf("expected 1 chunk, got %d", chunks)
	}
}

func TestFallback_BackendsOrder(t *testing.T) {
	a, b := newScripted(), newScripted()
	fb := NewFallbackBackend(a, nil, b)
	if len(fb.Backends()) != 2 {
		t.Fatalf("expected 2 backends (nil dropped), got %d", len(fb.Backends()))
	}
}

func TestFallback_ContextRespected(t *testing.T) {
	primary := newScripted(errors.New("down"))
	fb := NewFallbackBackend(primary, newScripted())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := fb.Chat(ctx, &ChatRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestMiddleware_ComposesInOrder(t *testing.T) {
	core := newScripted()
	mw := NewMiddleware(core,
		RetryMiddleware(RetryConfig{MaxAttempts: 3}),
		RateLimitMiddleware(RateLimitConfig{Capacity: 100, RefillRate: 1000}),
	)
	composed := mw.Build()
	rb, ok := composed.(*RetryBackend)
	if !ok {
		t.Fatalf("expected outermost *RetryBackend, got %T", composed)
	}
	if _, ok := rb.Backend().(*RateLimitBackend); !ok {
		t.Fatalf("expected inner *RateLimitBackend, got %T", rb.Backend())
	}
}

func TestMiddleware_EndToEndRetriesThroughRateLimit(t *testing.T) {
	core := newScripted(errors.New("boom"), "ok")
	mw := NewMiddleware(core,
		RetryMiddleware(RetryConfig{MaxAttempts: 3}),
		RateLimitMiddleware(RateLimitConfig{Capacity: 100, RefillRate: 1000}),
	)
	composed := mw.Build()
	composed.(*RetryBackend).sleep = testNoSleep
	composed.(*RetryBackend).Backend().(*RateLimitBackend).sleep = testNoSleep
	resp, err := composed.Chat(context.Background(), &ChatRequest{})
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Fatalf("unexpected content %q", resp.Message.Content)
	}
	if c := core.calls(); c != 2 {
		t.Fatalf("expected 2 core calls (1 retry), got %d", c)
	}
}

func TestMiddleware_StreamThroughChain(t *testing.T) {
	core := newScripted(errors.New("boom"), "ok")
	mw := NewMiddleware(core, RetryMiddleware(RetryConfig{MaxAttempts: 3}))
	composed := mw.Build()
	composed.(*RetryBackend).sleep = testNoSleep
	var chunks int
	if err := composed.ChatStream(context.Background(), &ChatRequest{}, func(*StreamChunk) error { chunks++; return nil }); err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if chunks != 1 {
		t.Fatalf("expected 1 chunk, got %d", chunks)
	}
}

func TestCircuitBreaker_String(t *testing.T) {
	if CircuitClosed.String() != "closed" || CircuitOpen.String() != "open" || CircuitHalfOpen.String() != "half-open" {
		t.Fatal("unexpected state strings")
	}
}
