package tracing

import (
	"context"
	"sync"
	"time"
)

// Tracer creates and manages spans, handing them to a set of processors.
type Tracer struct {
	processors []SpanProcessor
}

// NewTracer returns a Tracer that reports spans to the given processors. With
// no processors, spans are still created and nested but not recorded anywhere.
func NewTracer(processors ...SpanProcessor) *Tracer {
	return &Tracer{processors: processors}
}

// Start begins a new span named name, parented to the span carried in ctx (or
// a root span if there is none), and returns a context carrying the new span
// along with the span itself. Call the span's End method to finish it.
func (t *Tracer) Start(ctx context.Context, name string, opts ...SpanOption) (context.Context, *Span) {
	cfg := &spanConfig{kind: SpanKindInternal}
	for _, o := range opts {
		o(cfg)
	}

	var traceID TraceID
	var parent SpanID
	if p := SpanFromContext(ctx); p != nil {
		traceID = p.TraceID
		parent = p.SpanID
	}
	if !cfg.traceID.IsZero() {
		traceID = cfg.traceID
	}
	if !cfg.parentID.IsZero() {
		parent = cfg.parentID
	}
	if traceID.IsZero() {
		traceID = NewTraceID()
	}

	span := &Span{
		TraceID:    traceID,
		SpanID:     NewSpanID(),
		ParentID:   parent,
		Name:       name,
		Kind:       cfg.kind,
		Status:     StatusUnset,
		Start:      time.Now().UTC(),
		attributes: cloneAttributes(cfg.attributes),
		tracer:     t,
	}
	for _, p := range t.processors {
		p.OnStart(span)
	}
	return ContextWithSpan(ctx, span), span
}

func (t *Tracer) endSpan(span *Span) {
	for _, p := range t.processors {
		p.OnEnd(span)
	}
}

// spanConfig accumulates options for a new span.
type spanConfig struct {
	kind       SpanKind
	attributes map[string]interface{}
	traceID    TraceID
	parentID   SpanID
}

// SpanOption customizes a new span.
type SpanOption func(*spanConfig)

// WithKind sets the span's kind.
func WithKind(k SpanKind) SpanOption {
	return func(c *spanConfig) { c.kind = k }
}

// WithAttributes sets initial tags on the span.
func WithAttributes(attrs map[string]interface{}) SpanOption {
	return func(c *spanConfig) { c.attributes = attrs }
}

// WithTraceID starts the span in the given trace (for continuing a remote
// trace parsed from a propagation header).
func WithTraceID(id TraceID) SpanOption {
	return func(c *spanConfig) { c.traceID = id }
}

// WithParent sets an explicit parent span ID.
func WithParent(id SpanID) SpanOption {
	return func(c *spanConfig) { c.parentID = id }
}

type ctxKey int

const (
	spanContextKey ctxKey = iota
)

// ContextWithSpan returns a context carrying span.
func ContextWithSpan(ctx context.Context, s *Span) context.Context {
	return context.WithValue(ctx, spanContextKey, s)
}

// SpanFromContext returns the span carried in ctx, or nil if there is none.
func SpanFromContext(ctx context.Context) *Span {
	if ctx == nil {
		return nil
	}
	if s, ok := ctx.Value(spanContextKey).(*Span); ok {
		return s
	}
	return nil
}

var (
	defaultMu     sync.RWMutex
	defaultTracer = NewTracer()
)

// SetDefaultTracer sets the process-wide tracer used by StartSpan. Passing nil
// resets it to a no-op tracer.
func SetDefaultTracer(t *Tracer) {
	if t == nil {
		t = NewTracer()
	}
	defaultMu.Lock()
	defaultTracer = t
	defaultMu.Unlock()
}

// DefaultTracer returns the process-wide tracer.
func DefaultTracer() *Tracer {
	defaultMu.RLock()
	defer defaultMu.RUnlock()
	return defaultTracer
}

// StartSpan begins a span using the process-wide default tracer, parented to
// the span in ctx if present.
func StartSpan(ctx context.Context, name string, opts ...SpanOption) (context.Context, *Span) {
	return DefaultTracer().Start(ctx, name, opts...)
}

// cloneAttributes returns a copy of attrs (nil-safe).
func cloneAttributes(attrs map[string]interface{}) map[string]interface{} {
	if len(attrs) == 0 {
		return nil
	}
	out := make(map[string]interface{}, len(attrs))
	for k, v := range attrs {
		out[k] = v
	}
	return out
}
