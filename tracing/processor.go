package tracing

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
)

// SpanProcessor receives spans as they start and end. Implement this to forward
// spans to a backend (in-memory, console, or an OpenTelemetry collector).
type SpanProcessor interface {
	// OnStart is called when a span begins.
	OnStart(span *Span)
	// OnEnd is called when a span ends.
	OnEnd(span *Span)
}

// InMemoryProcessor collects ended spans for inspection, testing, and
// lightweight dashboards.
type InMemoryProcessor struct {
	mu    sync.Mutex
	spans []*Span
}

// NewInMemoryProcessor returns an empty InMemoryProcessor.
func NewInMemoryProcessor() *InMemoryProcessor {
	return &InMemoryProcessor{}
}

// OnStart is a no-op for the in-memory processor (only ended spans are kept).
func (p *InMemoryProcessor) OnStart(span *Span) {}

// OnEnd records the span.
func (p *InMemoryProcessor) OnEnd(span *Span) {
	p.mu.Lock()
	p.spans = append(p.spans, span)
	p.mu.Unlock()
}

// Spans returns a copy of all ended spans in completion order.
func (p *InMemoryProcessor) Spans() []*Span {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*Span, len(p.spans))
	copy(out, p.spans)
	return out
}

// Traces groups ended spans by trace ID. Within each trace, spans are ordered
// by start time.
func (p *InMemoryProcessor) Traces() map[string][]*Span {
	all := p.Spans()
	groups := make(map[string][]*Span)
	for _, s := range all {
		id := s.TraceID.String()
		groups[id] = append(groups[id], s)
	}
	for _, spans := range groups {
		sort.SliceStable(spans, func(i, j int) bool {
			return spans[i].Start.Before(spans[j].Start)
		})
	}
	return groups
}

// Count returns the number of ended spans.
func (p *InMemoryProcessor) Count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.spans)
}

// Reset clears all collected spans.
func (p *InMemoryProcessor) Reset() {
	p.mu.Lock()
	p.spans = nil
	p.mu.Unlock()
}

// ConsoleProcessor writes a one-line summary of each ended span to a writer.
// It is useful for development and debugging.
type ConsoleProcessor struct {
	mu sync.Mutex
	w  io.Writer
}

// NewConsoleProcessor returns a ConsoleProcessor writing to w.
func NewConsoleProcessor(w io.Writer) *ConsoleProcessor {
	return &ConsoleProcessor{w: w}
}

// OnStart is a no-op.
func (p *ConsoleProcessor) OnStart(span *Span) {}

// OnEnd writes a summary line for the span.
func (p *ConsoleProcessor) OnEnd(span *Span) {
	var attrs strings.Builder
	for k, v := range span.Attributes() {
		if attrs.Len() > 0 {
			attrs.WriteString(" ")
		}
		fmt.Fprintf(&attrs, "%s=%v", k, v)
	}
	p.mu.Lock()
	fmt.Fprintf(p.w, "trace=%s span=%s parent=%s name=%s kind=%s status=%s duration=%s %s\n",
		span.TraceID.String(),
		span.SpanID.String(),
		span.ParentID.String(),
		span.Name,
		span.Kind.String(),
		span.Status.String(),
		span.Duration(),
		strings.TrimSpace(attrs.String()),
	)
	p.mu.Unlock()
}
