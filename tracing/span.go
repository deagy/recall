package tracing

import (
	"sync"
	"time"
)

// SpanKind classifies a span's role in a trace.
type SpanKind int

// Span kinds, mirroring OpenTelemetry.
const (
	// SpanKindInternal is the default; the span is for internal processing.
	SpanKindInternal SpanKind = iota
	// SpanKindServer indicates the span handles an incoming request.
	SpanKindServer
	// SpanKindClient indicates the span makes an outgoing request.
	SpanKindClient
	// SpanKindProducer indicates the span publishes a message.
	SpanKindProducer
	// SpanKindConsumer indicates the span processes a message.
	SpanKindConsumer
)

// String returns the lowercase kind name.
func (k SpanKind) String() string {
	switch k {
	case SpanKindServer:
		return "server"
	case SpanKindClient:
		return "client"
	case SpanKindProducer:
		return "producer"
	case SpanKindConsumer:
		return "consumer"
	default:
		return "internal"
	}
}

// SpanStatus is the terminal state of a span.
type SpanStatus int

// Span statuses, mirroring OpenTelemetry.
const (
	// StatusUnset means the span's status has not been set.
	StatusUnset SpanStatus = iota
	// StatusOK means the operation completed successfully.
	StatusOK
	// StatusError means the operation failed.
	StatusError
)

// String returns the lowercase status name.
func (s SpanStatus) String() string {
	switch s {
	case StatusOK:
		return "ok"
	case StatusError:
		return "error"
	default:
		return "unset"
	}
}

// Event is a timestamped annotation on a span.
type Event struct {
	// Name is the event name.
	Name string
	// Time is when the event occurred.
	Time time.Time
	// Attributes are optional event tags.
	Attributes map[string]interface{}
}

// Span is a unit of work within a trace. A span has a start time, an end
// time, a name, a kind, a set of attributes, a status, and zero or more
// events. Spans form a tree via ParentID.
type Span struct {
	// TraceID is the trace this span belongs to.
	TraceID TraceID
	// SpanID uniquely identifies this span.
	SpanID SpanID
	// ParentID is the ID of the parent span, or zero for a root span.
	ParentID SpanID
	// Name is the operation name.
	Name string
	// Kind classifies the span.
	Kind SpanKind
	// Status is the terminal state.
	Status SpanStatus
	// StatusMsg is an optional message accompanying StatusError.
	StatusMsg string
	// Start is when the span began.
	Start time.Time
	// EndTime is when the span finished (zero until End is called).
	EndTime time.Time

	mu         sync.Mutex
	attributes map[string]interface{}
	events     []Event
	ended      bool
	tracer     *Tracer
}

// SetAttribute records a key/value tag on the span. It is safe for concurrent
// use.
func (s *Span) SetAttribute(k string, v interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.attributes == nil {
		s.attributes = make(map[string]interface{})
	}
	s.attributes[k] = v
}

// Attribute returns the value of a tag, or nil if absent.
func (s *Span) Attribute(k string) interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.attributes[k]
}

// Attributes returns a copy of the span's tags.
func (s *Span) Attributes() map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]interface{}, len(s.attributes))
	for k, v := range s.attributes {
		out[k] = v
	}
	return out
}

// AddEvent records a timestamped annotation with optional tags.
func (s *Span) AddEvent(name string, attrs map[string]interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, Event{Name: name, Time: time.Now().UTC(), Attributes: attrs})
}

// Events returns a copy of the span's events in order.
func (s *Span) Events() []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Event, len(s.events))
	copy(out, s.events)
	return out
}

// SetStatus sets the terminal status and optional message.
func (s *Span) SetStatus(st SpanStatus, msg string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Status = st
	s.StatusMsg = msg
}

// IsRoot reports whether the span has no parent.
func (s *Span) IsRoot() bool { return s.ParentID.IsZero() }

// Duration returns the elapsed time from start to end (or now if not ended).
func (s *Span) Duration() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	end := s.EndTime
	if end.IsZero() {
		end = time.Now()
	}
	return end.Sub(s.Start)
}

// IsEnded reports whether the span has been ended.
func (s *Span) IsEnded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ended
}

// End finishes the span, records its end time, and hands it to the tracer's
// processors. It is safe to call more than once; only the first call takes
// effect.
func (s *Span) End() {
	s.mu.Lock()
	if s.ended {
		s.mu.Unlock()
		return
	}
	s.ended = true
	s.EndTime = time.Now().UTC()
	tracer := s.tracer
	s.mu.Unlock()
	if tracer != nil {
		tracer.endSpan(s)
	}
}
