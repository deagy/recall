// Package tracing provides a small, dependency-free, OpenTelemetry-compatible
// tracing toolkit: 128-bit trace IDs and 64-bit span IDs, spans with
// attributes/events/status, context propagation, W3C `traceparent` inject/parse
// for cross-node correlation, and pluggable span processors (in-memory,
// console).
//
// The span model maps 1:1 onto OpenTelemetry spans (trace ID, span ID, parent
// span ID, name, kind, start/end, attributes, status), so completed spans can
// be forwarded to an OTel collector through a custom SpanProcessor. The
// package intentionally does not depend on the OTel SDK to keep Recall
// zero-dependency and zero-CGO.
package tracing

import (
	crand "crypto/rand"
	"encoding/hex"
	"time"
)

// TraceIDBytes is the byte length of a trace identifier (128-bit).
const TraceIDBytes = 16

// SpanIDBytes is the byte length of a span identifier (64-bit).
const SpanIDBytes = 8

// TraceID uniquely identifies a trace.
type TraceID [TraceIDBytes]byte

// NewTraceID returns a random, non-zero trace ID.
func NewTraceID() TraceID {
	var t TraceID
	copy(t[:], randomBytes(TraceIDBytes))
	return t
}

// String returns the lowercase hex encoding of the trace ID.
func (t TraceID) String() string { return hex.EncodeToString(t[:]) }

// IsZero reports whether the trace ID is all zeros.
func (t TraceID) IsZero() bool {
	for _, b := range t {
		if b != 0 {
			return false
		}
	}
	return true
}

// SpanID uniquely identifies a span within a trace.
type SpanID [SpanIDBytes]byte

// NewSpanID returns a random, non-zero span ID.
func NewSpanID() SpanID {
	var s SpanID
	copy(s[:], randomBytes(SpanIDBytes))
	return s
}

// String returns the lowercase hex encoding of the span ID.
func (s SpanID) String() string { return hex.EncodeToString(s[:]) }

// IsZero reports whether the span ID is all zeros.
func (s SpanID) IsZero() bool {
	for _, b := range s {
		if b != 0 {
			return false
		}
	}
	return true
}

// randomBytes returns n cryptographically random bytes, falling back to a
// time-derived value if the random source fails. The result is never all
// zeros.
func randomBytes(n int) []byte {
	b := make([]byte, n)
	if _, err := crand.Read(b); err != nil {
		seed := time.Now().UnixNano()
		for i := range b {
			b[i] = byte(seed >> (8 * i))
		}
	}
	allZero := true
	for _, v := range b {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		b[0] = 1
	}
	return b
}
