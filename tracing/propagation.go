package tracing

import (
	"context"
	"encoding/hex"
	"strings"
)

// TraceParentHeader is the HTTP header name for W3C Trace Context.
const TraceParentHeader = "traceparent"

// TraceFlags are the 8-bit flags carried in a traceparent header.
type TraceFlags uint8

// FlagSampled indicates the trace is sampled for recording.
const FlagSampled TraceFlags = 0x01

// IsSampled reports whether the sampled flag is set.
func (f TraceFlags) IsSampled() bool { return f&FlagSampled != 0 }

// TraceParent is a parsed W3C traceparent header.
type TraceParent struct {
	// Version is the protocol version byte.
	Version byte
	// TraceID is the 128-bit trace identifier.
	TraceID TraceID
	// SpanID is the 64-bit parent span identifier.
	SpanID SpanID
	// Flags are the trace flags.
	Flags TraceFlags
}

// Inject renders the W3C traceparent header value for the given trace and
// span, setting the sampled flag when sampled is true.
func Inject(traceID TraceID, spanID SpanID, sampled bool) string {
	flags := "00"
	if sampled {
		flags = "01"
	}
	return "00-" + traceID.String() + "-" + spanID.String() + "-" + flags
}

// ParseTraceParent parses a W3C traceparent header value of the form
// `version-traceid-spanid-flags`. It reports false if the value is malformed.
func ParseTraceParent(v string) (TraceParent, bool) {
	var tp TraceParent
	parts := strings.Split(v, "-")
	if len(parts) != 4 {
		return tp, false
	}
	if len(parts[0]) != 2 || len(parts[1]) != 32 || len(parts[2]) != 16 || len(parts[3]) != 2 {
		return tp, false
	}
	version, err := hex.DecodeString(parts[0])
	if err != nil || len(version) != 1 || version[0] == 0xff {
		return tp, false
	}
	traceBytes, err := hex.DecodeString(parts[1])
	if err != nil || len(traceBytes) != TraceIDBytes {
		return tp, false
	}
	spanBytes, err := hex.DecodeString(parts[2])
	if err != nil || len(spanBytes) != SpanIDBytes {
		return tp, false
	}
	flagBytes, err := hex.DecodeString(parts[3])
	if err != nil || len(flagBytes) != 1 {
		return tp, false
	}
	var traceID TraceID
	copy(traceID[:], traceBytes)
	if traceID.IsZero() {
		return tp, false
	}
	var spanID SpanID
	copy(spanID[:], spanBytes)
	if spanID.IsZero() {
		return tp, false
	}
	tp.Version = version[0]
	tp.TraceID = traceID
	tp.SpanID = spanID
	tp.Flags = TraceFlags(flagBytes[0])
	return tp, true
}

// StartRemoteSpan begins a span that continues the trace identified by tp, as
// a child of tp.SpanID. It is the entry point for correlating a request with
// an upstream service's trace.
func StartRemoteSpan(ctx context.Context, name string, tp TraceParent, opts ...SpanOption) (context.Context, *Span) {
	all := make([]SpanOption, 0, len(opts)+2)
	all = append(all, WithTraceID(tp.TraceID), WithParent(tp.SpanID))
	all = append(all, opts...)
	return StartSpan(ctx, name, all...)
}
