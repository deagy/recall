package tracing

import (
	"context"
	"strings"
	"testing"
)

func TestTraceParent_RoundTrip(t *testing.T) {
	tid := NewTraceID()
	sid := NewSpanID()
	for _, sampled := range []bool{true, false} {
		v := Inject(tid, sid, sampled)
		parsed, ok := ParseTraceParent(v)
		if !ok {
			t.Fatalf("expected to parse %q", v)
		}
		if parsed.TraceID != tid {
			t.Fatalf("trace ID mismatch: %v != %v", parsed.TraceID, tid)
		}
		if parsed.SpanID != sid {
			t.Fatalf("span ID mismatch: %v != %v", parsed.SpanID, sid)
		}
		if parsed.Flags.IsSampled() != sampled {
			t.Fatalf("sampled flag mismatch: got %v, want %v", parsed.Flags.IsSampled(), sampled)
		}
		if parsed.Version != 0 {
			t.Fatalf("expected version 0, got %v", parsed.Version)
		}
	}
}

func TestTraceParent_Format(t *testing.T) {
	tid := NewTraceID()
	sid := NewSpanID()
	v := Inject(tid, sid, true)
	parts := strings.Split(v, "-")
	if len(parts) != 4 {
		t.Fatalf("expected 4 dash-separated parts, got %v", parts)
	}
	if parts[0] != "00" || parts[3] != "01" {
		t.Fatalf("unexpected version/flags: %v", parts)
	}
	if len(parts[1]) != 32 || len(parts[2]) != 16 {
		t.Fatalf("unexpected ID lengths: %v", parts)
	}
}

func TestTraceParent_Malformed(t *testing.T) {
	a32 := strings.Repeat("a", 32)
	a16 := strings.Repeat("a", 16)
	bad := []string{
		"",
		"00-",
		"00-abc",
		"00-" + strings.Repeat("0", 32) + "-" + a16 + "-01", // zero trace ID
		"00-" + a32 + "-" + strings.Repeat("0", 16) + "-01", // zero span ID
		"ff-" + a32 + "-" + a16 + "-01",                     // invalid version
		"00-" + strings.Repeat("a", 31) + "-" + a16 + "-01", // wrong trace length
		"00-" + a32 + "-" + strings.Repeat("a", 15) + "-01", // wrong span length
		"00-" + strings.Repeat("g", 32) + "-" + a16 + "-01", // non-hex
	}
	for _, v := range bad {
		if _, ok := ParseTraceParent(v); ok {
			t.Fatalf("expected %q to be rejected as malformed", v)
		}
	}
}

func TestStartRemoteSpan(t *testing.T) {
	proc := NewInMemoryProcessor()
	old := DefaultTracer()
	SetDefaultTracer(NewTracer(proc))
	defer SetDefaultTracer(old)

	tid := NewTraceID()
	remoteParent := NewSpanID()
	tp := TraceParent{TraceID: tid, SpanID: remoteParent, Flags: FlagSampled}

	_, span := StartRemoteSpan(context.Background(), "remote-child", tp)
	span.End()

	spans := proc.Spans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].TraceID != tid {
		t.Fatal("expected the span to continue the remote trace")
	}
	if spans[0].ParentID != remoteParent {
		t.Fatal("expected the span to be a child of the remote parent")
	}
}
