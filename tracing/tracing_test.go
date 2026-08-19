package tracing

import (
	"context"
	"testing"
	"time"
)

func TestTraceID_SpanID(t *testing.T) {
	tid := NewTraceID()
	sid := NewSpanID()
	if tid.IsZero() {
		t.Fatal("expected a non-zero trace ID")
	}
	if sid.IsZero() {
		t.Fatal("expected a non-zero span ID")
	}
	if len(tid.String()) != 32 {
		t.Fatalf("expected 32-hex trace ID, got %q", tid.String())
	}
	if len(sid.String()) != 16 {
		t.Fatalf("expected 16-hex span ID, got %q", sid.String())
	}
	if NewTraceID() == NewTraceID() {
		t.Fatal("expected unique trace IDs")
	}
}

func TestSpan_ChildOfParent(t *testing.T) {
	tr := NewTracer()
	ctx := context.Background()

	ctx, root := tr.Start(ctx, "root")
	_, child := tr.Start(ctx, "child")
	child.SetAttribute("k", "v")
	child.End()
	root.End()

	if child.TraceID != root.TraceID {
		t.Fatal("expected the child to share the root's trace ID")
	}
	if child.ParentID != root.SpanID {
		t.Fatal("expected the child's parent to be the root span ID")
	}
	if !root.IsRoot() {
		t.Fatal("expected the root span to have no parent")
	}
}

func TestSpan_AttributesEventsStatus(t *testing.T) {
	tr := NewTracer()
	_, span := tr.Start(context.Background(), "op")
	span.SetAttribute("a", 1)
	span.SetAttribute("b", "two")
	span.AddEvent("checkpoint", map[string]interface{}{"x": 1})
	span.SetStatus(StatusOK, "")
	span.End()

	if span.Attribute("a") != 1 {
		t.Fatalf("expected a=1, got %v", span.Attribute("a"))
	}
	if span.Attribute("b") != "two" {
		t.Fatalf("expected b=two, got %v", span.Attribute("b"))
	}
	events := span.Events()
	if len(events) != 1 || events[0].Name != "checkpoint" {
		t.Fatalf("expected one checkpoint event, got %v", events)
	}
	if span.Status != StatusOK {
		t.Fatalf("expected OK status, got %v", span.Status)
	}
	attrs := span.Attributes()
	if attrs["a"] != 1 || attrs["b"] != "two" {
		t.Fatalf("unexpected attributes copy: %v", attrs)
	}
}

func TestSpan_EndIdempotentAndDuration(t *testing.T) {
	proc := NewInMemoryProcessor()
	tr := NewTracer(proc)
	_, span := tr.Start(context.Background(), "op")
	time.Sleep(5 * time.Millisecond)
	span.End()
	span.End() // second call is a no-op

	if proc.Count() != 1 {
		t.Fatalf("expected the span reported once, got %d", proc.Count())
	}
	if !span.IsEnded() {
		t.Fatal("expected the span to be ended")
	}
	if span.Duration() < 5*time.Millisecond {
		t.Fatalf("expected duration >= 5ms, got %v", span.Duration())
	}
}

func TestSpan_ContextPropagation(t *testing.T) {
	tr := NewTracer()
	ctx, root := tr.Start(context.Background(), "root")
	if SpanFromContext(ctx) != root {
		t.Fatal("expected the context to carry the root span")
	}
	if SpanFromContext(context.Background()) != nil {
		t.Fatal("expected no span in a bare context")
	}
	if SpanFromContext(nil) != nil {
		t.Fatal("expected no span in a nil context")
	}
	root.End()
}

func TestSpan_OptionOverrides(t *testing.T) {
	tr := NewTracer()
	remoteTrace := NewTraceID()
	remoteParent := NewSpanID()
	_, span := tr.Start(context.Background(), "op",
		WithKind(SpanKindClient),
		WithTraceID(remoteTrace),
		WithParent(remoteParent),
		WithAttributes(map[string]interface{}{"tag": "x"}),
	)
	if span.TraceID != remoteTrace {
		t.Fatal("expected the span to use the provided trace ID")
	}
	if span.ParentID != remoteParent {
		t.Fatal("expected the span to use the provided parent ID")
	}
	if span.Kind != SpanKindClient {
		t.Fatalf("expected client kind, got %v", span.Kind)
	}
	if span.Attribute("tag") != "x" {
		t.Fatalf("expected tag=x, got %v", span.Attribute("tag"))
	}
	span.End()
}

func TestDefaultTracer_StartSpan(t *testing.T) {
	proc := NewInMemoryProcessor()
	old := DefaultTracer()
	SetDefaultTracer(NewTracer(proc))
	defer SetDefaultTracer(old)

	_, span := StartSpan(context.Background(), "default-op")
	span.End()
	if proc.Count() != 1 {
		t.Fatalf("expected the default tracer to record the span, got %d", proc.Count())
	}
}
