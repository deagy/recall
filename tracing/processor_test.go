package tracing

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestInMemoryProcessor_CollectsAndGroups(t *testing.T) {
	proc := NewInMemoryProcessor()
	tr := NewTracer(proc)
	ctx := context.Background()

	// Trace A: a1 is the parent of a2.
	ctxA, a1 := tr.Start(ctx, "a1")
	a1.End()
	_, a2 := tr.Start(ctxA, "a2")
	a2.End()
	// Trace B: a single root span.
	_, b1 := tr.Start(ctx, "b1")
	b1.End()

	if proc.Count() != 3 {
		t.Fatalf("expected 3 spans, got %d", proc.Count())
	}
	traces := proc.Traces()
	if len(traces) != 2 {
		t.Fatalf("expected 2 traces, got %d", len(traces))
	}
	var aTraceID string
	for id, spans := range traces {
		if len(spans) == 2 {
			aTraceID = id
		}
	}
	if aTraceID == "" {
		t.Fatal("expected a trace containing 2 spans")
	}
}

func TestInMemoryProcessor_Reset(t *testing.T) {
	proc := NewInMemoryProcessor()
	tr := NewTracer(proc)
	_, s := tr.Start(context.Background(), "op")
	s.End()
	if proc.Count() != 1 {
		t.Fatalf("expected 1 span, got %d", proc.Count())
	}
	proc.Reset()
	if proc.Count() != 0 {
		t.Fatalf("expected 0 spans after reset, got %d", proc.Count())
	}
}

func TestConsoleProcessor_Writes(t *testing.T) {
	var buf bytes.Buffer
	proc := NewConsoleProcessor(&buf)
	tr := NewTracer(proc)
	_, s := tr.Start(context.Background(), "op")
	s.SetAttribute("k", "v")
	s.SetStatus(StatusOK, "")
	s.End()

	out := buf.String()
	if !strings.Contains(out, "name=op") || !strings.Contains(out, "k=v") {
		t.Fatalf("expected console output to include name and attribute, got: %q", out)
	}
	if !strings.Contains(out, "status=ok") {
		t.Fatalf("expected status=ok in output, got: %q", out)
	}
}
