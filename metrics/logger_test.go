package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestLogger_TextFormat(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelDebug, false)
	l.Info("hello", String("k", "v"), Int("n", 5))
	out := buf.String()
	for _, want := range []string{"[INFO]", "hello", "k=v", "n=5"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output, got: %q", want, out)
		}
	}
}

func TestLogger_JsonFormat(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelDebug, true)
	l.Warn("careful", String("k", "v"))
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(buf.String()), &m); err != nil {
		t.Fatalf("expected valid JSON: %v\n%s", err, buf.String())
	}
	if m["level"] != "warn" || m["msg"] != "careful" || m["k"] != "v" {
		t.Fatalf("unexpected json fields: %v", m)
	}
}

func TestLogger_LevelFiltering(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelWarn, false)
	l.Debug("no")
	l.Info("no")
	l.Warn("yes")
	l.Error("yes")
	out := buf.String()
	if strings.Contains(out, "no") {
		t.Fatalf("debug/info should be filtered at LevelWarn, got: %q", out)
	}
	if !strings.Contains(out, "yes") {
		t.Fatalf("warn/error should be emitted, got: %q", out)
	}
}

func TestLogger_WithFields(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelDebug, true).With(String("svc", "recall"))
	l.Info("msg")
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(buf.String()), &m); err != nil {
		t.Fatalf("expected valid JSON: %v", err)
	}
	if m["svc"] != "recall" || m["msg"] != "msg" {
		t.Fatalf("expected inherited field, got: %v", m)
	}
}

func TestLogger_CorrelationID(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelDebug, true)
	id := NewCorrelationID()
	ctx := WithCorrelationID(context.Background(), id)
	l.Ctx(ctx, LevelInfo, "with-id")
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(buf.String()), &m); err != nil {
		t.Fatalf("expected valid JSON: %v", err)
	}
	if m["correlation_id"] != id {
		t.Fatalf("expected correlation_id %q, got %v", id, m["correlation_id"])
	}
}

func TestLogger_ErrorField(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelDebug, true)
	l.Error("boom", Error("err", errors.New("fail")))
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(buf.String()), &m); err != nil {
		t.Fatalf("expected valid JSON: %v", err)
	}
	if m["err"] != "fail" {
		t.Fatalf("expected err=fail, got %v", m["err"])
	}
}

func TestLogger_DurationField(t *testing.T) {
	var buf bytes.Buffer
	l := NewLogger(&buf, LevelDebug, true)
	l.Info("took", Duration("d", 1500000000)) // 1.5s
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(buf.String()), &m); err != nil {
		t.Fatalf("expected valid JSON: %v", err)
	}
	if m["d"] != "1.5s" {
		t.Fatalf("expected d=1.5s, got %v", m["d"])
	}
}

func TestCorrelationID_Properties(t *testing.T) {
	id := NewCorrelationID()
	if len(id) != 16 {
		t.Fatalf("expected 16-char id, got %q (len %d)", id, len(id))
	}
	ctx := WithCorrelationID(context.Background(), id)
	if CorrelationID(ctx) != id {
		t.Fatalf("expected %q, got %q", id, CorrelationID(ctx))
	}
	if CorrelationID(context.Background()) != "" {
		t.Fatal("expected empty correlation id for a bare context")
	}
	if CorrelationID(nil) != "" {
		t.Fatal("expected empty correlation id for nil context")
	}
	// Uniqueness.
	if NewCorrelationID() == NewCorrelationID() {
		t.Fatal("expected unique correlation ids")
	}
}
