package metrics

import (
	"context"
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// Level is a log severity.
type Level int

// Log levels, from most to least verbose.
const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

// String returns the lowercase level name.
func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "debug"
	case LevelInfo:
		return "info"
	case LevelWarn:
		return "warn"
	case LevelError:
		return "error"
	default:
		return "unknown"
	}
}

// Field is a structured log key/value pair.
type Field struct {
	Key   string
	Value interface{}
}

// String builds a string Field.
func String(k, v string) Field { return Field{Key: k, Value: v} }

// Int builds an int Field.
func Int(k string, v int) Field { return Field{Key: k, Value: v} }

// Float builds a float64 Field.
func Float(k string, v float64) Field { return Field{Key: k, Value: v} }

// Bool builds a bool Field.
func Bool(k string, v bool) Field { return Field{Key: k, Value: v} }

// Duration builds a duration Field (rendered as a human-readable string).
func Duration(k string, v time.Duration) Field { return Field{Key: k, Value: v} }

// Error builds an error Field (rendered as its message, or null when nil).
func Error(k string, err error) Field { return Field{Key: k, Value: err} }

// Logger is a structured, level-filtering logger. It can emit JSON or
// key=value text and, when given a context carrying a correlation ID (see
// WithCorrelationID), includes it in every record. Child loggers built via
// With share the underlying writer and its lock, so output is not
// interleaved.
type Logger struct {
	mu     *sync.Mutex
	w      io.Writer
	level  Level
	json   bool
	fields []Field
}

// NewLogger returns a Logger writing to w, filtering at level, in JSON (true)
// or key=value text (false) format.
func NewLogger(w io.Writer, level Level, json bool) *Logger {
	return &Logger{w: w, level: level, json: json, mu: &sync.Mutex{}}
}

// With returns a child Logger that always includes the given fields. The
// child shares the parent's writer and lock.
func (l *Logger) With(fields ...Field) *Logger {
	nf := make([]Field, 0, len(l.fields)+len(fields))
	nf = append(nf, l.fields...)
	nf = append(nf, fields...)
	return &Logger{w: l.w, level: l.level, json: l.json, fields: nf, mu: l.mu}
}

// SetLevel changes the minimum level that will be emitted.
func (l *Logger) SetLevel(level Level) {
	l.mu.Lock()
	l.level = level
	l.mu.Unlock()
}

// Debug logs at LevelDebug.
func (l *Logger) Debug(msg string, fields ...Field) {
	l.Ctx(context.Background(), LevelDebug, msg, fields...)
}

// Info logs at LevelInfo.
func (l *Logger) Info(msg string, fields ...Field) {
	l.Ctx(context.Background(), LevelInfo, msg, fields...)
}

// Warn logs at LevelWarn.
func (l *Logger) Warn(msg string, fields ...Field) {
	l.Ctx(context.Background(), LevelWarn, msg, fields...)
}

// Error logs at LevelError.
func (l *Logger) Error(msg string, fields ...Field) {
	l.Ctx(context.Background(), LevelError, msg, fields...)
}

// Ctx logs at the given level, injecting the correlation ID from ctx when
// present.
func (l *Logger) Ctx(ctx context.Context, level Level, msg string, fields ...Field) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if level < l.level {
		return
	}
	all := make([]Field, 0, len(l.fields)+len(fields))
	all = append(all, l.fields...)
	all = append(all, fields...)
	if id := CorrelationID(ctx); id != "" {
		all = append(all, String("correlation_id", id))
	}
	l.writeLocked(level, msg, all)
}

func (l *Logger) writeLocked(level Level, msg string, fields []Field) {
	ts := time.Now().UTC().Format(time.RFC3339Nano)
	if l.json {
		m := map[string]interface{}{
			"time":  ts,
			"level": level.String(),
			"msg":   msg,
		}
		for _, f := range fields {
			m[f.Key] = normalizeValue(f.Value)
		}
		b, err := json.Marshal(m)
		if err != nil {
			fmt.Fprintf(l.w, "%s [%s] %s\n", ts, level.String(), msg)
			return
		}
		b = append(b, '\n')
		_, _ = l.w.Write(b)
		return
	}
	var sb strings.Builder
	sb.WriteString(ts)
	sb.WriteString(" [")
	sb.WriteString(strings.ToUpper(level.String()))
	sb.WriteString("] ")
	sb.WriteString(msg)
	for _, f := range fields {
		fmt.Fprintf(&sb, " %s=%v", f.Key, normalizeValue(f.Value))
	}
	sb.WriteByte('\n')
	_, _ = l.w.Write([]byte(sb.String()))
}

// normalizeValue converts a Field value into a display/JSON-friendly
// representation.
func normalizeValue(v interface{}) interface{} {
	switch t := v.(type) {
	case error:
		if t == nil {
			return nil
		}
		return t.Error()
	case time.Duration:
		return t.String()
	default:
		return v
	}
}

type ctxKey int

const correlationIDKey ctxKey = iota

// NewCorrelationID returns a new random correlation ID (16 hex characters).
func NewCorrelationID() string {
	var b [8]byte
	if _, err := crand.Read(b[:]); err != nil {
		// crypto/rand should not fail; fall back to a time-based ID.
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// WithCorrelationID returns a copy of ctx carrying the given correlation ID.
func WithCorrelationID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, correlationIDKey, id)
}

// CorrelationID returns the correlation ID stored in ctx, or "" if none.
func CorrelationID(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if id, ok := ctx.Value(correlationIDKey).(string); ok {
		return id
	}
	return ""
}
