package metrics

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRenderPrometheus_Families(t *testing.T) {
	r := NewRegistry()
	r.Counter("reqs", "Total requests.", map[string]string{"code": "200"}).Inc()
	r.Gauge("temp", "Temperature.", nil).Set(3)

	out := r.RenderPrometheus()
	for _, want := range []string{
		"# HELP reqs Total requests.",
		"# TYPE reqs counter",
		`reqs{code="200"} 1`,
		"# HELP temp Temperature.",
		"# TYPE temp gauge",
		"temp 3",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderPrometheus_Histogram(t *testing.T) {
	r := NewRegistry()
	h := r.Histogram("lat", "Latency.", []float64{1, 2}, nil)
	h.Observe(0.5)
	h.Observe(1.5)
	out := r.RenderPrometheus()
	for _, want := range []string{
		"# TYPE lat histogram",
		`lat_bucket{le="1"} 1`,
		`lat_bucket{le="2"} 2`,
		`lat_bucket{le="+Inf"} 2`,
		"lat_sum 2",
		"lat_count 2",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestRenderPrometheus_LabelEscaping(t *testing.T) {
	r := NewRegistry()
	r.Counter("c", "help", map[string]string{"doc": "a\"b\\c\nd"}).Inc()
	out := r.RenderPrometheus()
	if !strings.Contains(out, `doc="a\"b\\c\nd"`) {
		t.Fatalf("expected escaped label value, got:\n%s", out)
	}
}

func TestRenderPrometheus_DeterministicAndSorted(t *testing.T) {
	r := NewRegistry()
	// Register in non-sorted order.
	r.Counter("z", "help", nil).Inc()
	r.Counter("a", "help", nil).Inc()
	r.Gauge("m", "help", nil).Set(1)

	out1 := r.RenderPrometheus()
	out2 := r.RenderPrometheus()
	if out1 != out2 {
		t.Fatal("expected deterministic output across renders")
	}
	// Families sorted by base name: a < m < z.
	ia := strings.Index(out1, "# HELP a ")
	im := strings.Index(out1, "# HELP m ")
	iz := strings.Index(out1, "# HELP z ")
	if !(ia >= 0 && ia < im && im < iz) {
		t.Fatalf("expected families sorted a<m<z, got a@%d m@%d z@%d", ia, im, iz)
	}
}

func TestHTTPHandler(t *testing.T) {
	r := NewRegistry()
	r.Counter("c", "help", nil).Inc()
	h := r.HTTPHandler()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("unexpected content type %q", ct)
	}
	if !strings.Contains(rec.Body.String(), "c 1") {
		t.Fatalf("expected body to contain 'c 1', got:\n%s", rec.Body.String())
	}
}
