package metrics

import (
	"fmt"
	"net/http"
	"strings"
)

// PrometheusContentType is the MIME type Prometheus scrapers expect.
const PrometheusContentType = "text/plain; version=0.0.4; charset=utf-8"

// RenderPrometheus renders every metric in the registry using the
// Prometheus text exposition format. Metric families are grouped by base name
// and sorted, so output is deterministic.
func (r *Registry) RenderPrometheus() string {
	r.mu.RLock()
	ms := make([]Metric, 0, len(r.metrics))
	for _, m := range r.metrics {
		ms = append(ms, m)
	}
	r.mu.RUnlock()

	sortMetrics(ms)

	var b strings.Builder
	current := ""
	for _, m := range ms {
		if m.BaseName() != current {
			if current != "" {
				b.WriteByte('\n')
			}
			fmt.Fprintf(&b, "# HELP %s %s\n", m.BaseName(), m.Help())
			fmt.Fprintf(&b, "# TYPE %s %s\n", m.BaseName(), m.Type())
			current = m.BaseName()
		}
		for _, line := range m.seriesLines() {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// HTTPHandler returns an http.Handler that serves the registry in Prometheus
// text format, suitable for mounting at a "/metrics" endpoint.
func (r *Registry) HTTPHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", PrometheusContentType)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(r.RenderPrometheus()))
	})
}

// defaultBuckets is a reasonable default set of latency bucket bounds
// (seconds) for sub-second to several-second operations.
func defaultBuckets() []float64 {
	return []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}
}

// DefaultLatencyBuckets returns a copy of the default latency bucket bounds
// (seconds).
func DefaultLatencyBuckets() []float64 {
	out := make([]float64, len(defaultBuckets()))
	copy(out, defaultBuckets())
	return out
}
