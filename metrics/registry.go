// Package metrics provides a small, dependency-free, concurrency-safe
// observability toolkit: counters, gauges, histograms with percentile
// estimates, Prometheus text-format export, structured logging with
// correlation IDs, and ready-made metric bundles for the store, embedder,
// cache, and graph subsystems.
//
// The package is intentionally stdlib-only so it can be used anywhere in
// Recall without introducing external dependencies or CGO.
package metrics

import (
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Metric is implemented by every metric tracked in a Registry. It exposes
// enough information to render Prometheus text output and to order series
// deterministically.
type Metric interface {
	// BaseName returns the metric name without type suffixes (e.g.
	// "recall_search_latency" for a histogram whose samples are
	// recall_search_latency_bucket/_sum/_count).
	BaseName() string
	// Help returns a human-readable description for "# HELP".
	Help() string
	// Type returns the Prometheus type: "counter", "gauge", or "histogram".
	Type() string
	// Labels returns the fixed label set for this series.
	Labels() map[string]string
	// seriesLines returns the Prometheus sample line(s) for this series,
	// excluding the "# HELP"/"# TYPE" header lines.
	seriesLines() []string
}

// Registry is a thread-safe collection of metrics. Metric objects are
// created (or fetched, if an identical name+label series already exists) via
// the Counter/Gauge/Histogram constructors.
type Registry struct {
	mu      sync.RWMutex
	metrics map[string]Metric
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{metrics: make(map[string]Metric)}
}

// Counter returns the counter identified by name and labels, creating it on
// first use.
func (r *Registry) Counter(name, help string, labels map[string]string) *Counter {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := fingerprint(name, labels)
	if m, ok := r.metrics[key]; ok {
		return m.(*Counter)
	}
	c := &Counter{name: name, help: help, labels: cloneLabels(labels)}
	r.metrics[key] = c
	return c
}

// Gauge returns the gauge identified by name and labels, creating it on
// first use.
func (r *Registry) Gauge(name, help string, labels map[string]string) *Gauge {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := fingerprint(name, labels)
	if m, ok := r.metrics[key]; ok {
		return m.(*Gauge)
	}
	g := &Gauge{name: name, help: help, labels: cloneLabels(labels)}
	r.metrics[key] = g
	return g
}

// Histogram returns the histogram identified by name and labels, creating it
// on first use. The bucket bounds are fixed at first creation; subsequent
// calls with the same name+labels ignore the supplied buckets.
func (r *Registry) Histogram(name, help string, buckets []float64, labels map[string]string) *Histogram {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := fingerprint(name, labels)
	if m, ok := r.metrics[key]; ok {
		return m.(*Histogram)
	}
	h := newHistogram(name, help, buckets, cloneLabels(labels))
	r.metrics[key] = h
	return h
}

// Get returns all series whose base name matches name (any label set), in
// deterministic order.
func (r *Registry) Get(name string) []Metric {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []Metric
	for _, m := range r.metrics {
		if m.BaseName() == name {
			out = append(out, m)
		}
	}
	sortMetrics(out)
	return out
}

// Len returns the number of series in the registry.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.metrics)
}

// Snapshot returns a point-in-time copy of every counter and gauge value,
// keyed by sample name (name + labels). Histograms are represented by their
// "_count" sample. Useful for alerting and lightweight dashboards.
func (r *Registry) Snapshot() map[string]float64 {
	r.mu.RLock()
	ms := make([]Metric, 0, len(r.metrics))
	for _, m := range r.metrics {
		ms = append(ms, m)
	}
	r.mu.RUnlock()

	out := make(map[string]float64, len(ms))
	for _, m := range ms {
		switch v := m.(type) {
		case *Counter:
			out[sampleName(v.name, v.labels)] = v.Value()
		case *Gauge:
			out[sampleName(v.name, v.labels)] = v.Value()
		case *Histogram:
			out[sampleName(v.name+"_count", v.labels)] = float64(v.Count())
		}
	}
	return out
}

// Counter is a monotonically increasing metric.
type Counter struct {
	name   string
	help   string
	labels map[string]string

	mu    sync.Mutex
	value float64
}

func (c *Counter) BaseName() string          { return c.name }
func (c *Counter) Help() string              { return c.help }
func (c *Counter) Type() string              { return "counter" }
func (c *Counter) Labels() map[string]string { return c.labels }

// Inc increments the counter by one.
func (c *Counter) Inc() { c.Add(1) }

// Add increases the counter by v. Negative values are ignored (counters are
// monotonic).
func (c *Counter) Add(v float64) {
	if v < 0 {
		return
	}
	c.mu.Lock()
	c.value += v
	c.mu.Unlock()
}

// Value returns the current counter value.
func (c *Counter) Value() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.value
}

func (c *Counter) seriesLines() []string {
	return []string{sampleName(c.name, c.labels) + " " + formatFloat(c.Value())}
}

// Gauge is a metric that can go up and down.
type Gauge struct {
	name   string
	help   string
	labels map[string]string

	mu    sync.Mutex
	value float64
}

func (g *Gauge) BaseName() string          { return g.name }
func (g *Gauge) Help() string              { return g.help }
func (g *Gauge) Type() string              { return "gauge" }
func (g *Gauge) Labels() map[string]string { return g.labels }

// Set sets the gauge to v.
func (g *Gauge) Set(v float64) {
	g.mu.Lock()
	g.value = v
	g.mu.Unlock()
}

// Inc increases the gauge by one.
func (g *Gauge) Inc() { g.Add(1) }

// Dec decreases the gauge by one.
func (g *Gauge) Dec() { g.Add(-1) }

// Add changes the gauge by v (which may be negative).
func (g *Gauge) Add(v float64) {
	g.mu.Lock()
	g.value += v
	g.mu.Unlock()
}

// Value returns the current gauge value.
func (g *Gauge) Value() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.value
}

func (g *Gauge) seriesLines() []string {
	return []string{sampleName(g.name, g.labels) + " " + formatFloat(g.Value())}
}

// fingerprint produces a stable map key for a name + label set.
func fingerprint(name string, labels map[string]string) string {
	return name + "|" + labelKey(labels)
}

// labelKey renders a label set as a deterministic "k=v,k2=v2" string.
func labelKey(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(labels[k])
	}
	return b.String()
}

// sortMetrics orders metrics by base name then label key for deterministic
// output.
func sortMetrics(ms []Metric) {
	sort.SliceStable(ms, func(i, j int) bool {
		if ms[i].BaseName() != ms[j].BaseName() {
			return ms[i].BaseName() < ms[j].BaseName()
		}
		return labelKey(ms[i].Labels()) < labelKey(ms[j].Labels())
	})
}

// sampleName renders `name{k1="v1",k2="v2"}` (or just `name` when there are
// no labels).
func sampleName(name string, labels map[string]string) string {
	if len(labels) == 0 {
		return name
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteString(name)
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		writeLabel(&b, k, labels[k])
	}
	b.WriteByte('}')
	return b.String()
}

// writeLabel writes a single `key="value"` label pair (value escaped) to b.
func writeLabel(b *strings.Builder, k, v string) {
	b.WriteString(k)
	b.WriteByte('=')
	b.WriteByte('"')
	b.WriteString(escapeLabelValue(v))
	b.WriteByte('"')
}

// escapeLabelValue escapes a Prometheus label value.
func escapeLabelValue(v string) string {
	var b strings.Builder
	for _, r := range v {
		switch r {
		case '\\':
			b.WriteString(`\\`)
		case '"':
			b.WriteString(`\"`)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// formatFloat renders a float compactly (e.g. 0.5, 3, 1e+06).
func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// cloneLabels returns a copy of labels (nil-safe).
func cloneLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}
