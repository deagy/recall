package metrics

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"
)

// reservoirCap is the maximum number of observations retained for percentile
// estimation. When the number of observations is below this, the reservoir
// holds every observation and percentiles are exact.
const reservoirCap = 1024

// Histogram records observations into fixed buckets (for Prometheus output)
// and a bounded random reservoir (for p50/p95/p99-style percentile
// estimates).
type Histogram struct {
	name   string
	help   string
	labels map[string]string

	mu        sync.Mutex
	buckets   []float64 // strictly increasing upper bounds
	counts    []uint64  // observations per bucket (non-cumulative)
	sum       float64
	count     uint64
	reservoir []float64
	rng       *rand.Rand
}

// newHistogram builds a Histogram. buckets must be sorted ascending; if it is
// not, a copy is sorted. An empty bucket list yields a histogram that still
// tracks sum/count and percentiles but renders no _bucket samples.
func newHistogram(name, help string, buckets []float64, labels map[string]string) *Histogram {
	b := make([]float64, len(buckets))
	copy(b, buckets)
	sort.Float64s(b)
	src := rand.NewSource(time.Now().UnixNano())
	return &Histogram{
		name:    name,
		help:    help,
		labels:  labels,
		buckets: b,
		counts:  make([]uint64, len(b)),
		rng:     rand.New(src),
	}
}

func (h *Histogram) BaseName() string          { return h.name }
func (h *Histogram) Help() string              { return h.help }
func (h *Histogram) Type() string              { return "histogram" }
func (h *Histogram) Labels() map[string]string { return h.labels }

// Observe records a single observation.
func (h *Histogram) Observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.sum += v
	h.count++

	if len(h.buckets) > 0 {
		// First bucket whose upper bound is >= v.
		idx := sort.SearchFloat64s(h.buckets, v)
		if idx < len(h.counts) {
			h.counts[idx]++
		}
	}

	// Vitter's Algorithm R reservoir sampling.
	if len(h.reservoir) < reservoirCap {
		h.reservoir = append(h.reservoir, v)
	} else {
		j := h.rng.Intn(int(h.count)) // 0..count-1, count == i+1
		if j < reservoirCap {
			h.reservoir[j] = v
		}
	}
}

// Sum returns the sum of all observations.
func (h *Histogram) Sum() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.sum
}

// Count returns the number of observations.
func (h *Histogram) Count() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.count
}

// Mean returns the average observation, or 0 if empty.
func (h *Histogram) Mean() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.count == 0 {
		return 0
	}
	return h.sum / float64(h.count)
}

// Percentile returns the estimated p-th percentile (p in (0, 100]) using the
// nearest-rank method over the retained reservoir. It returns 0 when no
// observations have been recorded. Out-of-range p values are clamped.
func (h *Histogram) Percentile(p float64) float64 {
	if p <= 0 {
		p = 1
	}
	if p > 100 {
		p = 100
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	n := len(h.reservoir)
	if n == 0 {
		return 0
	}
	data := make([]float64, n)
	copy(data, h.reservoir)
	sort.Float64s(data)
	rank := int(math.Ceil(p / 100.0 * float64(n)))
	if rank < 1 {
		rank = 1
	}
	if rank > n {
		rank = n
	}
	return data[rank-1]
}

// Common percentile shortcuts.

// P50 returns the median.
func (h *Histogram) P50() float64 { return h.Percentile(50) }

// P95 returns the 95th percentile.
func (h *Histogram) P95() float64 { return h.Percentile(95) }

// P99 returns the 99th percentile.
func (h *Histogram) P99() float64 { return h.Percentile(99) }

func (h *Histogram) seriesLines() []string {
	h.mu.Lock()
	defer h.mu.Unlock()

	base := h.name
	lines := make([]string, 0, len(h.buckets)+3)
	var cumulative uint64
	for i, bound := range h.buckets {
		cumulative += h.counts[i]
		lines = append(lines, fmt.Sprintf(
			"%s_bucket%s %d",
			base,
			labelFragment(withLabel(h.labels, "le", formatFloat(bound))),
			cumulative,
		))
	}
	lines = append(lines, fmt.Sprintf("%s_bucket%s %d", base, labelFragment(withLabel(h.labels, "le", "+Inf")), h.count))
	lines = append(lines, sampleName(base+"_sum", h.labels)+" "+formatFloat(h.sum))
	lines = append(lines, sampleName(base+"_count", h.labels)+" "+formatFloat(float64(h.count)))
	return lines
}

// withLabel returns a copy of base with the extra label k=v added.
func withLabel(base map[string]string, k, v string) map[string]string {
	m := make(map[string]string, len(base)+1)
	for a, b := range base {
		m[a] = b
	}
	m[k] = v
	return m
}

// labelFragment renders `{k1="v1",k2="v2"}` (or "" when empty) for use after
// a metric family name.
func labelFragment(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
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
