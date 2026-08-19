package metrics

import "testing"

func TestHistogram_BucketsAndStats(t *testing.T) {
	r := NewRegistry()
	h := r.Histogram("lat", "help", []float64{0.1, 0.5, 1.0}, nil)
	vals := []float64{0.05, 0.2, 0.4, 0.6, 0.9, 1.2}
	for _, v := range vals {
		h.Observe(v)
	}
	if h.Count() != 6 {
		t.Fatalf("expected count 6, got %d", h.Count())
	}
	sum := 0.05 + 0.2 + 0.4 + 0.6 + 0.9 + 1.2
	if !approx(h.Sum(), sum) {
		t.Fatalf("expected sum %v, got %v", sum, h.Sum())
	}
	if !approx(h.Mean(), sum/6) {
		t.Fatalf("expected mean %v, got %v", sum/6, h.Mean())
	}
}

// When the number of observations is below the reservoir cap, the reservoir
// holds every value and percentiles are exact (nearest-rank).
func TestHistogram_PercentilesExact(t *testing.T) {
	r := NewRegistry()
	h := r.Histogram("lat", "help", []float64{10}, nil)
	for i := 1; i <= 100; i++ {
		h.Observe(float64(i))
	}
	if !approx(h.P50(), 50) {
		t.Fatalf("expected p50=50, got %v", h.P50())
	}
	if !approx(h.P95(), 95) {
		t.Fatalf("expected p95=95, got %v", h.P95())
	}
	if !approx(h.P99(), 99) {
		t.Fatalf("expected p99=99, got %v", h.P99())
	}
	if !approx(h.Percentile(100), 100) {
		t.Fatalf("expected p100=100, got %v", h.Percentile(100))
	}
	if !approx(h.Percentile(1), 1) {
		t.Fatalf("expected p1=1, got %v", h.Percentile(1))
	}
}

func TestHistogram_Empty(t *testing.T) {
	h := newHistogram("x", "help", nil, nil)
	if h.P50() != 0 || h.Mean() != 0 || h.Count() != 0 || h.Sum() != 0 {
		t.Fatal("expected zero values for an empty histogram")
	}
}

func TestHistogram_SortsUnsortedBuckets(t *testing.T) {
	r := NewRegistry()
	h := r.Histogram("lat", "help", []float64{1.0, 0.1, 0.5}, nil)
	h.Observe(0.2) // falls into the [0.1, 0.5) bucket
	lines := h.seriesLines()
	want := map[string]bool{
		`lat_bucket{le="0.1"} 0`: false,
		`lat_bucket{le="0.5"} 1`: false,
	}
	for _, l := range lines {
		if _, ok := want[l]; ok {
			want[l] = true
		}
	}
	for l, ok := range want {
		if !ok {
			t.Fatalf("expected line %q in:\n%s", l, joinLines(lines))
		}
	}
}

func TestHistogram_ReservoirBounded(t *testing.T) {
	h := newHistogram("x", "help", nil, nil)
	for i := 0; i < reservoirCap*2; i++ {
		h.Observe(float64(i))
	}
	// The reservoir must stay bounded.
	h.mu.Lock()
	n := len(h.reservoir)
	h.mu.Unlock()
	if n != reservoirCap {
		t.Fatalf("expected reservoir capped at %d, got %d", reservoirCap, n)
	}
	// Even sampled, the mean should be close to the true mean (0..2cap-1).
	if h.Count() != reservoirCap*2 {
		t.Fatalf("expected count %d, got %d", reservoirCap*2, h.Count())
	}
}

func approx(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func joinLines(lines []string) string {
	out := ""
	for i, l := range lines {
		if i > 0 {
			out += "\n"
		}
		out += l
	}
	return out
}
