package metrics

import (
	"sync"
	"testing"
)

func TestCounter_IncAndAdd(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("requests", "Total requests.", nil)
	c.Inc()
	c.Add(4)
	if c.Value() != 5 {
		t.Fatalf("expected 5, got %v", c.Value())
	}
	// Negative adds are ignored (counters are monotonic).
	c.Add(-10)
	if c.Value() != 5 {
		t.Fatalf("expected negative add to be ignored, got %v", c.Value())
	}
}

func TestGauge_SetIncDec(t *testing.T) {
	r := NewRegistry()
	g := r.Gauge("temp", "Temperature.", nil)
	g.Set(10)
	g.Inc()
	g.Dec()
	g.Add(-5)
	if g.Value() != 5 {
		t.Fatalf("expected 5, got %v", g.Value())
	}
}

func TestRegistry_GetOrCreate(t *testing.T) {
	r := NewRegistry()
	labels := map[string]string{"ns": "a"}
	c1 := r.Counter("req", "help", labels)
	c2 := r.Counter("req", "help", labels)
	if c1 != c2 {
		t.Fatal("expected the same counter instance for identical name+labels")
	}
	// Different labels -> different series.
	c3 := r.Counter("req", "help", map[string]string{"ns": "b"})
	if c1 == c3 {
		t.Fatal("expected different counter for different labels")
	}
	if r.Len() != 2 {
		t.Fatalf("expected 2 series, got %d", r.Len())
	}
	// Mutating the caller's labels map must not affect the stored series.
	labels["ns"] = "mutated"
	if got := c1.Labels()["ns"]; got != "a" {
		t.Fatalf("expected stored labels to be isolated from caller map, got %q", got)
	}
}

func TestRegistry_GetAndSnapshot(t *testing.T) {
	r := NewRegistry()
	r.Counter("a", "help", nil).Add(3)
	r.Gauge("b", "help", nil).Set(2)
	r.Histogram("c", "help", []float64{1}, nil).Observe(0.5)

	if got := r.Get("a"); len(got) != 1 {
		t.Fatalf("expected 1 series for 'a', got %d", len(got))
	}
	snap := r.Snapshot()
	if snap["a"] != 3 {
		t.Fatalf("expected snapshot a=3, got %v", snap["a"])
	}
	if snap["b"] != 2 {
		t.Fatalf("expected snapshot b=2, got %v", snap["b"])
	}
	if snap["c_count"] != 1 {
		t.Fatalf("expected snapshot c_count=1, got %v", snap["c_count"])
	}
}

func TestCounter_Concurrent(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("n", "help", nil)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				c.Inc()
			}
		}()
	}
	wg.Wait()
	if c.Value() != 16000 {
		t.Fatalf("expected 16000, got %v", c.Value())
	}
}
