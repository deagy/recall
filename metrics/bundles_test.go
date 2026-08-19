package metrics

import (
	"testing"
	"time"
)

func TestStoreMetrics(t *testing.T) {
	r := NewRegistry()
	m := NewStoreMetrics(r)
	m.RecordSearch(10 * time.Millisecond)
	m.RecordSearch(50 * time.Millisecond)
	m.RecordSearchError()
	m.SetSize(123)

	if m.Size() != 123 {
		t.Fatalf("expected size 123, got %v", m.Size())
	}
	// p50 of {0.01, 0.05}: nearest rank ceil(0.5*2)=1 -> 0.01
	if !approx(m.SearchLatencyP50(), 0.01) {
		t.Fatalf("expected p50 0.01, got %v", m.SearchLatencyP50())
	}
	// Error rate = 1 error / (2 ok + 1 err) = 1/3
	rate := m.ErrorRate()
	if rate < 0.33 || rate > 0.34 {
		t.Fatalf("expected error rate ~0.333, got %v", rate)
	}
}

func TestStoreMetrics_NoSearches(t *testing.T) {
	r := NewRegistry()
	m := NewStoreMetrics(r)
	if m.ErrorRate() != 0 {
		t.Fatalf("expected 0 error rate with no searches, got %v", m.ErrorRate())
	}
	if m.SearchLatencyP99() != 0 {
		t.Fatalf("expected 0 latency with no samples, got %v", m.SearchLatencyP99())
	}
}

func TestCacheMetrics_HitRatio(t *testing.T) {
	r := NewRegistry()
	m := NewCacheMetrics(r)
	m.RecordHit()
	m.RecordHit()
	m.RecordMiss()
	m.RecordEviction()
	m.SetSize(42)

	if !approx(m.HitRatio(), 2.0/3.0) {
		t.Fatalf("expected hit ratio 2/3, got %v", m.HitRatio())
	}
	if m.Evictions() != 1 {
		t.Fatalf("expected 1 eviction, got %v", m.Evictions())
	}
}

func TestCacheMetrics_NoLookups(t *testing.T) {
	r := NewRegistry()
	m := NewCacheMetrics(r)
	if m.HitRatio() != 0 {
		t.Fatalf("expected 0 hit ratio with no lookups, got %v", m.HitRatio())
	}
}

func TestGraphMetrics(t *testing.T) {
	r := NewRegistry()
	m := NewGraphMetrics(r)
	m.RecordTraversal(3)
	m.RecordTraversal(5)
	m.RecordInference()
	m.SetEntities(10)
	m.SetRelations(20)

	if m.Inferences() != 1 {
		t.Fatalf("expected 1 inference, got %v", m.Inferences())
	}
	// p95 of {3,5}: nearest rank ceil(0.95*2)=2 -> 5
	if !approx(m.TraversalDepthP95(), 5) {
		t.Fatalf("expected depth p95 5, got %v", m.TraversalDepthP95())
	}
}

func TestEmbeddingMetrics(t *testing.T) {
	r := NewRegistry()
	m := NewEmbeddingMetrics(r)
	m.RecordEmbedding(5*time.Millisecond, 384)
	if m.Dim() != 384 {
		t.Fatalf("expected dim 384, got %v", m.Dim())
	}
	if m.LatencyP50() <= 0 {
		t.Fatalf("expected positive latency, got %v", m.LatencyP50())
	}
}

func TestBundles_ShareUnderlyingSeries(t *testing.T) {
	r := NewRegistry()
	a := NewStoreMetrics(r)
	b := NewStoreMetrics(r)
	a.searches.Inc()
	if b.searches.Value() != 1 {
		t.Fatalf("expected bundles to share the same underlying series, got %v", b.searches.Value())
	}
}
