package metrics

import "time"

// StoreMetrics records metrics for the store/search subsystem: search
// latency percentiles, throughput, error rate, and store size.
type StoreMetrics struct {
	searchLatency *Histogram
	searches      *Counter
	searchErrors  *Counter
	uploads       *Counter
	uploadErrors  *Counter
	uploadLatency *Histogram
	size          *Gauge
}

// NewStoreMetrics creates and registers the standard store metrics. Calling
// it multiple times on the same Registry returns the same underlying series.
func NewStoreMetrics(reg *Registry) *StoreMetrics {
	return &StoreMetrics{
		searchLatency: reg.Histogram("recall_search_latency_seconds", "Search latency in seconds.", DefaultLatencyBuckets(), nil),
		searches:      reg.Counter("recall_search_total", "Total number of searches.", nil),
		searchErrors:  reg.Counter("recall_search_errors_total", "Total number of failed searches.", nil),
		uploads:       reg.Counter("recall_upload_total", "Total number of uploads.", nil),
		uploadErrors:  reg.Counter("recall_upload_errors_total", "Total number of failed uploads.", nil),
		uploadLatency: reg.Histogram("recall_upload_latency_seconds", "Upload latency in seconds.", DefaultLatencyBuckets(), nil),
		size:          reg.Gauge("recall_store_size", "Number of chunks in the store.", nil),
	}
}

// RecordSearch records a successful search with its latency.
func (m *StoreMetrics) RecordSearch(d time.Duration) {
	m.searches.Inc()
	m.searchLatency.Observe(d.Seconds())
}

// RecordSearchError records a failed search.
func (m *StoreMetrics) RecordSearchError() { m.searchErrors.Inc() }

// RecordUpload records a successful upload with its latency.
func (m *StoreMetrics) RecordUpload(d time.Duration) {
	m.uploads.Inc()
	m.uploadLatency.Observe(d.Seconds())
}

// RecordUploadError records a failed upload.
func (m *StoreMetrics) RecordUploadError() { m.uploadErrors.Inc() }

// SetSize sets the current store size (number of chunks).
func (m *StoreMetrics) SetSize(n int) { m.size.Set(float64(n)) }

// Size returns the current store size value.
func (m *StoreMetrics) Size() float64 { return m.size.Value() }

// SearchLatencyP50 returns the median search latency in seconds.
func (m *StoreMetrics) SearchLatencyP50() float64 { return m.searchLatency.P50() }

// SearchLatencyP95 returns the 95th percentile search latency in seconds.
func (m *StoreMetrics) SearchLatencyP95() float64 { return m.searchLatency.P95() }

// SearchLatencyP99 returns the 99th percentile search latency in seconds.
func (m *StoreMetrics) SearchLatencyP99() float64 { return m.searchLatency.P99() }

// ErrorRate returns the fraction of searches that failed (0..1).
func (m *StoreMetrics) ErrorRate() float64 {
	total := m.searches.Value() + m.searchErrors.Value()
	if total == 0 {
		return 0
	}
	return m.searchErrors.Value() / total
}

// EmbeddingMetrics records embedding-call latency, throughput, errors, and
// the observed embedding dimension.
type EmbeddingMetrics struct {
	latency *Histogram
	calls   *Counter
	errors  *Counter
	dim     *Gauge
}

// NewEmbeddingMetrics creates and registers the standard embedding metrics.
func NewEmbeddingMetrics(reg *Registry) *EmbeddingMetrics {
	return &EmbeddingMetrics{
		latency: reg.Histogram("recall_embedding_latency_seconds", "Embedding latency in seconds.", DefaultLatencyBuckets(), nil),
		calls:   reg.Counter("recall_embedding_total", "Total number of embedding calls.", nil),
		errors:  reg.Counter("recall_embedding_errors_total", "Total number of failed embedding calls.", nil),
		dim:     reg.Gauge("recall_embedding_dim", "Dimension of the most recent embedding.", nil),
	}
}

// RecordEmbedding records a successful embedding call with its latency and
// output dimension.
func (m *EmbeddingMetrics) RecordEmbedding(d time.Duration, dim int) {
	m.calls.Inc()
	m.latency.Observe(d.Seconds())
	if dim > 0 {
		m.dim.Set(float64(dim))
	}
}

// RecordError records a failed embedding call.
func (m *EmbeddingMetrics) RecordError() { m.errors.Inc() }

// LatencyP50 returns the median embedding latency in seconds.
func (m *EmbeddingMetrics) LatencyP50() float64 { return m.latency.P50() }

// LatencyP95 returns the 95th percentile embedding latency in seconds.
func (m *EmbeddingMetrics) LatencyP95() float64 { return m.latency.P95() }

// LatencyP99 returns the 99th percentile embedding latency in seconds.
func (m *EmbeddingMetrics) LatencyP99() float64 { return m.latency.P99() }

// Dim returns the most recently observed embedding dimension.
func (m *EmbeddingMetrics) Dim() float64 { return m.dim.Value() }

// CacheMetrics records cache hit/miss ratio, evictions, and current size.
type CacheMetrics struct {
	hits      *Counter
	misses    *Counter
	evictions *Counter
	size      *Gauge
}

// NewCacheMetrics creates and registers the standard cache metrics.
func NewCacheMetrics(reg *Registry) *CacheMetrics {
	return &CacheMetrics{
		hits:      reg.Counter("recall_cache_hits_total", "Total number of cache hits.", nil),
		misses:    reg.Counter("recall_cache_misses_total", "Total number of cache misses.", nil),
		evictions: reg.Counter("recall_cache_evictions_total", "Total number of cache evictions.", nil),
		size:      reg.Gauge("recall_cache_size", "Current number of entries in the cache.", nil),
	}
}

// RecordHit records a cache hit.
func (m *CacheMetrics) RecordHit() { m.hits.Inc() }

// RecordMiss records a cache miss.
func (m *CacheMetrics) RecordMiss() { m.misses.Inc() }

// RecordEviction records a cache eviction.
func (m *CacheMetrics) RecordEviction() { m.evictions.Inc() }

// SetSize sets the current number of cached entries.
func (m *CacheMetrics) SetSize(n int) { m.size.Set(float64(n)) }

// Hits returns the total hit count.
func (m *CacheMetrics) Hits() float64 { return m.hits.Value() }

// Misses returns the total miss count.
func (m *CacheMetrics) Misses() float64 { return m.misses.Value() }

// Evictions returns the total eviction count.
func (m *CacheMetrics) Evictions() float64 { return m.evictions.Value() }

// HitRatio returns hits/(hits+misses), or 0 if there have been no lookups.
func (m *CacheMetrics) HitRatio() float64 {
	total := m.hits.Value() + m.misses.Value()
	if total == 0 {
		return 0
	}
	return m.hits.Value() / total
}

// GraphMetrics records knowledge-graph traversal depth, inference counts, and
// entity/relation totals.
type GraphMetrics struct {
	traversals     *Counter
	traversalDepth *Histogram
	inferences     *Counter
	entities       *Gauge
	relations      *Gauge
}

// NewGraphMetrics creates and registers the standard graph metrics.
func NewGraphMetrics(reg *Registry) *GraphMetrics {
	return &GraphMetrics{
		traversals:     reg.Counter("recall_graph_traversals_total", "Total number of graph traversals.", nil),
		traversalDepth: reg.Histogram("recall_graph_traversal_depth", "Depth reached by graph traversals.", []float64{1, 2, 3, 4, 5, 10}, nil),
		inferences:     reg.Counter("recall_graph_inferences_total", "Total number of inferences performed.", nil),
		entities:       reg.Gauge("recall_graph_entities", "Number of entities in the graph.", nil),
		relations:      reg.Gauge("recall_graph_relations", "Number of relations in the graph.", nil),
	}
}

// RecordTraversal records a graph traversal that reached the given depth.
func (m *GraphMetrics) RecordTraversal(depth int) {
	m.traversals.Inc()
	if depth > 0 {
		m.traversalDepth.Observe(float64(depth))
	}
}

// RecordInference records a single performed inference.
func (m *GraphMetrics) RecordInference() { m.inferences.Inc() }

// SetEntities sets the current entity count.
func (m *GraphMetrics) SetEntities(n int) { m.entities.Set(float64(n)) }

// SetRelations sets the current relation count.
func (m *GraphMetrics) SetRelations(n int) { m.relations.Set(float64(n)) }

// Inferences returns the total inference count.
func (m *GraphMetrics) Inferences() float64 { return m.inferences.Value() }

// TraversalDepthP95 returns the 95th percentile traversal depth.
func (m *GraphMetrics) TraversalDepthP95() float64 { return m.traversalDepth.P95() }
