package embedder

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/deagy/recall/cache"
)

// Pipeline chains embedders and returns the first successful result,
// enabling failover between providers (e.g. a paid API with a local model
// as backup). All embedders must produce the same dimension, otherwise a
// store's index would silently mix incompatible vectors when failover
// kicks in.
type Pipeline struct {
	embedders []Embedder
}

// NewPipeline creates a Pipeline from the given embedders, tried in order.
func NewPipeline(embedders ...Embedder) (*Pipeline, error) {
	if len(embedders) == 0 {
		return nil, fmt.Errorf("embedder: pipeline requires at least one embedder")
	}
	dim := embedders[0].Dimension()
	if dim > 0 {
		for _, e := range embedders[1:] {
			if d := e.Dimension(); d > 0 && d != dim {
				return nil, fmt.Errorf("embedder: pipeline embedders must share dimension %d, got %d", dim, d)
			}
		}
	}
	return &Pipeline{embedders: embedders}, nil
}

// Dimension returns the embedding dimension of this pipeline (the first
// embedder's dimension).
func (p *Pipeline) Dimension() int {
	return p.embedders[0].Dimension()
}

// Embed converts a single text using the first embedder that succeeds.
func (p *Pipeline) Embed(ctx context.Context, text string) ([]float32, error) {
	var lastErr error
	for _, e := range p.embedders {
		vec, err := e.Embed(ctx, text)
		if err == nil {
			return vec, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("embedder: all %d pipeline embedders failed; last error: %w", len(p.embedders), lastErr)
}

// EmbedBatch converts multiple texts using the first embedder that
// succeeds for the whole batch.
func (p *Pipeline) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	var lastErr error
	for _, e := range p.embedders {
		vecs, err := e.EmbedBatch(ctx, texts)
		if err == nil {
			return vecs, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("embedder: all %d pipeline embedders failed; last error: %w", len(p.embedders), lastErr)
}

// CachingEmbedder wraps another embedder with an embedding cache to avoid
// redundant (and often paid) API calls for repeated texts.
type CachingEmbedder struct {
	inner  Embedder
	cache  *cache.EmbeddingCache
	ttl    time.Duration
	hits   int64
	misses int64
}

// NewCachingEmbedder wraps inner with the given cache. A non-positive TTL
// falls back to the cache package's default TTL.
func NewCachingEmbedder(inner Embedder, c *cache.EmbeddingCache, ttl time.Duration) *CachingEmbedder {
	if ttl <= 0 {
		ttl = cache.DefaultCacheConfig().DefaultTTL
	}
	return &CachingEmbedder{inner: inner, cache: c, ttl: ttl}
}

// Embed returns the cached vector when available, otherwise delegates to
// the wrapped embedder and stores the result.
func (c *CachingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if vec, ok := c.cache.Get(text); ok {
		atomic.AddInt64(&c.hits, 1)
		return vec, nil
	}
	atomic.AddInt64(&c.misses, 1)
	vec, err := c.inner.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	c.cache.Set(text, vec, c.ttl)
	return vec, nil
}

// EmbedBatch serves cached vectors where available and fetches only the
// missing texts from the wrapped embedder in a single batch call.
func (c *CachingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	n := len(texts)
	if n == 0 {
		return nil, nil
	}
	out := make([][]float32, n)
	var missingIdx []int
	var missingTexts []string
	for i, text := range texts {
		if vec, ok := c.cache.Get(text); ok {
			atomic.AddInt64(&c.hits, 1)
			out[i] = vec
		} else {
			atomic.AddInt64(&c.misses, 1)
			missingIdx = append(missingIdx, i)
			missingTexts = append(missingTexts, text)
		}
	}
	if len(missingIdx) == 0 {
		return out, nil
	}
	vecs, err := c.inner.EmbedBatch(ctx, missingTexts)
	if err != nil {
		return nil, err
	}
	for j, i := range missingIdx {
		out[i] = vecs[j]
		c.cache.Set(texts[i], vecs[j], c.ttl)
	}
	return out, nil
}

// Dimension returns the wrapped embedder's dimension.
func (c *CachingEmbedder) Dimension() int {
	return c.inner.Dimension()
}

// Stats returns the underlying cache statistics merged with the hit/miss
// counts observed by this embedder (the LRU layer itself does not track
// lookups).
func (c *CachingEmbedder) Stats() cache.CacheStats {
	stats := c.cache.Stats()
	stats.Hits = int(atomic.LoadInt64(&c.hits))
	stats.Misses = int(atomic.LoadInt64(&c.misses))
	return stats
}

// AutoDimension determines an embedder's output dimension by embedding a
// sample text. Useful for providers whose dimension is not well-known up
// front (e.g. arbitrary Ollama models).
func AutoDimension(ctx context.Context, e Embedder, sample string) (int, error) {
	vec, err := e.Embed(ctx, sample)
	if err != nil {
		return 0, err
	}
	if len(vec) == 0 {
		return 0, fmt.Errorf("embedder: empty embedding for sample text")
	}
	return len(vec), nil
}
