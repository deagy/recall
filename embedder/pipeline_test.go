package embedder

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/deagy/recall/cache"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeEmbedder is a controllable embedder for pipeline and cache tests.
type fakeEmbedder struct {
	dim   int
	err   error
	calls int32
	// batchSeen records the texts passed to each EmbedBatch call.
	batchSeen [][]string
}

func (f *fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	atomic.AddInt32(&f.calls, 1)
	if f.err != nil {
		return nil, f.err
	}
	return vecOf(f.dim, 1), nil
}

func (f *fakeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	atomic.AddInt32(&f.calls, 1)
	if f.err != nil {
		return nil, f.err
	}
	f.batchSeen = append(f.batchSeen, texts)
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = vecOf(f.dim, 1)
	}
	return out, nil
}

func (f *fakeEmbedder) Dimension() int { return f.dim }

func TestNewPipeline_Validation(t *testing.T) {
	_, err := NewPipeline()
	require.Error(t, err)

	_, err = NewPipeline(NewMockEmbedder(384), NewMockEmbedder(512))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "share dimension")

	p, err := NewPipeline(NewMockEmbedder(384), NewMockEmbedder(384))
	require.NoError(t, err)
	assert.Equal(t, 384, p.Dimension())
}

func TestPipeline_Fallback(t *testing.T) {
	primary := &fakeEmbedder{dim: 32, err: errors.New("primary down")}
	backup := NewMockEmbedder(32)

	p, err := NewPipeline(primary, backup)
	require.NoError(t, err)

	vec, err := p.Embed(context.Background(), "hello")
	require.NoError(t, err)
	expected, err := backup.Embed(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, expected, vec)
	assert.Equal(t, int32(1), atomic.LoadInt32(&primary.calls), "primary should have been tried")
}

func TestPipeline_AllFail(t *testing.T) {
	first := &fakeEmbedder{dim: 32, err: errors.New("boom-1")}
	second := &fakeEmbedder{dim: 32, err: errors.New("boom-2")}

	p, err := NewPipeline(first, second)
	require.NoError(t, err)

	_, err = p.Embed(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "all 2 pipeline embedders failed")
	assert.Contains(t, err.Error(), "boom-2")

	_, err = p.EmbedBatch(context.Background(), []string{"a", "b"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "boom-2")
}

func TestPipeline_EmbedBatch_Fallback(t *testing.T) {
	primary := &fakeEmbedder{dim: 16, err: errors.New("batch failed")}
	backup := &fakeEmbedder{dim: 16}

	p, err := NewPipeline(primary, backup)
	require.NoError(t, err)

	vecs, err := p.EmbedBatch(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, vecs, 2)
	assert.Equal(t, vecOf(16, 1), vecs[0])
	// The backup received the full batch.
	require.Len(t, backup.batchSeen, 1)
	assert.Equal(t, []string{"a", "b"}, backup.batchSeen[0])
}

func TestCachingEmbedder_Embed_Hit(t *testing.T) {
	inner := &fakeEmbedder{dim: 8}
	c := NewCachingEmbedder(inner, cache.NewEmbeddingCache(100), 0)
	ctx := context.Background()

	vec1, err := c.Embed(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, vecOf(8, 1), vec1)
	assert.Equal(t, int32(1), atomic.LoadInt32(&inner.calls))

	vec2, err := c.Embed(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, vec1, vec2)
	assert.Equal(t, int32(1), atomic.LoadInt32(&inner.calls), "second call must be a cache hit")

	stats := c.Stats()
	assert.Equal(t, 1, stats.Hits)
	assert.Equal(t, 1, stats.Misses)
	assert.Equal(t, 8, c.Dimension())
}

func TestCachingEmbedder_EmbedBatch_PartialHit(t *testing.T) {
	inner := &fakeEmbedder{dim: 4}
	c := NewCachingEmbedder(inner, cache.NewEmbeddingCache(100), time.Minute)
	ctx := context.Background()

	// Warm the cache for "x".
	_, err := c.Embed(ctx, "x")
	require.NoError(t, err)

	// Batch with a mix of cached and uncached texts: only "y" should be
	// sent to the inner embedder.
	vecs, err := c.EmbedBatch(ctx, []string{"x", "y"})
	require.NoError(t, err)
	require.Len(t, vecs, 2)
	require.Len(t, inner.batchSeen, 1)
	assert.Equal(t, []string{"y"}, inner.batchSeen[0])

	// Now everything is cached.
	_, err = c.EmbedBatch(ctx, []string{"x", "y"})
	require.NoError(t, err)
	assert.Len(t, inner.batchSeen, 1, "fully cached batch must not call the inner embedder")
}

func TestCachingEmbedder_TTLExpiry(t *testing.T) {
	inner := &fakeEmbedder{dim: 4}
	c := NewCachingEmbedder(inner, cache.NewEmbeddingCache(100), time.Millisecond)
	ctx := context.Background()

	_, err := c.Embed(ctx, "hello")
	require.NoError(t, err)
	time.Sleep(5 * time.Millisecond)

	_, err = c.Embed(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, int32(2), atomic.LoadInt32(&inner.calls), "expired entry must be re-embedded")
}

func TestCachingEmbedder_InnerErrorNotCached(t *testing.T) {
	var fail int32
	inner := &flakyEmbedder{dim: 4, fail: &fail}
	c := NewCachingEmbedder(inner, cache.NewEmbeddingCache(100), time.Minute)
	ctx := context.Background()

	atomic.StoreInt32(&fail, 1)
	_, err := c.Embed(ctx, "hello")
	require.Error(t, err)

	atomic.StoreInt32(&fail, 0)
	vec, err := c.Embed(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, vecOf(4, 1), vec)
	// Cached after success; subsequent call is a hit.
	_, err = c.Embed(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, int32(2), inner.calls)
}

// flakyEmbedder fails while *fail is non-zero.
type flakyEmbedder struct {
	dim   int
	fail  *int32
	calls int32
}

func (f *flakyEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	atomic.AddInt32(&f.calls, 1)
	if atomic.LoadInt32(f.fail) != 0 {
		return nil, fmt.Errorf("transient failure")
	}
	return vecOf(f.dim, 1), nil
}

func (f *flakyEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		vec, err := f.Embed(ctx, texts[i])
		if err != nil {
			return nil, err
		}
		out[i] = vec
	}
	return out, nil
}

func (f *flakyEmbedder) Dimension() int { return f.dim }

func TestAutoDimension(t *testing.T) {
	e := NewMockEmbedder(32)
	dim, err := AutoDimension(context.Background(), e, "sample")
	require.NoError(t, err)
	assert.Equal(t, 32, dim)

	_, err = AutoDimension(context.Background(), &fakeEmbedder{err: errors.New("nope")}, "sample")
	require.Error(t, err)

	empty := &emptyEmbedder{}
	_, err = AutoDimension(context.Background(), empty, "sample")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty embedding")
}

// emptyEmbedder returns a zero-length vector.
type emptyEmbedder struct{}

func (e *emptyEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return []float32{}, nil
}

func (e *emptyEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return nil, nil
}

func (e *emptyEmbedder) Dimension() int { return 0 }
