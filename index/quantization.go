package index

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// ScalarQuantizer implements 8-bit scalar quantization (SQ8): each
// embedding dimension is mapped to a single byte using a per-dimension
// linear scale fitted during Train. This compresses vectors 4x
// (dim*4 bytes -> dim bytes) at a small, bounded recall cost.
type ScalarQuantizer struct {
	dim    int
	mins   []float32
	maxs   []float32
	ranges []float32 // maxs[i]-mins[i]; 0 marks a constant dimension
	fitted bool
}

// NewScalarQuantizer creates a quantizer for vectors of the given
// dimension. It must be trained (Train) before Quantize is called.
func NewScalarQuantizer(dim int) (*ScalarQuantizer, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("quantizer: dimension must be positive, got %d", dim)
	}
	return &ScalarQuantizer{
		dim:    dim,
		mins:   make([]float32, dim),
		maxs:   make([]float32, dim),
		ranges: make([]float32, dim),
	}, nil
}

// Dimension returns the vector dimension this quantizer was built for.
func (q *ScalarQuantizer) Dimension() int { return q.dim }

// Trained reports whether the quantizer has fitted scales.
func (q *ScalarQuantizer) Trained() bool { return q.fitted }

// Train fits per-dimension min/max scales from the provided vectors.
// Training is idempotent: calling it again refits the scales.
func (q *ScalarQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("quantizer: cannot train on zero vectors")
	}
	for i := 0; i < q.dim; i++ {
		q.mins[i] = 1 << 30
		q.maxs[i] = -(1 << 30)
	}
	for _, v := range vectors {
		if len(v) != q.dim {
			return fmt.Errorf("quantizer: vector length %d does not match dimension %d", len(v), q.dim)
		}
		for i, x := range v {
			if x < q.mins[i] {
				q.mins[i] = x
			}
			if x > q.maxs[i] {
				q.maxs[i] = x
			}
		}
	}
	for i := 0; i < q.dim; i++ {
		q.ranges[i] = q.maxs[i] - q.mins[i]
	}
	q.fitted = true
	return nil
}

// Quantize maps a float vector to one byte per dimension.
func (q *ScalarQuantizer) Quantize(vec []float32) ([]uint8, error) {
	if !q.fitted {
		return nil, fmt.Errorf("quantizer: not trained")
	}
	if len(vec) != q.dim {
		return nil, core.ErrEmbeddingMismatch
	}
	code := make([]uint8, q.dim)
	for i, x := range vec {
		r := q.ranges[i]
		if r == 0 {
			code[i] = 0
			continue
		}
		f := (x - q.mins[i]) / r * 255
		if f < 0 {
			f = 0
		}
		if f > 255 {
			f = 255
		}
		code[i] = uint8(f + 0.5)
	}
	return code, nil
}

// Dequantize maps a byte code back to a float vector.
func (q *ScalarQuantizer) Dequantize(code []uint8) ([]float32, error) {
	if !q.fitted {
		return nil, fmt.Errorf("quantizer: not trained")
	}
	if len(code) != q.dim {
		return nil, core.ErrEmbeddingMismatch
	}
	vec := make([]float32, q.dim)
	for i, c := range code {
		if q.ranges[i] == 0 {
			vec[i] = q.mins[i]
			continue
		}
		vec[i] = q.mins[i] + float32(c)/255*q.ranges[i]
	}
	return vec, nil
}

// MeanAbsError measures the average per-dimension reconstruction error
// over the given vectors. It is a practical measure of quantization loss.
func (q *ScalarQuantizer) MeanAbsError(vectors [][]float32) (float64, error) {
	if !q.fitted {
		return 0, fmt.Errorf("quantizer: not trained")
	}
	var total, n float64
	for _, v := range vectors {
		if len(v) != q.dim {
			return 0, core.ErrEmbeddingMismatch
		}
		code, err := q.Quantize(v)
		if err != nil {
			return 0, err
		}
		back, err := q.Dequantize(code)
		if err != nil {
			return 0, err
		}
		for i := 0; i < q.dim; i++ {
			d := float64(v[i]) - float64(back[i])
			if d < 0 {
				d = -d
			}
			total += d
			n++
		}
	}
	if n == 0 {
		return 0, nil
	}
	return total / n, nil
}

// QuantizedIndex is a memory index that stores 8-bit scalar-quantized
// vectors instead of raw float32s, cutting vector memory 4x. Search
// dequantizes candidate vectors, so results are exact with respect to
// the quantized representation.
//
// The index must be created with an already-trained ScalarQuantizer.
type QuantizedIndex struct {
	ns string
	qz *ScalarQuantizer

	mu     sync.RWMutex
	chunks map[string]*core.Chunk
	codes  map[string][]uint8
}

// NewQuantizedIndex creates a QuantizedIndex over the given trained
// quantizer.
func NewQuantizedIndex(ns string, qz *ScalarQuantizer) (*QuantizedIndex, error) {
	if qz == nil {
		return nil, fmt.Errorf("quantized index: quantizer is required")
	}
	return &QuantizedIndex{
		ns:     ns,
		qz:     qz,
		chunks: make(map[string]*core.Chunk),
		codes:  make(map[string][]uint8),
	}, nil
}

// Add inserts a chunk. The chunk must carry an embedding of the correct
// dimension; the quantizer must be trained.
func (q *QuantizedIndex) Add(_ context.Context, chunk *core.Chunk) error {
	if chunk == nil {
		return core.ErrInvalidChunk
	}
	if len(chunk.Embedding) == 0 {
		return core.ErrInvalidEmbedding
	}
	code, err := q.qz.Quantize(chunk.Embedding)
	if err != nil {
		return err
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.chunks[chunk.ID] = chunk
	q.codes[chunk.ID] = code
	return nil
}

// AddBatch inserts multiple chunks.
func (q *QuantizedIndex) AddBatch(ctx context.Context, chunks []*core.Chunk) error {
	for _, c := range chunks {
		if err := q.Add(ctx, c); err != nil {
			return err
		}
	}
	return nil
}

// Delete removes a chunk from the index.
func (q *QuantizedIndex) Delete(_ context.Context, id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if _, ok := q.chunks[id]; !ok {
		return core.ErrNotFound
	}
	delete(q.chunks, id)
	delete(q.codes, id)
	return nil
}

// Search finds the most similar chunks to the query vector using
// dequantized candidates.
func (q *QuantizedIndex) Search(_ context.Context, query []float32, opts SearchOptions) ([]SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	type scored struct {
		chunk *core.Chunk
		score float64
	}
	q.mu.RLock()
	defer q.mu.RUnlock()

	var results []scored
	for id, chunk := range q.chunks {
		if !matchesAllFilters(chunk, opts.Filters) {
			continue
		}
		vec, err := q.qz.Dequantize(q.codes[id])
		if err != nil {
			return nil, err
		}
		s := embedder.CosineSimilarity(query, vec)
		if s < opts.MinScore {
			continue
		}
		results = append(results, scored{chunk, s})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].score != results[j].score {
			return results[i].score > results[j].score
		}
		return results[i].chunk.ID < results[j].chunk.ID
	})
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	out := make([]SearchResult, len(results))
	for i, r := range results {
		out[i] = SearchResult{Chunk: r.chunk, Score: r.score}
	}
	return out, nil
}

// Count returns the number of indexed chunks.
func (q *QuantizedIndex) Count() int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	return len(q.chunks)
}

// Dimension returns the vector dimension.
func (q *QuantizedIndex) Dimension() int { return q.qz.Dimension() }

// Namespace returns the namespace of this index.
func (q *QuantizedIndex) Namespace() string { return q.ns }

// MemoryBytes reports how many bytes the quantized vectors occupy.
func (q *QuantizedIndex) MemoryBytes() int64 {
	q.mu.RLock()
	defer q.mu.RUnlock()
	var b int64
	for _, c := range q.codes {
		b += int64(len(c))
	}
	return b
}
