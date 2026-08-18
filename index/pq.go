package index

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/deagy/recall/core"
)

// ProductQuantizer implements product quantization (PQ): a vector of
// dimension dim is split into m contiguous sub-vectors of dimension
// dim/m, and each sub-vector is encoded as the index of its nearest
// centroid in a per-subspace codebook of k centroids (fitted with
// k-means++). A full vector therefore compresses from dim*4 bytes to m
// bytes (e.g. 768-dim with m=96 -> 96 bytes, an 8x reduction), and
// distances at search time are computed with addable quantization
// (ADC) using precomputed distance tables.
type ProductQuantizer struct {
	dim    int
	m      int // number of subspaces (codes per vector)
	k      int // codebook size (usually 256)
	subDim int
	seed   int64

	codebooks [][]float32 // [m] flattened centroids: each k*subDim floats
	fitted    bool
}

// NewProductQuantizer creates a PQ configuration. dim must be divisible
// by m, and k must be at least 2.
func NewProductQuantizer(dim, m, k int) (*ProductQuantizer, error) {
	if dim <= 0 || m <= 0 {
		return nil, fmt.Errorf("pq: dim and m must be positive (got dim=%d m=%d)", dim, m)
	}
	if k < 2 {
		return nil, fmt.Errorf("pq: codebook size k must be >= 2 (got %d)", k)
	}
	if dim%m != 0 {
		return nil, fmt.Errorf("pq: dimension %d not divisible by m=%d", dim, m)
	}
	return &ProductQuantizer{
		dim:    dim,
		m:      m,
		k:      k,
		subDim: dim / m,
		seed:   1,
	}, nil
}

// WithSeed sets the RNG seed used for k-means++ initialization, making
// training deterministic for a given seed.
func (p *ProductQuantizer) WithSeed(seed int64) *ProductQuantizer {
	p.seed = seed
	return p
}

// Dimensions reports the configured dim and m (subspaces).
func (p *ProductQuantizer) Dimensions() (dim, m int) { return p.dim, p.m }

// Trained reports whether codebooks are fitted.
func (p *ProductQuantizer) Trained() bool { return p.fitted }

// Train fits per-subspace codebooks via k-means++ over the provided
// vectors.
func (p *ProductQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("pq: cannot train on zero vectors")
	}
	if len(vectors) < p.k {
		return fmt.Errorf("pq: need at least k=%d training vectors, got %d", p.k, len(vectors))
	}
	for _, v := range vectors {
		if len(v) != p.dim {
			return fmt.Errorf("pq: vector length %d does not match dimension %d", len(v), p.dim)
		}
	}
	rng := rand.New(rand.NewSource(p.seed))
	p.codebooks = make([][]float32, p.m)
	for s := 0; s < p.m; s++ {
		sub := make([][]float32, len(vectors))
		for i, v := range vectors {
			sub[i] = v[s*p.subDim : (s+1)*p.subDim]
		}
		p.codebooks[s] = kmeansPlusPlus(sub, p.k, 25, rng)
	}
	p.fitted = true
	return nil
}

// Encode maps a vector to m code bytes (one per subspace).
func (p *ProductQuantizer) Encode(vec []float32) ([]uint8, error) {
	if !p.fitted {
		return nil, fmt.Errorf("pq: not trained")
	}
	if len(vec) != p.dim {
		return nil, fmt.Errorf("pq: vector length %d does not match dimension %d", len(vec), p.dim)
	}
	code := make([]uint8, p.m)
	for s := 0; s < p.m; s++ {
		sub := vec[s*p.subDim : (s+1)*p.subDim]
		best, bestD := 0, math.MaxFloat64
		cb := p.codebooks[s]
		for c := 0; c < p.k; c++ {
			d := l2sq(sub, cb[c*p.subDim:(c+1)*p.subDim])
			if d < bestD {
				bestD, best = d, c
			}
		}
		code[s] = uint8(best)
	}
	return code, nil
}

// Decode reconstructs the full-length vector from a code by
// concatenating the chosen centroids.
func (p *ProductQuantizer) Decode(code []uint8) ([]float32, error) {
	if !p.fitted {
		return nil, fmt.Errorf("pq: not trained")
	}
	if len(code) != p.m {
		return nil, fmt.Errorf("pq: code length %d does not match m=%d", len(code), p.m)
	}
	vec := make([]float32, p.dim)
	for s := 0; s < p.m; s++ {
		cb := p.codebooks[s]
		copy(vec[s*p.subDim:(s+1)*p.subDim], cb[int(code[s])*p.subDim:(int(code[s])+1)*p.subDim])
	}
	return vec, nil
}

// DistanceTable precomputes, for a query vector, the squared L2 distance
// from each query sub-vector to every centroid of the matching subspace.
// Search then scores a stored code by summing one table entry per
// subspace — O(m) instead of O(dim).
func (p *ProductQuantizer) DistanceTable(query []float32) ([][]float64, error) {
	if !p.fitted {
		return nil, fmt.Errorf("pq: not trained")
	}
	if len(query) != p.dim {
		return nil, fmt.Errorf("pq: query length %d does not match dimension %d", len(query), p.dim)
	}
	tables := make([][]float64, p.m)
	for s := 0; s < p.m; s++ {
		sub := query[s*p.subDim : (s+1)*p.subDim]
		table := make([]float64, p.k)
		cb := p.codebooks[s]
		for c := 0; c < p.k; c++ {
			table[c] = l2sq(sub, cb[c*p.subDim:(c+1)*p.subDim])
		}
		tables[s] = table
	}
	return tables, nil
}

// l2sq returns the squared L2 distance between two equal-length vectors.
func l2sq(a, b []float32) float64 {
	var d float64
	for i := range a {
		x := float64(a[i]) - float64(b[i])
		d += x * x
	}
	return d
}

func minL2sq(v []float32, centroids [][]float32) float64 {
	best := math.MaxFloat64
	for _, c := range centroids {
		d := l2sq(v, c)
		if d < best {
			best = d
		}
	}
	return best
}

// kmeansPlusPlus runs k-means with k-means++ seeding and returns k
// centroids as flattened float32s (k * len(data[0])).
func kmeansPlusPlus(data [][]float32, k, maxIters int, rng *rand.Rand) []float32 {
	d := len(data[0])
	if k > len(data) {
		k = len(data)
	}
	centroids := make([][]float32, k)
	// Seed the first centroid uniformly at random.
	centroids[0] = append([]float32(nil), data[rng.Intn(len(data))]...)
	for c := 1; c < k; c++ {
		dists := make([]float64, len(data))
		var total float64
		for i, v := range data {
			dists[i] = minL2sq(v, centroids[:c])
			total += dists[i]
		}
		pick := rng.Float64() * total
		var acc float64
		choice := len(data) - 1
		for i, dist := range dists {
			acc += dist
			if acc >= pick {
				choice = i
				break
			}
		}
		centroids[c] = append([]float32(nil), data[choice]...)
	}

	// Lloyd iterations.
	assignments := make([]int, len(data))
	for iter := 0; iter < maxIters; iter++ {
		changed := false
		for i, v := range data {
			best, bestD := 0, math.MaxFloat64
			for c, cen := range centroids {
				dist := l2sq(v, cen)
				if dist < bestD {
					bestD, best = dist, c
				}
			}
			if assignments[i] != best {
				assignments[i] = best
				changed = true
			}
		}
		sums := make([][]float32, k)
		counts := make([]int, k)
		for i, v := range data {
			c := assignments[i]
			if sums[c] == nil {
				sums[c] = make([]float32, d)
			}
			for j := 0; j < d; j++ {
				sums[c][j] += v[j]
			}
			counts[c]++
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Reseed an empty cluster with a random data point.
				centroids[c] = append([]float32(nil), data[rng.Intn(len(data))]...)
				continue
			}
			for j := 0; j < d; j++ {
				centroids[c][j] /= float32(counts[c])
			}
		}
		if !changed {
			break
		}
	}

	out := make([]float32, 0, k*d)
	for _, cen := range centroids {
		out = append(out, cen...)
	}
	return out
}

// PQIndex is a memory index that stores product-quantized codes instead
// of raw float32 vectors. Vectors are L2-normalized before encoding so
// ADC distances correspond to cosine similarity: for unit vectors,
// cos(a, b) = 1 - ||a-b||^2 / 2.
//
// The index must be created with an already-trained ProductQuantizer.
type PQIndex struct {
	ns string
	pq *ProductQuantizer

	mu     sync.RWMutex
	chunks map[string]*core.Chunk
	codes  map[string][]uint8
}

// NewPQIndex creates a PQIndex over the given trained quantizer.
func NewPQIndex(ns string, pq *ProductQuantizer) (*PQIndex, error) {
	if pq == nil {
		return nil, fmt.Errorf("pq index: quantizer is required")
	}
	return &PQIndex{
		ns:     ns,
		pq:     pq,
		chunks: make(map[string]*core.Chunk),
		codes:  make(map[string][]uint8),
	}, nil
}

// Add inserts a chunk. The chunk must carry a non-empty embedding; the
// quantizer must be trained.
func (p *PQIndex) Add(_ context.Context, chunk *core.Chunk) error {
	if chunk == nil {
		return core.ErrInvalidChunk
	}
	if len(chunk.Embedding) == 0 {
		return core.ErrInvalidEmbedding
	}
	code, err := p.pq.Encode(normalizedCopy(chunk.Embedding))
	if err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.chunks[chunk.ID] = chunk
	p.codes[chunk.ID] = code
	return nil
}

// AddBatch inserts multiple chunks.
func (p *PQIndex) AddBatch(ctx context.Context, chunks []*core.Chunk) error {
	for _, c := range chunks {
		if err := p.Add(ctx, c); err != nil {
			return err
		}
	}
	return nil
}

// Delete removes a chunk from the index.
func (p *PQIndex) Delete(_ context.Context, id string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.chunks[id]; !ok {
		return core.ErrNotFound
	}
	delete(p.chunks, id)
	delete(p.codes, id)
	return nil
}

// Search scores stored codes against the query with addable quantization:
// one distance table is built per query, then each code is scored in O(m).
func (p *PQIndex) Search(_ context.Context, query []float32, opts SearchOptions) ([]SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	tables, err := p.pq.DistanceTable(normalizedCopy(query))
	if err != nil {
		return nil, err
	}
	type scored struct {
		chunk *core.Chunk
		score float64
	}
	p.mu.RLock()
	defer p.mu.RUnlock()

	var results []scored
	for id, chunk := range p.chunks {
		if !matchesAllFilters(chunk, opts.Filters) {
			continue
		}
		code := p.codes[id]
		var d2 float64
		for s := 0; s < p.pq.m; s++ {
			d2 += tables[s][code[s]]
		}
		// Unit vectors: cosine = 1 - d2/2.
		sim := 1 - d2/2
		if sim > 1 {
			sim = 1
		}
		if sim < -1 {
			sim = -1
		}
		if sim < opts.MinScore {
			continue
		}
		results = append(results, scored{chunk, sim})
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
func (p *PQIndex) Count() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.chunks)
}

// Dimension returns the vector dimension.
func (p *PQIndex) Dimension() int { return p.pq.dim }

// Namespace returns the namespace of this index.
func (p *PQIndex) Namespace() string { return p.ns }

// MemoryBytes reports how many bytes the quantized codes occupy.
func (p *PQIndex) MemoryBytes() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	var b int64
	for _, c := range p.codes {
		b += int64(len(c))
	}
	return b
}

// normalizedCopy returns an L2-normalized copy of v. Zero vectors are
// returned unchanged.
func normalizedCopy(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	var norm float64
	for _, x := range out {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range out {
		out[i] *= inv
	}
	return out
}
