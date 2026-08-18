package index

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/deagy/recall/core"
)

// randUnitVector returns a deterministic pseudo-random unit vector.
func randUnitVector(seed int64, dim int) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, dim)
	var norm float64
	for i := range v {
		v[i] = float32(rng.Float64()*2 - 1)
		norm += float64(v[i]) * float64(v[i])
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func testChunk(id string, emb []float32) *core.Chunk {
	return &core.Chunk{ID: id, Content: "content " + id, Embedding: emb}
}

func TestScalarQuantizer_RoundTrip(t *testing.T) {
	const dim = 64
	qz, err := NewScalarQuantizer(dim)
	if err != nil {
		t.Fatal(err)
	}
	if qz.Trained() {
		t.Fatal("fresh quantizer should not be trained")
	}
	if _, err := qz.Quantize(randUnitVector(1, dim)); err == nil {
		t.Fatal("quantize before Train should fail")
	}

	vectors := make([][]float32, 50)
	for i := range vectors {
		vectors[i] = randUnitVector(int64(i), dim)
	}
	if err := qz.Train(vectors); err != nil {
		t.Fatal(err)
	}
	if !qz.Trained() {
		t.Fatal("trained flag not set")
	}

	// Reconstruction error must stay within one quantization step.
	mae, err := qz.MeanAbsError(vectors)
	if err != nil {
		t.Fatal(err)
	}
	maxStep := float64(1) / 255 * 2.1 // unit-vector ranges are <= ~2.1
	if mae > maxStep {
		t.Fatalf("mean abs error %f exceeds one quantization step %f", mae, maxStep)
	}

	code, err := qz.Quantize(vectors[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(code) != dim {
		t.Fatalf("code length %d != %d", len(code), dim)
	}
	back, err := qz.Dequantize(code)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < dim; i++ {
		if d := math.Abs(float64(vectors[0][i]) - float64(back[i])); d > maxStep {
			t.Fatalf("dim %d deviates by %f", i, d)
		}
	}
}

func TestScalarQuantizer_Errors(t *testing.T) {
	if _, err := NewScalarQuantizer(0); err == nil {
		t.Fatal("dim 0 should fail")
	}
	qz, _ := NewScalarQuantizer(4)
	if err := qz.Train(nil); err == nil {
		t.Fatal("empty training set should fail")
	}
	if err := qz.Train([][]float32{{1, 2, 3}}); err == nil {
		t.Fatal("wrong-length vector should fail")
	}
	if _, err := qz.Quantize([]float32{1, 2}); err == nil {
		t.Fatal("wrong-length quantize should fail")
	}
}

func TestScalarQuantizer_ConstantDimension(t *testing.T) {
	// A constant dimension (range 0) must not divide by zero.
	qz, _ := NewScalarQuantizer(3)
	if err := qz.Train([][]float32{{1, 5, 2}, {1, 5, 9}}); err != nil {
		t.Fatal(err)
	}
	code, err := qz.Quantize([]float32{1, 5, 7})
	if err != nil {
		t.Fatal(err)
	}
	back, err := qz.Dequantize(code)
	if err != nil {
		t.Fatal(err)
	}
	if back[0] != 1 {
		t.Fatalf("constant dim should dequantize exactly, got %f", back[0])
	}
}

func TestQuantizedIndex_SearchMatchesMemory(t *testing.T) {
	const dim = 32
	vectors := make([][]float32, 40)
	for i := range vectors {
		vectors[i] = randUnitVector(int64(i), dim)
	}
	qz, _ := NewScalarQuantizer(dim)
	if err := qz.Train(vectors); err != nil {
		t.Fatal(err)
	}
	qi, err := NewQuantizedIndex("qtest", qz)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewQuantizedIndex("qtest", nil); err == nil {
		t.Fatal("nil quantizer should fail")
	}

	mem := NewMemoryIndex("qtest", dim)
	ctx := context.Background()
	for i, v := range vectors {
		c := testChunk(string(rune('a'+i%26))+string(rune('0'+i/26)), v)
		if err := qi.Add(ctx, c); err != nil {
			t.Fatal(err)
		}
		if err := mem.Add(ctx, c); err != nil {
			t.Fatal(err)
		}
	}
	if qi.Count() != 40 || qi.Dimension() != dim || qi.Namespace() != "qtest" {
		t.Fatalf("count/dim/ns wrong: %d %d %s", qi.Count(), qi.Dimension(), qi.Namespace())
	}
	// 40 chunks * dim bytes each (quantized), not dim*4.
	if got := qi.MemoryBytes(); got != int64(40*dim) {
		t.Fatalf("MemoryBytes = %d, want %d", got, 40*dim)
	}

	query := randUnitVector(999, dim)
	qRes, err := qi.Search(ctx, query, DefaultSearchOptions(5))
	if err != nil {
		t.Fatal(err)
	}
	mRes, err := mem.Search(ctx, query, DefaultSearchOptions(5))
	if err != nil {
		t.Fatal(err)
	}
	if len(qRes) != 5 {
		t.Fatalf("want 5 results, got %d", len(qRes))
	}
	// Top-1 should agree with the exact (non-quantized) index.
	if qRes[0].Chunk.ID != mRes[0].Chunk.ID {
		t.Fatalf("top-1 mismatch: quantized=%s memory=%s", qRes[0].Chunk.ID, mRes[0].Chunk.ID)
	}

	// Delete works and shrinks memory accounting.
	if err := qi.Delete(ctx, qRes[0].Chunk.ID); err != nil {
		t.Fatal(err)
	}
	if qi.Count() != 39 || qi.MemoryBytes() != int64(39*dim) {
		t.Fatalf("post-delete state wrong: %d %d", qi.Count(), qi.MemoryBytes())
	}
	if err := qi.Delete(ctx, "missing"); err != core.ErrNotFound {
		t.Fatalf("delete missing should be ErrNotFound, got %v", err)
	}
}

func TestQuantizedIndex_AddErrors(t *testing.T) {
	qz, _ := NewScalarQuantizer(4)
	qi, _ := NewQuantizedIndex("ns", qz)
	ctx := context.Background()
	if err := qi.Add(ctx, nil); err != core.ErrInvalidChunk {
		t.Fatalf("nil chunk: %v", err)
	}
	if err := qi.Add(ctx, testChunk("x", nil)); err != core.ErrInvalidEmbedding {
		t.Fatalf("nil embedding: %v", err)
	}
	// Untrained quantizer.
	if err := qi.Add(ctx, testChunk("x", []float32{1, 2, 3, 4})); err == nil {
		t.Fatal("add on untrained quantizer should fail")
	}
}

// jitterCluster returns a unit vector near a deterministic cluster
// center with small per-index jitter.
func jitterCluster(t *testing.T, cluster, seed, dim int) []float32 {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(cluster*10000 + seed)))
	v := make([]float32, dim)
	for i := range v {
		v[i] = 0.3 + float32(rng.Float64()*0.1)
		if i < dim/2 && cluster%2 == 0 {
			v[i] = -v[i]
		}
	}
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func TestProductQuantizer_RoundTrip(t *testing.T) {
	const dim, m, k = 16, 4, 16
	pq, err := NewProductQuantizer(dim, m, k)
	if err != nil {
		t.Fatal(err)
	}
	if pq.Trained() {
		t.Fatal("fresh PQ should not be trained")
	}
	vectors := make([][]float32, 200)
	for i := range vectors {
		vectors[i] = randUnitVector(int64(i), dim)
	}
	if err := pq.Train(vectors); err != nil {
		t.Fatal(err)
	}
	if !pq.Trained() {
		t.Fatal("trained flag not set")
	}
	if _, err := pq.Encode(vectors[0]); err != nil {
		t.Fatal(err)
	}
	code, _ := pq.Encode(vectors[0])
	if len(code) != m {
		t.Fatalf("code length %d != m=%d", len(code), m)
	}
	decoded, err := pq.Decode(code)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != dim {
		t.Fatalf("decoded length %d != %d", len(decoded), dim)
	}
	// Reconstruction should be much closer than a random vector.
	var recon, randomErr float64
	for i := 0; i < dim; i++ {
		d := float64(vectors[0][i]) - float64(decoded[i])
		recon += d * d
		d2 := float64(vectors[0][i])
		randomErr += d2 * d2
	}
	if recon > randomErr*0.75 {
		t.Fatalf("reconstruction error %f not meaningfully below random %f", recon, randomErr)
	}
}

func TestProductQuantizer_Errors(t *testing.T) {
	if _, err := NewProductQuantizer(0, 4, 16); err == nil {
		t.Fatal("dim 0 should fail")
	}
	if _, err := NewProductQuantizer(8, 0, 16); err == nil {
		t.Fatal("m 0 should fail")
	}
	if _, err := NewProductQuantizer(8, 4, 1); err == nil {
		t.Fatal("k 1 should fail")
	}
	if _, err := NewProductQuantizer(10, 4, 16); err == nil {
		t.Fatal("dim not divisible by m should fail")
	}
	pq, _ := NewProductQuantizer(8, 4, 16)
	if _, err := pq.Encode([]float32{1, 2, 3, 4, 5, 6, 7, 8}); err == nil {
		t.Fatal("encode before training should fail")
	}
	if _, err := pq.DistanceTable([]float32{1, 2, 3, 4}); err == nil {
		t.Fatal("bad-length table query should fail")
	}
}

func TestPQIndex_SearchRecall(t *testing.T) {
	const dim, m, k = 32, 8, 32
	// Two well-separated clusters plus query near the first cluster.
	var vectors [][]float32
	for i := 0; i < 30; i++ {
		vectors = append(vectors, jitterCluster(t, 1, i, dim))
	}
	for i := 0; i < 30; i++ {
		vectors = append(vectors, jitterCluster(t, 2, i, dim))
	}
	pq, err := NewProductQuantizer(dim, m, k)
	if err != nil {
		t.Fatal(err)
	}
	if err := pq.Train(vectors); err != nil {
		t.Fatal(err)
	}
	pqi, err := NewPQIndex("pqtest", pq)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewPQIndex("pqtest", nil); err == nil {
		t.Fatal("nil quantizer should fail")
	}

	mem := NewMemoryIndex("pqtest", dim)
	ctx := context.Background()
	for i, v := range vectors {
		c := testChunk(fmt.Sprintf("c%d", i), v)
		if err := pqi.Add(ctx, c); err != nil {
			t.Fatal(err)
		}
		if err := mem.Add(ctx, c); err != nil {
			t.Fatal(err)
		}
	}
	if pqi.MemoryBytes() != int64(60*m) {
		t.Fatalf("PQ MemoryBytes = %d, want %d", pqi.MemoryBytes(), 60*m)
	}

	query := jitterCluster(t, 1, 777, dim)
	const topK = 10
	pqRes, err := pqi.Search(ctx, query, DefaultSearchOptions(topK))
	if err != nil {
		t.Fatal(err)
	}
	memRes, err := mem.Search(ctx, query, DefaultSearchOptions(topK))
	if err != nil {
		t.Fatal(err)
	}
	// Recall@10 against the exact index: most top-10 must overlap.
	want := make(map[string]bool, topK)
	for _, r := range memRes {
		want[r.Chunk.ID] = true
	}
	hits := 0
	for _, r := range pqRes {
		if want[r.Chunk.ID] {
			hits++
		}
	}
	if hits < topK-2 {
		t.Fatalf("recall@%d = %d/%d too low (PQ degraded ranking)", topK, hits, topK)
	}
	// Top-1 must be from cluster 1 (IDs c0..c29).
	topIdx, err := strconv.Atoi(pqRes[0].Chunk.ID[1:])
	if err != nil {
		t.Fatalf("unexpected ID format %q: %v", pqRes[0].Chunk.ID, err)
	}
	if topIdx >= 30 {
		t.Fatalf("top-1 from wrong cluster: %s", pqRes[0].Chunk.ID)
	}
}
