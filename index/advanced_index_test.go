package index

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/fuse"
)

func metaChunk(id, src string) *core.Chunk {
	return &core.Chunk{
		ID:      id,
		Content: "body of " + id,
		Metadata: map[string]core.Value{
			"source": core.String{Value: src},
		},
	}
}

func TestMetadataIndex_Candidates(t *testing.T) {
	mi := NewMetadataIndex()
	if err := mi.Add(nil); err != core.ErrInvalidChunk {
		t.Fatalf("nil chunk: %v", err)
	}
	// 4 chunks across two sources; two have a topic tag.
	c1 := metaChunk("c1", "web")
	c1.Metadata["topic"] = core.String{Value: "go"}
	c2 := metaChunk("c2", "web")
	c2.Metadata["topic"] = core.String{Value: "rust"}
	c3 := metaChunk("c3", "git")
	c4 := metaChunk("c4", "git")
	for _, c := range []*core.Chunk{c1, c2, c3, c4} {
		if err := mi.Add(c); err != nil {
			t.Fatal(err)
		}
	}
	if mi.Count() != 4 {
		t.Fatalf("count = %d", mi.Count())
	}

	// No filters -> no pre-filter needed.
	if set, ok := mi.Candidates(nil); ok || set != nil {
		t.Fatal("no filters should mean no pre-filter")
	}

	// Term filter narrows to source=web.
	set, ok := mi.Candidates([]Filter{&TermFilter{Key: "source", Value: "web"}})
	if !ok || len(set) != 2 {
		t.Fatalf("term filter candidates: ok=%v set=%v", ok, set)
	}
	for _, id := range []string{"c1", "c2"} {
		if _, present := set[id]; !present {
			t.Fatalf("expected %s in candidates", id)
		}
	}

	// TermIn filter on topic.
	set, _ = mi.Candidates([]Filter{&TermInFilter{Key: "topic", Values: []string{"go", "rust"}}})
	if len(set) != 2 {
		t.Fatalf("term-in candidates = %v", set)
	}

	// Combined: source=web AND topic=go -> only c1.
	set, _ = mi.Candidates([]Filter{
		&TermFilter{Key: "source", Value: "web"},
		&TermFilter{Key: "topic", Value: "go"},
	})
	if len(set) != 1 {
		t.Fatalf("combined candidates = %v", set)
	}

	// Impossible term -> empty.
	set, _ = mi.Candidates([]Filter{&TermFilter{Key: "source", Value: "s3"}})
	if len(set) != 0 {
		t.Fatalf("impossible term candidates = %v", set)
	}

	// Generic filter fallback: RangeFilter is not a term filter.
	c1.Metadata["rank"] = core.Number{Value: 3}
	c3.Metadata["rank"] = core.Number{Value: 7}
	if err := mi.Add(c1); err != nil {
		t.Fatal(err)
	}
	if err := mi.Add(c3); err != nil {
		t.Fatal(err)
	}
	lo, hi := 1.0, 5.0
	set, _ = mi.Candidates([]Filter{&RangeFilter{Key: "rank", Min: &lo, Max: &hi, MinIncl: true, MaxIncl: true}})
	if len(set) != 1 {
		t.Fatalf("range fallback candidates = %v", set)
	}

	// Values + Get + SortedIDs.
	if got := mi.Values("topic"); len(got) != 2 || got[0] != "go" || got[1] != "rust" {
		t.Fatalf("Values(topic) = %v", got)
	}
	if got := mi.Values("missing"); got != nil {
		t.Fatalf("Values(missing) = %v", got)
	}
	if _, ok := mi.Get("c1"); !ok {
		t.Fatal("Get(c1) should exist")
	}
	if got := SortedIDs(set); len(got) != 1 || got[0] != "c1" {
		t.Fatalf("SortedIDs = %v", got)
	}
}

func TestMetadataIndex_Remove(t *testing.T) {
	mi := NewMetadataIndex()
	c := metaChunk("c1", "web")
	if err := mi.Add(c); err != nil {
		t.Fatal(err)
	}
	if err := mi.Remove("c1"); err != nil {
		t.Fatal(err)
	}
	if mi.Count() != 0 {
		t.Fatalf("count after remove = %d", mi.Count())
	}
	if err := mi.Remove("c1"); err != core.ErrNotFound {
		t.Fatalf("second remove: %v", err)
	}
	if got := mi.Values("source"); got != nil {
		t.Fatalf("Values after remove = %v", got)
	}
}

type customFilter struct{ id string }

func (f *customFilter) Match(c *core.Chunk) bool { return c.ID == f.id }

func TestMetadataIndex_CustomFilter(t *testing.T) {
	mi := NewMetadataIndex()
	_ = mi.Add(metaChunk("a", "x"))
	_ = mi.Add(metaChunk("b", "x"))
	set, ok := mi.Candidates([]Filter{&customFilter{id: "b"}})
	if !ok || len(set) != 1 {
		t.Fatalf("custom filter: ok=%v set=%v", ok, set)
	}
}

func TestHybridIndex_KeywordOnlyHit(t *testing.T) {
	const dim = 16
	hi := NewHybridIndex("hyb", dim, nil)
	ctx := context.Background()

	// "needle" text lives only in one chunk whose vector is unrelated
	// to the query vector.
	cFar := testChunk("far", randUnitVector(1, dim))
	cFar.Content = "the unique needle term lives here"
	cNear := testChunk("near", randUnitVector(2, dim))
	cNear.Content = "generic content about vectors"
	cThird := testChunk("third", randUnitVector(3, dim))
	cThird.Content = "more generic content"
	for _, c := range []*core.Chunk{cFar, cNear, cThird} {
		if err := hi.Add(ctx, c); err != nil {
			t.Fatal(err)
		}
	}
	if hi.Count() != 3 || hi.Dimension() != dim || hi.Namespace() != "hyb" {
		t.Fatalf("state wrong: %d %d %s", hi.Count(), hi.Dimension(), hi.Namespace())
	}

	query := randUnitVector(100, dim)
	res, err := hi.Search(ctx, "needle", query, SearchOptions{
		TopK:       3,
		BM25Weight: 1.0, // pure BM25
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Chunk.ID != "far" {
		t.Fatalf("pure BM25 should return only 'far', got %+v", res)
	}

	// Balanced search: the keyword-only hit must still appear.
	res, err = hi.Search(ctx, "needle", query, SearchOptions{TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range res {
		if r.Chunk.ID == "far" {
			found = true
		}
	}
	if !found {
		t.Fatalf("keyword-only hit missing from balanced hybrid results: %+v", res)
	}

	// Custom RRF fusion from the options.
	res, err = hi.Search(ctx, "needle", query, SearchOptions{TopK: 3, Fusion: fuse.NewRRFFusion(10)})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].Chunk.ID != "far" {
		t.Fatalf("RRF results should lead with 'far': %+v", res)
	}

	// GetChunk + SearchBM25 passthrough.
	if c, ok := hi.GetChunk("far"); !ok || c.ID != "far" {
		t.Fatal("GetChunk(far) failed")
	}
	if got := hi.SearchBM25("needle"); len(got) != 1 || got[0].DocID != "far" {
		t.Fatalf("SearchBM25 = %+v", got)
	}

	// Delete removes from both sub-indexes.
	if err := hi.Delete(ctx, "far"); err != nil {
		t.Fatal(err)
	}
	if hi.Count() != 2 || len(hi.SearchBM25("needle")) != 0 {
		t.Fatal("delete did not clear both sub-indexes")
	}
	if _, ok := hi.GetChunk("far"); ok {
		t.Fatal("GetChunk after delete should miss")
	}

	// Add validation.
	if err := hi.Add(ctx, nil); err != core.ErrInvalidChunk {
		t.Fatalf("nil chunk: %v", err)
	}
	if err := hi.Add(ctx, &core.Chunk{ID: "x"}); err != core.ErrInvalidEmbedding {
		t.Fatalf("missing embedding: %v", err)
	}
}

func TestHybridIndex_ConstructorFusion(t *testing.T) {
	const dim = 16
	hi := NewHybridIndex("hyb2", dim, fuse.NewWeightedFusion(0.5))
	ctx := context.Background()
	a := testChunk("a", randUnitVector(1, dim))
	a.Content = "shared words here"
	b := testChunk("b", randUnitVector(2, dim))
	b.Content = "shared words there"
	if err := hi.Add(ctx, a); err != nil {
		t.Fatal(err)
	}
	if err := hi.Add(ctx, b); err != nil {
		t.Fatal(err)
	}
	res, err := hi.Search(ctx, "shared words", randUnitVector(7, dim), SearchOptions{TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("want both chunks, got %d", len(res))
	}
}

func TestMultiVectorIndex_Aggregations(t *testing.T) {
	const dim = 8
	ctx := context.Background()
	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	orth := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	anti := []float32{-1, 0, 0, 0, 0, 0, 0, 0}

	mv := NewMultiVectorIndex("mv", dim)
	if mv.Count() != 0 || mv.Dimension() != dim || mv.Namespace() != "mv" {
		t.Fatalf("initial state wrong")
	}

	// Single-embedding Add path.
	single := testChunk("single", query)
	if err := mv.Add(ctx, single); err != nil {
		t.Fatal(err)
	}

	// Multi-embedding: one aligned, one orthogonal.
	multi := testChunk("multi", nil)
	if err := mv.AddMulti(ctx, multi, [][]float32{query, orth}); err != nil {
		t.Fatal(err)
	}
	if n, ok := mv.VectorCount("multi"); !ok || n != 2 {
		t.Fatalf("VectorCount(multi) = %d %v", n, ok)
	}
	if n, ok := mv.VectorCount("nope"); ok || n != 0 {
		t.Fatalf("VectorCount(nope) = %d %v", n, ok)
	}
	if c, ok := mv.GetChunk("single"); !ok || c.ID != "single" {
		t.Fatal("GetChunk(single) failed")
	}

	// Third chunk with only orthogonal/anti vectors.
	mv.AddMulti(ctx, testChunk("orth", nil), [][]float32{orth, anti})

	res, err := mv.Search(ctx, query, DefaultSearchOptions(10))
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]float64{}
	for _, r := range res {
		byID[r.Chunk.ID] = r.Score
	}
	if byID["multi"] < 0.99 || byID["multi"] > 1.000001 {
		t.Fatalf("maxsim(multi) = %f, want ~1.0", byID["multi"])
	}
	if byID["orth"] > 0.01 {
		t.Fatalf("maxsim(orth) = %f, want ~0 (min -1..0)", byID["orth"])
	}

	// Mean: multi = (1 + 0)/2 = 0.5; orth = (0 + -1)/2 = -0.5.
	mv.Aggregation = MeanAggregation
	res, _ = mv.Search(ctx, query, SearchOptions{TopK: 10, MinScore: -2})
	byID = map[string]float64{}
	for _, r := range res {
		byID[r.Chunk.ID] = r.Score
	}
	if abs64(byID["multi"]-0.5) > 1e-6 {
		t.Fatalf("mean(multi) = %f, want 0.5", byID["multi"])
	}
	if abs64(byID["orth"]+0.5) > 1e-6 {
		t.Fatalf("mean(orth) = %f, want -0.5", byID["orth"])
	}

	// TopMean with TopN=2 on multi: same as mean here = 0.5.
	mv.Aggregation = TopMeanAggregation
	mv.TopN = 2
	res, _ = mv.Search(ctx, query, SearchOptions{TopK: 10, MinScore: -2})
	byID = map[string]float64{}
	for _, r := range res {
		byID[r.Chunk.ID] = r.Score
	}
	if abs64(byID["multi"]-0.5) > 1e-6 {
		t.Fatalf("topmean(multi) = %f, want 0.5", byID["multi"])
	}

	// MinScore filters out the negatives.
	res, _ = mv.Search(ctx, query, SearchOptions{TopK: 10, MinScore: 0.1})
	if len(res) != 2 {
		t.Fatalf("minscore should keep single+multi, got %d", len(res))
	}

	// Delete + errors.
	if err := mv.Delete(ctx, "multi"); err != nil {
		t.Fatal(err)
	}
	if mv.Count() != 2 {
		t.Fatalf("count after delete = %d", mv.Count())
	}
	if err := mv.Delete(ctx, "multi"); err != core.ErrNotFound {
		t.Fatalf("double delete: %v", err)
	}
	if err := mv.Add(ctx, testChunk("bad", nil)); err != core.ErrInvalidEmbedding {
		t.Fatalf("nil embedding: %v", err)
	}
	if err := mv.AddMulti(ctx, testChunk("bad", nil), nil); err == nil {
		t.Fatal("empty vector set should fail")
	}
	if err := mv.AddMulti(ctx, testChunk("bad", nil), [][]float32{{1, 2, 3}}); err != core.ErrEmbeddingMismatch {
		t.Fatalf("dimension mismatch: %v", err)
	}
	if _, err := mv.Search(ctx, []float32{1, 2}, DefaultSearchOptions(1)); err != core.ErrEmbeddingMismatch {
		t.Fatalf("query dimension mismatch: %v", err)
	}
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestSmallCovers(t *testing.T) {
	const dim = 16
	ctx := context.Background()

	// MultiVector.AddBatch + filter path.
	mv := NewMultiVectorIndex("batch", dim)
	a := testChunk("ba", randUnitVector(1, dim))
	a.Metadata = map[string]core.Value{"src": core.String{Value: "x"}}
	b := testChunk("bb", randUnitVector(2, dim))
	if err := mv.AddBatch(ctx, []*core.Chunk{a, b}); err != nil {
		t.Fatal(err)
	}
	res, err := mv.Search(ctx, randUnitVector(1, dim), SearchOptions{
		TopK:    10,
		Filters: []Filter{&TermFilter{Key: "src", Value: "x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Chunk.ID != "ba" {
		t.Fatalf("filtered search = %+v", res)
	}

	// PQ WithSeed + Dimensions + untrained Decode error paths.
	pq, err := NewProductQuantizer(8, 4, 8)
	if err != nil {
		t.Fatal(err)
	}
	pq.WithSeed(42)
	if d, m := pq.Dimensions(); d != 8 || m != 4 {
		t.Fatalf("Dimensions = %d %d", d, m)
	}
	if _, err := pq.Decode([]uint8{0, 1, 2, 3}); err == nil {
		t.Fatal("untrained decode should fail")
	}
	if _, err := pq.DistanceTable([]float32{1, 2, 3, 4, 5, 6, 7, 8}); err == nil {
		t.Fatal("untrained table should fail")
	}

	// SQ untrained dequantize error.
	qz, _ := NewScalarQuantizer(4)
	if _, err := qz.Dequantize([]uint8{1, 2, 3, 4}); err == nil {
		t.Fatal("untrained dequantize should fail")
	}
	if _, err := qz.MeanAbsError([][]float32{{1, 2, 3, 4}}); err == nil {
		t.Fatal("untrained error metric should fail")
	}

	// MetadataIndex Add with nil Value entry and re-Add replacement.
	mi := NewMetadataIndex()
	c := &core.Chunk{ID: "m1", Metadata: map[string]core.Value{"k": nil}}
	if err := mi.Add(c); err != nil {
		t.Fatal(err)
	}
	c.Metadata["k"] = core.String{Value: "v2"}
	if err := mi.Add(c); err != nil {
		t.Fatal(err)
	}
	set, _ := mi.Candidates([]Filter{&TermFilter{Key: "k", Value: "v2"}})
	if len(set) != 1 {
		t.Fatalf("re-add candidates = %v", set)
	}
}
