package reranker

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/deagy/recall/index"
)

// fakeRR is a minimal Reranker for A/B tests: it ranks candidates by
// per-chunk score when a lookup is provided, otherwise by coarse score.
type fakeRR struct {
	name   string
	lookup map[string]float64
	err    error
}

func (f *fakeRR) Name() string { return f.name }

func (f *fakeRR) Rerank(_ context.Context, _ string, results []index.SearchResult) ([]index.SearchResult, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]index.SearchResult, 0, len(results))
	for _, res := range results {
		r2 := res
		s := res.Score
		if res.Chunk != nil {
			if v, ok := f.lookup[res.Chunk.ID]; ok {
				s = v
			}
		}
		r2.RerankScore = s
		out = append(out, r2)
	}
	return finalize(f.name, out), nil
}

// evalRanking runs a variant over the given candidates and returns the
// ranking as chunk IDs, using the supplied lookup for labels.
func evalRanking(t *testing.T, e *Experiment, variant, query string, candidates []index.SearchResult) []string {
	t.Helper()
	out, err := e.Rerank(context.Background(), variant, query, candidates)
	if err != nil {
		t.Fatalf("rerank %s: %v", variant, err)
	}
	ids := make([]string, 0, len(out))
	for _, r := range out {
		ids = append(ids, r.Chunk.ID)
	}
	return ids
}

func TestExperiment_DetectsBetterVariant(t *testing.T) {
	e := NewABTest(ABConfig{})
	// Variant "a" ranks by the misleading coarse score; "b" applies the
	// true relevance signal.
	if err := e.AddVariant("a", &fakeRR{name: "a"}); err != nil {
		t.Fatalf("add a: %v", err)
	}
	lookup := map[string]float64{"good": 0.9, "bad": 0.1}
	if err := e.AddVariant("b", &fakeRR{name: "b", lookup: lookup}); err != nil {
		t.Fatalf("add b: %v", err)
	}

	queries := []string{"q1", "q2", "q3"}
	for _, q := range queries {
		cands := []index.SearchResult{
			mkResult("good", "the relevant answer here", 0.4),
			mkResult("bad", "an unrelated distractor", 0.9),
		}
		rankA := evalRanking(t, e, "a", q, cands)
		if err := e.RecordSample("a", MarkRelevantIDs(q, rankA, []string{"good"})); err != nil {
			t.Fatalf("record a: %v", err)
		}
		rankB := evalRanking(t, e, "b", q, cands)
		if err := e.RecordSample("b", MarkRelevantIDs(q, rankB, []string{"good"})); err != nil {
			t.Fatalf("record b: %v", err)
		}
	}

	res, err := e.Complete("a", "b")
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if res.B.NDCGAtK != 1.0 {
		t.Errorf("b NDCG = %f, want 1.0", res.B.NDCGAtK)
	}
	if res.A.NDCGAtK >= res.B.NDCGAtK {
		t.Errorf("a NDCG %f !< b NDCG %f", res.A.NDCGAtK, res.B.NDCGAtK)
	}
	if res.B.MRRAtK != 1.0 || res.A.MRRAtK <= 0 || res.A.MRRAtK >= 1.0 {
		t.Errorf("MRR a=%f b=%f, want b=1 and a in (0,1)", res.A.MRRAtK, res.B.MRRAtK)
	}
	if res.WinRateA != 0 {
		t.Errorf("WinRateA = %f, want 0 (b wins every sample)", res.WinRateA)
	}
	if !res.Significant || res.PValue >= 0.05 {
		t.Errorf("expected significant result (p=%f)", res.PValue)
	}
	if res.TStat > 0 {
		t.Errorf("TStat = %f, want negative (a worse than b)", res.TStat)
	}
	if res.A.Samples != 3 || res.B.Samples != 3 {
		t.Errorf("sample counts a=%d b=%d, want 3/3", res.A.Samples, res.B.Samples)
	}
	// Symmetry: completing in the other order flips win rate and sign.
	res2, err := e.Complete("b", "a")
	if err != nil {
		t.Fatalf("complete b,a: %v", err)
	}
	if res2.WinRateA != 1 || res2.TStat < 0 {
		t.Errorf("reversed: winRate=%f tStat=%f, want 1 / positive", res2.WinRateA, res2.TStat)
	}
}

func TestExperiment_TiedVariants(t *testing.T) {
	e := NewABTest(ABConfig{})
	if err := e.AddVariant("a", &fakeRR{name: "a"}); err != nil {
		t.Fatalf("add a: %v", err)
	}
	if err := e.AddVariant("b", &fakeRR{name: "b"}); err != nil {
		t.Fatalf("add b: %v", err)
	}
	for i := 0; i < 5; i++ {
		q := fmt.Sprintf("q%d", i)
		cands := []index.SearchResult{
			mkResult("c1", "one", 0.9),
			mkResult("c2", "two", 0.5),
		}
		rankA := evalRanking(t, e, "a", q, cands)
		if err := e.RecordSample("a", MarkTopRelevant(q, rankA)); err != nil {
			t.Fatalf("record a: %v", err)
		}
		rankB := evalRanking(t, e, "b", q, cands)
		if err := e.RecordSample("b", MarkTopRelevant(q, rankB)); err != nil {
			t.Fatalf("record b: %v", err)
		}
	}
	res, err := e.Complete("a", "b")
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if res.TStat != 0 {
		t.Errorf("tStat = %f, want 0 for identical arms", res.TStat)
	}
	if res.Significant {
		t.Error("identical arms should not be significant")
	}
	if res.PValue != 1 {
		t.Errorf("pValue = %f, want 1", res.PValue)
	}
	if res.A.NDCGAtK != 1 || res.B.NDCGAtK != 1 {
		t.Errorf("ndcg a=%f b=%f, want 1/1", res.A.NDCGAtK, res.B.NDCGAtK)
	}
}

func TestExperiment_Errors(t *testing.T) {
	e := NewABTest(ABConfig{})
	if err := e.AddVariant("", &fakeRR{name: "x"}); err == nil {
		t.Error("expected error for empty variant name")
	}
	if err := e.AddVariant("a", nil); err == nil {
		t.Error("expected error for nil reranker")
	}
	if err := e.AddVariant("a", &fakeRR{name: "a"}); err != nil {
		t.Fatalf("add a: %v", err)
	}
	if err := e.AddVariant("a", &fakeRR{name: "a"}); err == nil {
		t.Error("expected error for duplicate variant")
	}
	if err := e.RecordSample("nope", MarkTopRelevant("q", []string{"c"})); err == nil {
		t.Error("expected error for unknown variant")
	}
	if err := e.RecordSample("a", RelevanceSample{Query: "q"}); err == nil {
		t.Error("expected error for empty sample")
	}
	if _, err := e.Rerank(context.Background(), "nope", "q", nil); err == nil {
		t.Error("expected error for unknown variant in Rerank")
	}
	if _, err := e.SampleCount("nope"); err == nil {
		t.Error("expected error for unknown variant in SampleCount")
	}
	// Empty experiment: no samples recorded.
	if _, err := e.Complete("a", "b"); err == nil {
		t.Error("expected error when an arm is missing")
	}
	e.AddVariant("b", &fakeRR{name: "b"})
	if _, err := e.Complete("a", "b"); err == nil {
		t.Error("expected error when arms have no samples")
	}
	// Propagated sub-reranker error.
	e.AddVariant("boom", &fakeRR{name: "boom", err: errors.New("boom")})
	if _, err := e.Rerank(context.Background(), "boom", "q", []index.SearchResult{mkResult("c", "c", 0.5)}); err == nil {
		t.Error("expected propagated reranker error")
	}
}

func TestNDCGAtK(t *testing.T) {
	ranked := []RankedRelevance{
		{ChunkID: "r1", Relevant: 1},
		{ChunkID: "r2", Relevant: 0},
		{ChunkID: "r3", Relevant: 1},
		{ChunkID: "r4", Relevant: 1},
	}
	// k=1: top is relevant -> 1.
	if got := ndcgAtK(ranked, 1); got != 1 {
		t.Errorf("ndcg@1 = %f, want 1", got)
	}
	// k=4: DCG = 1 + 1/3 + 1/4; IDCG = 1 + 1/2 + 1/3.
	want := (1 + 1/math.Log2(4) + 1/math.Log2(5)) / (1 + 1/math.Log2(3) + 1/math.Log2(4))
	if got := ndcgAtK(ranked, 4); math.Abs(got-want) > 1e-12 {
		t.Errorf("ndcg@4 = %f, want %f", got, want)
	}
	// No relevant items -> 0.
	if got := ndcgAtK([]RankedRelevance{{}, {}}, 2); got != 0 {
		t.Errorf("ndcg empty relevance = %f, want 0", got)
	}
	// Perfect ordering with k beyond sample length.
	perfect := []RankedRelevance{{Relevant: 1}, {Relevant: 1}}
	if got := ndcgAtK(perfect, 10); got != 1 {
		t.Errorf("ndcg perfect = %f, want 1", got)
	}
}

func TestMRRAndPrecision(t *testing.T) {
	samples := []RelevanceSample{
		{Query: "q1", Ranked: []RankedRelevance{{Relevant: 1}}},
		{Query: "q2", Ranked: []RankedRelevance{{Relevant: 0}, {Relevant: 1}}},
		{Query: "q3", Ranked: []RankedRelevance{{Relevant: 0}, {Relevant: 0}}},
	}
	mrr := mrrPerSample(samples)
	want := []float64{1.0, 0.5, 0}
	for i := range want {
		if mrr[i] != want[i] {
			t.Errorf("mrr[%d] = %f, want %f", i, mrr[i], want[i])
		}
	}
	prec := precisionPerSample(samples, 1)
	for i, w := range []float64{1, 0, 0} {
		if prec[i] != w {
			t.Errorf("precision@1[%d] = %f, want %f", i, prec[i], w)
		}
	}
	// k larger than sample length must not divide by zero.
	prec2 := precisionPerSample(samples, 5)
	if prec2[0] != 1 || prec2[1] != 0.5 || prec2[2] != 0 {
		t.Errorf("precision with k>len = %v, want [1 0.5 0]", prec2)
	}
}

func TestMarkHelpers(t *testing.T) {
	s := MarkTopRelevant("q", []string{"a", "b", "c"})
	if s.Query != "q" || len(s.Ranked) != 3 {
		t.Fatalf("unexpected sample: %+v", s)
	}
	if s.Ranked[0].Relevant != 1 || s.Ranked[1].Relevant != 0 || s.Ranked[2].Relevant != 0 {
		t.Errorf("wrong relevance flags: %+v", s.Ranked)
	}
	s2 := MarkRelevantIDs("q", []string{"a", "b", "c"}, []string{"c", "a"})
	if s2.Ranked[0].Relevant != 1 || s2.Ranked[1].Relevant != 0 || s2.Ranked[2].Relevant != 1 {
		t.Errorf("wrong relevance flags: %+v", s2.Ranked)
	}
}

func TestNormCDFApprox(t *testing.T) {
	// Known standard normal CDF values.
	cases := []struct {
		x    float64
		want float64
	}{
		{0, 0.5},
		{1, 0.8413447},
		{-1, 0.1586553},
		{1.96, 0.975002},
	}
	for _, c := range cases {
		if got := normCDF(c.x); math.Abs(got-c.want) > 1e-5 {
			t.Errorf("normCDF(%f) = %f, want %f", c.x, got, c.want)
		}
	}
}
