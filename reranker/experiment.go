package reranker

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/deagy/recall/index"
)

// DefaultNDCGCutoff is the cutoff K used for NDCG when the experiment
// config does not specify one.
const DefaultNDCGCutoff = 3

// RelevanceSample is one query's labeled evaluation: the candidate set and
// the ground-truth relevance of each candidate in the order the variant
// ranked them. Relevance uses the same binary convention as LTRExample
// (1 = relevant, 0 = not).
type RelevanceSample struct {
	// Query is the query this sample belongs to.
	Query string
	// Ranked is the candidate set in the order the variant ranked them,
	// each with its binary relevance.
	Ranked []RankedRelevance
}

// RankedRelevance pairs a chunk with its ground-truth relevance.
type RankedRelevance struct {
	// ChunkID identifies the chunk.
	ChunkID string
	// Relevant is 1 for relevant, 0 for not.
	Relevant float64
}

// MarkTopRelevant returns a sample that marks the first chunk of the
// ranking as relevant and every other chunk as not.
func MarkTopRelevant(query string, chunkIDs []string) RelevanceSample {
	s := RelevanceSample{Query: query, Ranked: make([]RankedRelevance, 0, len(chunkIDs))}
	for i, id := range chunkIDs {
		relevant := 0.0
		if i == 0 {
			relevant = 1
		}
		s.Ranked = append(s.Ranked, RankedRelevance{ChunkID: id, Relevant: relevant})
	}
	return s
}

// MarkRelevantIDs returns a sample for the given ordered ranking in which
// exactly the listed chunk IDs are marked relevant.
func MarkRelevantIDs(query string, chunkIDs, relevantIDs []string) RelevanceSample {
	want := make(map[string]bool, len(relevantIDs))
	for _, id := range relevantIDs {
		want[id] = true
	}
	s := RelevanceSample{Query: query, Ranked: make([]RankedRelevance, 0, len(chunkIDs))}
	for _, id := range chunkIDs {
		relevant := 0.0
		if want[id] {
			relevant = 1
		}
		s.Ranked = append(s.Ranked, RankedRelevance{ChunkID: id, Relevant: relevant})
	}
	return s
}

// ABConfig configures an ABTest.
type ABConfig struct {
	// NDCGCutoff is the K used for the NDCG@K metric. Defaults to
	// DefaultNDCGCutoff.
	NDCGCutoff int
}

// arm bundles one variant under test with its accumulated samples.
type arm struct {
	name     string
	reranker Reranker
	samples  []RelevanceSample
}

// Experiment is an A/B test over rerankers: each arm is a named variant
// whose reranking quality is measured against labeled RelevanceSamples.
// Collect samples per arm (typically by running each variant's Rerank and
// labeling the results), then call Complete for a full comparison.
type Experiment struct {
	mu      sync.RWMutex
	ndcgK   int
	arms    map[string]*arm
	armKeys []string
}

// NewABTest creates an empty experiment.
func NewABTest(cfg ABConfig) *Experiment {
	k := cfg.NDCGCutoff
	if k < 1 {
		k = DefaultNDCGCutoff
	}
	return &Experiment{
		ndcgK: k,
		arms:  make(map[string]*arm),
	}
}

// AddVariant registers a named reranker variant. Names must be unique.
func (e *Experiment) AddVariant(name string, rr Reranker) error {
	if name == "" {
		return fmt.Errorf("reranker: abtest: variant name must not be empty")
	}
	if rr == nil {
		return fmt.Errorf("reranker: abtest: variant %q has a nil reranker", name)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, ok := e.arms[name]; ok {
		return fmt.Errorf("reranker: abtest: variant %q already registered", name)
	}
	e.arms[name] = &arm{name: name, reranker: rr}
	e.armKeys = append(e.armKeys, name)
	return nil
}

// RecordSample appends one labeled sample to the named variant.
func (e *Experiment) RecordSample(variant string, s RelevanceSample) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	a, ok := e.arms[variant]
	if !ok {
		return fmt.Errorf("reranker: abtest: unknown variant %q", variant)
	}
	if len(s.Ranked) == 0 {
		return fmt.Errorf("reranker: abtest: sample for %q has no ranked results", variant)
	}
	a.samples = append(a.samples, s)
	return nil
}

// Rerank runs the named variant's reranker over the given candidates,
// returning the ranking so it can be labeled and fed back via RecordSample.
func (e *Experiment) Rerank(ctx context.Context, variant, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	e.mu.RLock()
	a, ok := e.arms[variant]
	e.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("reranker: abtest: unknown variant %q", variant)
	}
	return a.reranker.Rerank(ctx, query, results)
}

// SampleCount returns the number of labeled samples recorded for the named
// variant.
func (e *Experiment) SampleCount(variant string) (int, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	a, ok := e.arms[variant]
	if !ok {
		return 0, fmt.Errorf("reranker: abtest: unknown variant %q", variant)
	}
	return len(a.samples), nil
}

// VariantMetrics holds the retrieval-quality metrics for one arm.
type VariantMetrics struct {
	// Name is the variant name.
	Name string
	// Samples is the number of labeled samples this arm was scored on.
	Samples int
	// NDCGAtK is the mean NDCG@K across samples.
	NDCGAtK float64
	// MRRAtK is the mean reciprocal rank of the first relevant result.
	MRRAtK float64
	// PrecisionAtK is the fraction of relevant results in the top K.
	PrecisionAtK float64
}

// ExperimentResult is the completed comparison of two arms.
type ExperimentResult struct {
	// A and B are the metrics for each arm.
	A, B VariantMetrics
	// WinRateA is the fraction of pairwise sample comparisons where A's
	// NDCG strictly exceeds B's.
	WinRateA float64
	// TStat and PValue come from a Welch t-test on the per-sample NDCG
	// values (A vs B). PValue is approximated with the normal CDF.
	TStat  float64
	PValue float64
	// Significant reports whether PValue < 0.05.
	Significant bool
}

// Complete finalizes the experiment by comparing the two named arms. Each
// arm must have at least one sample; arm order is irrelevant.
func (e *Experiment) Complete(a, b string) (*ExperimentResult, error) {
	e.mu.RLock()
	aa, okA := e.arms[a]
	bb, okB := e.arms[b]
	e.mu.RUnlock()
	if !okA {
		return nil, fmt.Errorf("reranker: abtest: unknown variant %q", a)
	}
	if !okB {
		return nil, fmt.Errorf("reranker: abtest: unknown variant %q", b)
	}
	if len(aa.samples) == 0 || len(bb.samples) == 0 {
		return nil, fmt.Errorf("reranker: abtest: both arms need at least one sample")
	}

	ndcgA := perSampleNDCG(aa.samples, e.ndcgK)
	ndcgB := perSampleNDCG(bb.samples, e.ndcgK)

	res := &ExperimentResult{
		A: summarize(a, aa.samples, ndcgA),
		B: summarize(b, bb.samples, ndcgB),
	}
	res.TStat, res.PValue, res.Significant = welchTTest(ndcgA, ndcgB)
	wins := 0
	for i := range ndcgA {
		if ndcgA[i] > ndcgB[i] {
			wins++
		}
	}
	res.WinRateA = float64(wins) / float64(len(ndcgA))
	return res, nil
}

// summarize computes the per-arm summary metrics from its samples.
func summarize(name string, samples []RelevanceSample, ndcg []float64) VariantMetrics {
	m := VariantMetrics{Name: name, Samples: len(samples)}
	m.NDCGAtK = mean(ndcg)
	m.MRRAtK = mean(mrrPerSample(samples))
	m.PrecisionAtK = mean(precisionPerSample(samples, len(ndcg)))
	return m
}

// perSampleNDCG returns the NDCG@K of every sample.
func perSampleNDCG(samples []RelevanceSample, k int) []float64 {
	out := make([]float64, len(samples))
	for i, s := range samples {
		out[i] = ndcgAtK(s.Ranked, k)
	}
	return out
}

// mrrPerSample returns the reciprocal rank of the first relevant result for
// every sample (0 when the sample has no relevant result).
func mrrPerSample(samples []RelevanceSample) []float64 {
	out := make([]float64, len(samples))
	for i, s := range samples {
		for j, r := range s.Ranked {
			if r.Relevant >= 1 {
				out[i] = 1.0 / float64(j+1)
				break
			}
		}
	}
	return out
}

// precisionPerSample returns the fraction of the top-K results that are
// relevant for every sample.
func precisionPerSample(samples []RelevanceSample, k int) []float64 {
	out := make([]float64, len(samples))
	for i, s := range samples {
		top := s.Ranked
		if k < len(top) {
			top = top[:k]
		}
		if len(top) == 0 {
			continue
		}
		rel := 0.0
		for _, r := range top {
			if r.Relevant >= 1 {
				rel++
			}
		}
		out[i] = rel / float64(len(top))
	}
	return out
}

// ndcgAtK computes NDCG@K for one ranking with binary relevance.
func ndcgAtK(ranked []RankedRelevance, k int) float64 {
	top := ranked
	if k < len(top) {
		top = top[:k]
	}
	if len(top) == 0 {
		return 0
	}
	var dcg float64
	for i, r := range top {
		if r.Relevant > 0 {
			dcg += r.Relevant / math.Log2(float64(i+2))
		}
	}
	if dcg == 0 {
		return 0
	}
	// Ideal ranking: all relevant items of the sample in the top positions.
	relevance := make([]float64, 0, len(top))
	for _, r := range ranked {
		relevance = append(relevance, r.Relevant)
	}
	sort.SliceStable(relevance, func(i, j int) bool { return relevance[i] > relevance[j] })
	if k < len(relevance) {
		relevance = relevance[:k]
	}
	var idcg float64
	for i, r := range relevance {
		idcg += r / math.Log2(float64(i+2))
	}
	return dcg / idcg
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

// welchTTest runs Welch's t-test comparing two independent samples and
// returns (t, p, significant). The p-value uses a normal CDF approximation
// of the t-distribution, which is accurate for the moderate sample sizes
// typical of A/B tests.
func welchTTest(a, b []float64) (float64, float64, bool) {
	if len(a) == 0 || len(b) == 0 {
		return 0, 1, false
	}
	meanA, varA := meanAndVar(a)
	meanB, varB := meanAndVar(b)
	seA := varA / float64(len(a))
	seB := varB / float64(len(b))
	se2 := seA + seB
	if se2 == 0 {
		if meanA == meanB {
			return 0, 1, false
		}
		t := 1e6
		if meanA < meanB {
			t = -1e6
		}
		return t, 0, true
	}
	t := (meanA - meanB) / math.Sqrt(se2)
	p := 2 * (1 - normCDF(math.Abs(t)))
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	return t, p, p < 0.05
}

// meanAndVar returns the sample mean and population variance of xs.
func meanAndVar(xs []float64) (float64, float64) {
	m := mean(xs)
	var sum float64
	for _, x := range xs {
		d := x - m
		sum += d * d
	}
	return m, sum / float64(len(xs))
}

// normCDF returns the standard normal CDF at x via the Abramowitz & Stegun
// 7.1.26 approximation of the error function (|error| < 1.5e-7).
func normCDF(x float64) float64 {
	return 0.5 * (1 + erfApprox(x/math.Sqrt2))
}

func erfApprox(x float64) float64 {
	// Save the sign; the approximation is for x >= 0.
	sign := 1.0
	if x < 0 {
		sign = -1
		x = -x
	}
	t := 1.0 / (1.0 + 0.3275911*x)
	y := 1.0 - (((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*math.Exp(-x*x)
	return sign * y
}
