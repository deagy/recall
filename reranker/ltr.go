package reranker

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/deagy/recall/index"
)

// FeatureFunc computes the feature vector for a (query, result) pair. The
// vector length must be constant across all calls for a given reranker.
type FeatureFunc func(query string, res index.SearchResult) []float64

// LTRExample is a single labeled training example: a (query, result) pair and
// a relevance label in [0,1] (1 = relevant, 0 = not).
type LTRExample struct {
	// Query is the retrieval query.
	Query string
	// Result is the candidate (its coarse Score and Chunk are used by the
	// default feature function).
	Result index.SearchResult
	// Label is the ground-truth relevance in [0,1].
	Label float64
}

// LTRConfig configures an LTRanker's training.
type LTRConfig struct {
	// Features overrides the default feature extraction. When nil, the
	// built-in features (coarse score, keyword overlap, length) are used.
	Features FeatureFunc

	// LearningRate is the gradient-descent step size. Defaults to 0.5.
	LearningRate float64

	// Epochs is the number of full passes over the training set. Defaults to 200.
	Epochs int

	// L2 is the L2 weight-decay coefficient. Defaults to 0.001.
	L2 float64
}

// LTRanker is a pointwise learning-to-rank model: a logistic-regression
// classifier over per-candidate features. After Fit, it scores candidates by
// the learned relevance probability. Before any training it acts as an
// identity reranker (coarse score preserved), so it is always safe to wire in.
type LTRanker struct {
	features     FeatureFunc
	learningRate float64
	epochs       int
	l2           float64

	weights []float64
	bias    float64
	fitted  bool
}

// NewLTRanker creates an LTRanker from a config.
func NewLTRanker(cfg LTRConfig) *LTRanker {
	lr := cfg.LearningRate
	if lr <= 0 {
		lr = 0.5
	}
	epochs := cfg.Epochs
	if epochs <= 0 {
		epochs = 200
	}
	l2 := cfg.L2
	if l2 < 0 {
		l2 = 0.001
	}
	features := cfg.Features
	if features == nil {
		features = DefaultFeatures
	}
	return &LTRanker{
		features:     features,
		learningRate: lr,
		epochs:       epochs,
		l2:           l2,
	}
}

// Name implements Reranker.
func (r *LTRanker) Name() string { return "ltr" }

// DefaultFeatures returns the built-in feature vector: coarse retrieval
// score, query-term overlap fraction, and an inverse-length term.
func DefaultFeatures(query string, res index.SearchResult) []float64 {
	overlap := 0.0
	if res.Chunk != nil {
		overlap = termOverlap(query, res.Chunk.Content)
	}
	length := 1.0
	if res.Chunk != nil {
		length = float64(len(strings.Fields(res.Chunk.Content)))
	}
	return []float64{
		res.Score, // coarse (vector) score
		overlap,   // fraction of query terms found in the passage
		1.0 / (1.0 + length),
	}
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1.0 + e)
}

func termOverlap(query, passage string) float64 {
	qTerms := tokenizeNoStop(strings.ToLower(query))
	if len(qTerms) == 0 {
		return 0
	}
	pSet := make(map[string]bool)
	for _, t := range tokenizeNoStop(strings.ToLower(passage)) {
		pSet[t] = true
	}
	seen := make(map[string]bool, len(qTerms))
	hits := 0
	for _, t := range qTerms {
		if seen[t] {
			continue
		}
		seen[t] = true
		if pSet[t] {
			hits++
		}
	}
	return float64(hits) / float64(len(seen))
}

// tokenizeNoStop splits text into lowercase alnum tokens (no stop-word
// removal, so even short queries contribute signal).
func tokenizeNoStop(s string) []string {
	var toks []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			toks = append(toks, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return toks
}

// Fitted reports whether the model has been trained.
func (r *LTRanker) Fitted() bool { return r.fitted }

// Fit trains the logistic model on the given labeled examples using full-batch
// gradient descent with L2 decay.
func (r *LTRanker) Fit(_ context.Context, examples []LTRExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("reranker: ltr: need at least one training example")
	}
	xs := make([][]float64, len(examples))
	ys := make([]float64, len(examples))
	dim := 0
	for i, ex := range examples {
		f := r.features(ex.Query, ex.Result)
		if dim == 0 {
			dim = len(f)
		} else if len(f) != dim {
			return fmt.Errorf("reranker: ltr: inconsistent feature dimension %d (want %d)", len(f), dim)
		}
		xs[i] = f
		ys[i] = ex.Label
	}

	w := make([]float64, dim)
	bias := 0.0
	n := float64(len(examples))
	for epoch := 0; epoch < r.epochs; epoch++ {
		gw := make([]float64, dim)
		gb := 0.0
		for i := range xs {
			z := bias
			for j := range w {
				z += w[j] * xs[i][j]
			}
			p := sigmoid(z)
			err := p - ys[i]
			gb += err
			for j := range w {
				gw[j] += err * xs[i][j]
			}
		}
		for j := range w {
			// Full-batch mean gradient plus L2 penalty on the weights.
			w[j] -= r.learningRate * (gw[j]/n + r.l2*w[j])
		}
		bias -= r.learningRate * (gb / n)
	}

	r.weights = w
	r.bias = bias
	r.fitted = true
	return nil
}

// Rerank scores each candidate with the learned model (or the coarse score
// when unfitted) and returns them ordered by the relevance score.
func (r *LTRanker) Rerank(_ context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}
	out := make([]index.SearchResult, 0, len(results))
	for _, res := range results {
		r2 := res
		if !r.fitted {
			r2.RerankScore = res.Score
		} else {
			f := r.features(query, res)
			z := r.bias
			for j := range r.weights {
				if j < len(f) {
					z += r.weights[j] * f[j]
				}
			}
			r2.RerankScore = sigmoid(z)
		}
		out = append(out, r2)
	}
	return finalize(r.Name(), out), nil
}
