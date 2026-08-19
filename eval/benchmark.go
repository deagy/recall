package eval

import (
	"context"
	"fmt"
	"sort"
)

// Answerer produces an answer for a question given the retrieved context.
type Answerer interface {
	// Answer returns an answer to the question grounded on the context.
	Answer(ctx context.Context, question, context string) (string, error)
}

// RetrievalSystem retrieves ranked chunk IDs and can resolve their text. It is
// satisfied by a thin adapter over store.Store (Retrieve via Search,
// ContextText via GetChunk).
type RetrievalSystem interface {
	Retriever

	// ContextText returns the text content of a chunk by ID.
	ContextText(id string) (string, bool)
}

// BenchmarkSuite runs an evaluation dataset against a retrieval system and
// supports regression comparison against a baseline report.
type BenchmarkSuite struct {
	// Dataset is the evaluation dataset to run.
	Dataset *Dataset

	// K is the number of results to retrieve per query.
	K int
}

// NewBenchmarkSuite creates a BenchmarkSuite for a dataset at cutoff K.
func NewBenchmarkSuite(ds *Dataset, k int) *BenchmarkSuite {
	if k <= 0 {
		k = 10
	}
	return &BenchmarkSuite{Dataset: ds, K: k}
}

// Run evaluates retrieval quality for every query in the dataset and returns
// an aggregate report.
func (s *BenchmarkSuite) Run(ctx context.Context, r Retriever) (*Report, error) {
	if r == nil {
		return nil, fmt.Errorf("eval: BenchmarkSuite requires a Retriever")
	}
	results := make([]PerQueryResult, 0, len(s.Dataset.Queries))
	for _, q := range s.Dataset.Queries {
		ids, err := r.Retrieve(ctx, q.Query, s.K)
		if err != nil {
			return nil, fmt.Errorf("eval: retrieve %q: %w", q.Query, err)
		}
		m := ComputeRetrievalMetrics(q.Query, ids, q.RelevantIDs, q.Relevance, s.K)
		results = append(results, PerQueryResult{ID: q.ID, Query: q.Query, Metrics: m})
	}
	return NewReportFromResults(s.Dataset.Name, s.K, results), nil
}

// RunWithAnswers runs retrieval, generates answers, and scores both retrieval
// and answer quality (faithfulness, relevance, correctness).
func (s *BenchmarkSuite) RunWithAnswers(ctx context.Context, sys RetrievalSystem, a Answerer, j Judge) (*Report, error) {
	if sys == nil || a == nil || j == nil {
		return nil, fmt.Errorf("eval: RunWithAnswers requires a retrieval system, answerer, and judge")
	}
	results := make([]PerQueryResult, 0, len(s.Dataset.Queries))
	for _, q := range s.Dataset.Queries {
		ids, err := sys.Retrieve(ctx, q.Query, s.K)
		if err != nil {
			return nil, fmt.Errorf("eval: retrieve %q: %w", q.Query, err)
		}
		m := ComputeRetrievalMetrics(q.Query, ids, q.RelevantIDs, q.Relevance, s.K)

		var contextParts []string
		for _, id := range ids {
			if text, ok := sys.ContextText(id); ok && text != "" {
				contextParts = append(contextParts, text)
			}
		}
		contextText := joinContext(contextParts)

		answer, err := a.Answer(ctx, q.Query, contextText)
		if err != nil {
			return nil, fmt.Errorf("eval: answer %q: %w", q.Query, err)
		}
		quality, err := j.Judge(ctx, JudgeInput{
			Question:  q.Query,
			Context:   contextText,
			Answer:    answer,
			Reference: q.ReferenceAnswer,
		})
		if err != nil {
			return nil, fmt.Errorf("eval: judge %q: %w", q.Query, err)
		}

		results = append(results, PerQueryResult{
			ID:      q.ID,
			Query:   q.Query,
			Metrics: m,
			Answer:  answer,
			Quality: quality,
		})
	}
	return NewReportFromResults(s.Dataset.Name, s.K, results), nil
}

// joinContext joins context parts with a blank line separator.
func joinContext(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += "\n\n"
		}
		out += p
	}
	return out
}

// MetricDelta is the change of a single metric between a baseline and a
// current report.
type MetricDelta struct {
	// Name is the metric name (e.g., "mean_mrr").
	Name string

	// Baseline is the metric value in the baseline report.
	Baseline float64

	// Current is the metric value in the current report.
	Current float64

	// Delta is Current - Baseline (negative is a drop).
	Delta float64
}

// Comparison is the result of comparing a current report against a baseline.
type Comparison struct {
	// Baseline is the reference report.
	Baseline *Report

	// Current is the report being checked.
	Current *Report

	// Deltas lists every compared metric.
	Deltas []MetricDelta

	// Regressions are metrics that dropped by more than the tolerance.
	Regressions []MetricDelta

	// Improvements are metrics that rose by more than the tolerance.
	Improvements []MetricDelta

	// Passed is true when there are no regressions.
	Passed bool
}

// Compare checks a current report against a baseline for regressions. A metric
// is a regression when it drops by more than tolerance (absolute); a
// tolerance of 0 flags any drop. Answer-quality metrics are only compared when
// both reports contain them.
func Compare(current, baseline *Report, tolerance float64) *Comparison {
	type named struct {
		name string
		base float64
		cur  float64
	}
	var metrics []named
	add := func(name string, base, cur float64) {
		metrics = append(metrics, named{name, base, cur})
	}
	add("mean_precision", baseline.MeanPrecision, current.MeanPrecision)
	add("mean_recall", baseline.MeanRecall, current.MeanRecall)
	add("mean_mrr", baseline.MeanMRR, current.MeanMRR)
	add("mean_ndcg", baseline.MeanNDCG, current.MeanNDCG)
	if baseline.HasAnswerMetrics && current.HasAnswerMetrics {
		add("mean_faithfulness", baseline.MeanFaithfulness, current.MeanFaithfulness)
		add("mean_answer_relevance", baseline.MeanAnswerRelevance, current.MeanAnswerRelevance)
		add("mean_correctness", baseline.MeanCorrectness, current.MeanCorrectness)
	}

	c := &Comparison{Baseline: baseline, Current: current}
	for _, m := range metrics {
		d := MetricDelta{Name: m.name, Baseline: m.base, Current: m.cur, Delta: m.cur - m.base}
		c.Deltas = append(c.Deltas, d)
		if d.Delta < -tolerance {
			c.Regressions = append(c.Regressions, d)
		} else if d.Delta > tolerance {
			c.Improvements = append(c.Improvements, d)
		}
	}
	c.Passed = len(c.Regressions) == 0

	// Deterministic ordering for stable output.
	sort.Slice(c.Deltas, func(i, j int) bool { return c.Deltas[i].Name < c.Deltas[j].Name })
	sort.Slice(c.Regressions, func(i, j int) bool { return c.Regressions[i].Name < c.Regressions[j].Name })
	sort.Slice(c.Improvements, func(i, j int) bool { return c.Improvements[i].Name < c.Improvements[j].Name })
	return c
}
