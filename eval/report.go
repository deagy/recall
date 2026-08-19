package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// PerQueryResult holds the evaluation outcome for a single query.
type PerQueryResult struct {
	// ID is the evaluation case ID.
	ID string

	// Query is the query text.
	Query string

	// Metrics holds the retrieval metrics for this query.
	Metrics RetrievalMetrics

	// Answer is the generated answer (empty for retrieval-only runs).
	Answer string

	// Quality holds the answer-quality scores (zero when not evaluated).
	Quality AnswerQuality
}

// Report aggregates evaluation results across a dataset.
type Report struct {
	// Dataset is the dataset name this report was produced from.
	Dataset string `json:"dataset"`

	// K is the cutoff used for the @K metrics.
	K int `json:"k"`

	// NumQueries is the number of queries evaluated.
	NumQueries int `json:"num_queries"`

	// MeanPrecision is the mean Precision@K across queries.
	MeanPrecision float64 `json:"mean_precision"`

	// MeanRecall is the mean Recall@K across queries.
	MeanRecall float64 `json:"mean_recall"`

	// MeanMRR is the mean MRR across queries.
	MeanMRR float64 `json:"mean_mrr"`

	// MeanNDCG is the mean NDCG@K across queries.
	MeanNDCG float64 `json:"mean_ndcg"`

	// HasAnswerMetrics reports whether answer-quality metrics are present.
	HasAnswerMetrics bool `json:"has_answer_metrics,omitempty"`

	// MeanFaithfulness is the mean faithfulness score (when evaluated).
	MeanFaithfulness float64 `json:"mean_faithfulness,omitempty"`

	// MeanAnswerRelevance is the mean answer-relevance score (when evaluated).
	MeanAnswerRelevance float64 `json:"mean_answer_relevance,omitempty"`

	// MeanCorrectness is the mean correctness score (when evaluated).
	MeanCorrectness float64 `json:"mean_correctness,omitempty"`

	// PerQuery holds the per-query breakdown.
	PerQuery []PerQueryResult `json:"per_query"`

	// GeneratedAt is when the report was produced.
	GeneratedAt time.Time `json:"generated_at"`
}

// NewReportFromResults builds a Report from per-query results, computing the
// aggregate means.
func NewReportFromResults(dataset string, k int, results []PerQueryResult) *Report {
	r := &Report{
		Dataset:    dataset,
		K:          k,
		NumQueries: len(results),
		PerQuery:   results,
	}
	r.GeneratedAt = time.Now().UTC()

	if len(results) == 0 {
		return r
	}
	var p, rec, mrr, ndcg, faith, rel, corr float64
	hasAnswer := false
	for _, res := range results {
		p += res.Metrics.Precision
		rec += res.Metrics.Recall
		mrr += res.Metrics.MRR
		ndcg += res.Metrics.NDCG
		if res.Answer != "" || res.Quality != (AnswerQuality{}) {
			hasAnswer = true
			faith += res.Quality.Faithfulness
			rel += res.Quality.Relevance
			corr += res.Quality.Correctness
		}
	}
	n := float64(len(results))
	r.MeanPrecision = p / n
	r.MeanRecall = rec / n
	r.MeanMRR = mrr / n
	r.MeanNDCG = ndcg / n
	if hasAnswer {
		r.HasAnswerMetrics = true
		r.MeanFaithfulness = faith / n
		r.MeanAnswerRelevance = rel / n
		r.MeanCorrectness = corr / n
	}
	return r
}

// ToJSON serializes the report to indented JSON.
func (r *Report) ToJSON() ([]byte, error) {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("eval: marshal report: %w", err)
	}
	return data, nil
}

// SaveJSON writes the report to a JSON file at path.
func (r *Report) SaveJSON(path string) error {
	data, err := r.ToJSON()
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("eval: write report: %w", err)
	}
	return nil
}

// LoadReport reads a JSON report from path. It is the counterpart to
// SaveJSON and enables golden-file regression testing in CI.
func LoadReport(path string) (*Report, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("eval: read report: %w", err)
	}
	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("eval: parse report: %w", err)
	}
	return &r, nil
}

// Markdown renders a human-readable Markdown report.
func (r *Report) Markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Evaluation Report: %s\n\n", r.Dataset)
	fmt.Fprintf(&b, "- Queries: %d\n- K: %d\n\n", r.NumQueries, r.K)
	b.WriteString("| Metric | Value |\n|--------|-------|\n")
	writeRow := func(name string, v float64) {
		fmt.Fprintf(&b, "| %s | %.4f |\n", name, v)
	}
	writeRow(fmt.Sprintf("Precision@%d", r.K), r.MeanPrecision)
	writeRow(fmt.Sprintf("Recall@%d", r.K), r.MeanRecall)
	writeRow("MRR", r.MeanMRR)
	writeRow(fmt.Sprintf("NDCG@%d", r.K), r.MeanNDCG)
	if r.HasAnswerMetrics {
		writeRow("Faithfulness", r.MeanFaithfulness)
		writeRow("Answer Relevance", r.MeanAnswerRelevance)
		writeRow("Correctness", r.MeanCorrectness)
	}
	b.WriteString("\n## Per-query\n\n")
	b.WriteString("| Query | Precision | Recall | MRR | NDCG |\n")
	b.WriteString("|-------|-----------|--------|-----|------|\n")
	sorted := make([]PerQueryResult, len(r.PerQuery))
	copy(sorted, r.PerQuery)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Query < sorted[j].Query })
	for _, res := range sorted {
		m := res.Metrics
		fmt.Fprintf(&b, "| %s | %.4f | %.4f | %.4f | %.4f |\n",
			res.Query, m.Precision, m.Recall, m.MRR, m.NDCG)
	}
	return b.String()
}
