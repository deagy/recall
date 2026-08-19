package eval

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeRetriever struct {
	results map[string][]string
}

func (f *fakeRetriever) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	ids := f.results[query]
	if len(ids) > k {
		ids = ids[:k]
	}
	return ids, nil
}

type fakeRetrievalSystem struct {
	fakeRetriever
	texts map[string]string
}

func (f *fakeRetrievalSystem) ContextText(id string) (string, bool) {
	t, ok := f.texts[id]
	return t, ok
}

type fakeAnswerer struct{ answer string }

func (f *fakeAnswerer) Answer(ctx context.Context, question, context string) (string, error) {
	return f.answer, nil
}

func testDataset() *Dataset {
	ds := NewDataset("demo")
	ds.Add(EvalQuery{
		Query:           "Capital of France?",
		RelevantIDs:     []string{"c1", "c2"},
		Context:         "Paris is the capital of France.",
		ReferenceAnswer: "Paris is the capital of France.",
	})
	ds.Add(EvalQuery{
		Query:       "Who wrote it?",
		RelevantIDs: []string{"c3"},
	})
	return ds
}

func TestBenchmarkSuite_Run(t *testing.T) {
	r := &fakeRetriever{results: map[string][]string{
		"Capital of France?": {"c1", "c9", "c2"},
		"Who wrote it?":      {"c8", "c3"},
	}}
	report, err := NewBenchmarkSuite(testDataset(), 3).Run(context.Background(), r)
	require.NoError(t, err)
	assert.Equal(t, "demo", report.Dataset)
	assert.Equal(t, 2, report.NumQueries)
	assert.Equal(t, 3, report.K)
	// Both queries have their first relevant at rank 1 or 2 -> MRR in (0.5,1]
	assert.Greater(t, report.MeanMRR, 0.5)
	assert.Greater(t, report.MeanRecall, 0.0)
	assert.NotEmpty(t, report.PerQuery)
}

func TestBenchmarkSuite_Run_NilRetriever(t *testing.T) {
	_, err := NewBenchmarkSuite(testDataset(), 3).Run(context.Background(), nil)
	require.Error(t, err)
}

func TestBenchmarkSuite_RunWithAnswers(t *testing.T) {
	sys := &fakeRetrievalSystem{
		fakeRetriever: fakeRetriever{results: map[string][]string{
			"Capital of France?": {"c1", "c2"},
			"Who wrote it?":      {"c3"},
		}},
		texts: map[string]string{
			"c1": "Paris is the capital of France.",
			"c2": "France is in Europe.",
			"c3": "The book was written in 1999.",
		},
	}
	a := &fakeAnswerer{answer: "Paris is the capital of France."}
	report, err := NewBenchmarkSuite(testDataset(), 3).RunWithAnswers(context.Background(), sys, a, NewOverlapJudge())
	require.NoError(t, err)
	assert.True(t, report.HasAnswerMetrics)
	require.Len(t, report.PerQuery, 2)

	// Query 1: answer fully grounded in retrieved context, matches reference.
	assert.InDelta(t, 1.0, report.PerQuery[0].Quality.Faithfulness, 1e-9)
	assert.InDelta(t, 1.0, report.PerQuery[0].Quality.Relevance, 1e-9)
	assert.InDelta(t, 1.0, report.PerQuery[0].Quality.Correctness, 1e-9)
	// Query 2: retrieved context ("written in 1999") supports none of the answer.
	assert.Equal(t, 0.0, report.PerQuery[1].Quality.Faithfulness)
	assert.Equal(t, 0.0, report.PerQuery[1].Quality.Correctness, "no reference -> 0")

	assert.InDelta(t, 0.5, report.MeanFaithfulness, 1e-9)
	assert.InDelta(t, 0.5, report.MeanAnswerRelevance, 1e-9)
	assert.InDelta(t, 0.5, report.MeanCorrectness, 1e-9)
}

func TestBenchmarkSuite_RunWithAnswers_MissingDeps(t *testing.T) {
	s := NewBenchmarkSuite(testDataset(), 3)
	sys := &fakeRetrievalSystem{fakeRetriever: fakeRetriever{}}
	_, err := s.RunWithAnswers(context.Background(), nil, &fakeAnswerer{}, NewOverlapJudge())
	require.Error(t, err)
	_, err = s.RunWithAnswers(context.Background(), sys, nil, NewOverlapJudge())
	require.Error(t, err)
	_, err = s.RunWithAnswers(context.Background(), sys, &fakeAnswerer{}, nil)
	require.Error(t, err)
}

func TestCompare_Regression(t *testing.T) {
	baseline := NewReportFromResults("d", 3, []PerQueryResult{
		{Query: "a", Metrics: RetrievalMetrics{Precision: 1, Recall: 1, MRR: 1, NDCG: 1}},
	})
	current := NewReportFromResults("d", 3, []PerQueryResult{
		{Query: "a", Metrics: RetrievalMetrics{Precision: 0.4, Recall: 0.4, MRR: 0.4, NDCG: 0.4}},
	})
	c := Compare(current, baseline, 0.01)
	assert.False(t, c.Passed)
	assert.NotEmpty(t, c.Regressions)
	assert.Empty(t, c.Improvements)
}

func TestCompare_PassAndImprove(t *testing.T) {
	baseline := NewReportFromResults("d", 3, []PerQueryResult{
		{Query: "a", Metrics: RetrievalMetrics{Precision: 0.5, Recall: 0.5, MRR: 0.5, NDCG: 0.5}},
	})
	current := NewReportFromResults("d", 3, []PerQueryResult{
		{Query: "a", Metrics: RetrievalMetrics{Precision: 0.8, Recall: 0.8, MRR: 0.8, NDCG: 0.8}},
	})
	c := Compare(current, baseline, 0.01)
	assert.True(t, c.Passed)
	assert.Empty(t, c.Regressions)
	assert.NotEmpty(t, c.Improvements)
}

func TestReport_JSONRoundTripAndMarkdown(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")

	report := NewReportFromResults("demo", 3, []PerQueryResult{
		{Query: "q1", Metrics: RetrievalMetrics{Precision: 0.9, Recall: 0.8, MRR: 0.7, NDCG: 0.6}},
	})
	require.NoError(t, report.SaveJSON(path))

	loaded, err := LoadReport(path)
	require.NoError(t, err)
	assert.InDelta(t, report.MeanPrecision, loaded.MeanPrecision, 1e-9)
	assert.Equal(t, "demo", loaded.Dataset)

	md := report.Markdown()
	assert.Contains(t, md, "# Evaluation Report: demo")
	assert.Contains(t, md, "Precision@3")
	assert.Contains(t, md, "q1")
}

func TestReport_Markdown_IncludesAnswerMetrics(t *testing.T) {
	report := NewReportFromResults("demo", 3, []PerQueryResult{
		{Query: "q1", Metrics: RetrievalMetrics{Precision: 1}, Answer: "an answer",
			Quality: AnswerQuality{Faithfulness: 0.9, Relevance: 0.8, Correctness: 0.7}},
	})
	md := report.Markdown()
	assert.True(t, report.HasAnswerMetrics)
	assert.True(t, strings.Contains(md, "Faithfulness"))
	assert.True(t, strings.Contains(md, "Correctness"))
}
