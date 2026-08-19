package main

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/eval"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// storeRetriever adapts a store.Store to eval.Retriever.
type storeRetriever struct {
	st     store.Store
	ns     string
	hybrid bool
}

func (r storeRetriever) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	opts := index.DefaultSearchOptions(k)
	if r.ns != "" {
		opts.Filters = namespaceFilter(r.ns)
	}
	var (
		results []index.SearchResult
		err     error
	)
	if r.hybrid {
		opts.Hybrid = true
		opts.BM25Weight = 0.5
		results, err = r.st.SearchHybrid(ctx, query, opts)
	} else {
		results, err = r.st.Search(ctx, query, opts)
	}
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(results))
	for _, res := range results {
		if res.Chunk != nil {
			ids = append(ids, res.Chunk.ID)
		}
	}
	return ids, nil
}

func newEvalCmd(o *globalOptions) *cobra.Command {
	var (
		topK   int
		save   string
		hybrid bool
	)
	cmd := &cobra.Command{
		Use:   "eval [dataset.json]",
		Short: "Run evaluation benchmarks against the store",
		Long: `Run a retrieval evaluation dataset (see eval.Dataset JSON format)
against the local store and report Precision@K, Recall@K, MRR, and NDCG@K
overall and per query.

Use --save to write the full report as JSON (consumable by
` + "`recall eval compare`" + `).`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			path := "dataset.json"
			if len(args) == 1 {
				path = args[0]
			}
			return runEval(cmd, o, path, topK, save, hybrid)
		},
	}
	cmd.Flags().IntVarP(&topK, "top-k", "k", 10, "cutoff for the @K metrics")
	cmd.Flags().StringVar(&save, "save", "", "write the full report JSON to this path")
	cmd.Flags().BoolVar(&hybrid, "hybrid", false, "retrieve with hybrid search")
	cmd.AddCommand(newEvalCompareCmd(o))
	return cmd
}

func runEval(cmd *cobra.Command, o *globalOptions, datasetPath string, topK int, save string, hybrid bool) error {
	ctx := cmd.Context()
	if err := o.requireLocal("recall eval"); err != nil {
		return err
	}

	ds, err := eval.LoadDataset(datasetPath)
	if err != nil {
		return err
	}
	if ds.Len() == 0 {
		return fmt.Errorf("dataset %s contains no queries", datasetPath)
	}

	st, err := o.openLocalStore()
	if err != nil {
		return err
	}
	defer st.Close()

	ns := ""
	if cmd.Root().Flags().Changed("namespace") {
		ns = o.namespace
	}
	suite := eval.NewBenchmarkSuite(ds, topK)
	report, err := suite.Run(ctx, storeRetriever{st: st, ns: ns, hybrid: hybrid})
	if err != nil {
		return err
	}

	if save != "" {
		if err := report.SaveJSON(save); err != nil {
			return err
		}
	}

	p := newPrinter(cmd, o.output)
	return p.emit(report, func(p *printer) {
		tw := p.table()
		fmt.Fprintf(tw, "dataset:\t%s\n", report.Dataset)
		fmt.Fprintf(tw, "k:\t%d\n", report.K)
		fmt.Fprintf(tw, "queries:\t%d\n", report.NumQueries)
		fmt.Fprintf(tw, "mean precision@%d:\t%.4f\n", report.K, report.MeanPrecision)
		fmt.Fprintf(tw, "mean recall@%d:\t%.4f\n", report.K, report.MeanRecall)
		fmt.Fprintf(tw, "mean mrr:\t%.4f\n", report.MeanMRR)
		fmt.Fprintf(tw, "mean ndcg@%d:\t%.4f\n", report.K, report.MeanNDCG)
		if report.HasAnswerMetrics {
			fmt.Fprintf(tw, "mean faithfulness:\t%.4f\n", report.MeanFaithfulness)
			fmt.Fprintf(tw, "mean answer relevance:\t%.4f\n", report.MeanAnswerRelevance)
			fmt.Fprintf(tw, "mean correctness:\t%.4f\n", report.MeanCorrectness)
		}
		if save != "" {
			fmt.Fprintf(tw, "report saved:\t%s\n", save)
		}
		tw.Flush()
		fmt.Fprintln(p.w, "\nper query:")
		qt := p.table("id", "precision", "recall", "mrr", "ndcg")
		for _, q := range report.PerQuery {
			fmt.Fprintf(qt, "%s\t%.4f\t%.4f\t%.4f\t%.4f\n", q.ID, q.Metrics.Precision, q.Metrics.Recall, q.Metrics.MRR, q.Metrics.NDCG)
		}
		qt.Flush()
	})
}

// compareMetricOut is one compared metric in `recall eval compare` output.
type compareMetricOut struct {
	Metric   string  `json:"metric" yaml:"metric"`
	Baseline float64 `json:"baseline" yaml:"baseline"`
	Current  float64 `json:"current" yaml:"current"`
	Delta    float64 `json:"delta" yaml:"delta"`
}

// compareOutput is the result of `recall eval compare`.
type compareOutput struct {
	Passed       bool               `json:"passed" yaml:"passed"`
	Tolerance    float64            `json:"tolerance" yaml:"tolerance"`
	Baseline     string             `json:"baseline" yaml:"baseline"`
	Current      string             `json:"current" yaml:"current"`
	Deltas       []compareMetricOut `json:"deltas" yaml:"deltas"`
	Regressions  []compareMetricOut `json:"regressions,omitempty" yaml:"regressions,omitempty"`
	Improvements []compareMetricOut `json:"improvements,omitempty" yaml:"improvements,omitempty"`
}

func newEvalCompareCmd(o *globalOptions) *cobra.Command {
	var tolerance float64
	cmd := &cobra.Command{
		Use:   "compare <baseline.json> <current.json>",
		Short: "Compare two evaluation reports for regressions",
		Long: `Compare a current evaluation report against a baseline report. A
metric is a regression when it drops by more than --tolerance (absolute).

Exit code: 0 when there are no regressions, 2 when at least one metric
regressed (useful as a CI gate).`,
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runEvalCompare(cmd, o, args[0], args[1], tolerance)
		},
	}
	cmd.Flags().Float64Var(&tolerance, "tolerance", 0.01, "allowed absolute drop per metric before it counts as a regression")
	return cmd
}

func runEvalCompare(cmd *cobra.Command, o *globalOptions, baselinePath, currentPath string, tolerance float64) error {
	baseline, err := eval.LoadReport(baselinePath)
	if err != nil {
		return err
	}
	current, err := eval.LoadReport(currentPath)
	if err != nil {
		return err
	}
	cmp := eval.Compare(current, baseline, tolerance)

	out := &compareOutput{
		Passed:    cmp.Passed,
		Tolerance: tolerance,
		Baseline:  baseline.Dataset,
		Current:   current.Dataset,
		Deltas:    make([]compareMetricOut, 0, len(cmp.Deltas)),
	}
	for _, d := range cmp.Deltas {
		out.Deltas = append(out.Deltas, compareMetricOut{Metric: d.Name, Baseline: d.Baseline, Current: d.Current, Delta: d.Delta})
	}
	for _, d := range cmp.Regressions {
		out.Regressions = append(out.Regressions, compareMetricOut{Metric: d.Name, Baseline: d.Baseline, Current: d.Current, Delta: d.Delta})
	}
	for _, d := range cmp.Improvements {
		out.Improvements = append(out.Improvements, compareMetricOut{Metric: d.Name, Baseline: d.Baseline, Current: d.Current, Delta: d.Delta})
	}

	p := newPrinter(cmd, o.output)
	err = p.emit(out, func(p *printer) {
		tw := p.table("metric", "baseline", "current", "delta", "direction")
		for _, d := range out.Deltas {
			dir := " "
			switch {
			case d.Delta < -tolerance:
				dir = "v regression"
			case d.Delta > tolerance:
				dir = "^ improvement"
			}
			fmt.Fprintf(tw, "%s\t%.4f\t%.4f\t%+.4f\t%s\n", d.Metric, d.Baseline, d.Current, d.Delta, dir)
		}
		tw.Flush()
		if out.Passed {
			fmt.Fprintln(p.w, "PASS: no regressions within tolerance")
		} else {
			fmt.Fprintf(p.w, "FAIL: %d regression(s): %s\n", len(out.Regressions), joinStrings(metricNames(out.Regressions)))
		}
	})
	if err != nil {
		return err
	}
	if !out.Passed {
		return &exitError{Code: 2, Message: fmt.Sprintf("eval compare: %d regression(s) detected", len(out.Regressions))}
	}
	return nil
}

func metricNames(ds []compareMetricOut) []string {
	names := make([]string, 0, len(ds))
	for _, d := range ds {
		names = append(names, d.Metric)
	}
	return names
}
