package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/deagy/recall/eval"
)

// writeReport saves a synthetic eval report with every mean metric set to v.
func writeReport(t *testing.T, name string, v float64) string {
	t.Helper()
	r := &eval.Report{
		Dataset: "cli-test", K: 3, NumQueries: 1,
		MeanPrecision: v, MeanRecall: v, MeanMRR: v, MeanNDCG: v,
		GeneratedAt: time.Now(),
	}
	path := filepath.Join(t.TempDir(), name)
	if err := r.SaveJSON(path); err != nil {
		t.Fatalf("saving report: %v", err)
	}
	return path
}

func TestLocalEvalAndCompare(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)
	mustRunCLI(t, "--config", cfgPath, "upload", writeNotesFile(t))

	// Discover a real chunk ID to use as ground truth.
	out := mustRunCLI(t, "--config", cfgPath, "search", "garbage collection", "-o", "json")
	var so searchOutput
	decodeJSON(t, out, &so)
	if so.Count == 0 {
		t.Fatal("no search results to evaluate against")
	}
	chunkID := so.Results[0].ID

	ds := fmt.Sprintf(`{"Name":"cli-test","Queries":[{"ID":"q1","Query":"garbage collection","RelevantIDs":["%s"]}]}`, chunkID)
	dsPath := filepath.Join(t.TempDir(), "dataset.json")
	if err := os.WriteFile(dsPath, []byte(ds), 0o644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(t.TempDir(), "report.json")
	out = mustRunCLI(t, "--config", cfgPath, "eval", dsPath, "--top-k", "3", "--save", reportPath)
	if !strings.Contains(out, "mean mrr") {
		t.Fatalf("eval output: %s", out)
	}
	if _, err := os.Stat(reportPath); err != nil {
		t.Fatalf("--save did not write the report: %v", err)
	}

	// JSON output parses as an eval report.
	out = mustRunCLI(t, "--config", cfgPath, "eval", dsPath, "--top-k", "3", "-o", "json")
	var rep eval.Report
	decodeJSON(t, out, &rep)
	if rep.NumQueries != 1 || rep.K != 3 {
		t.Errorf("eval report = %+v", rep)
	}

	// Comparing a report with itself passes.
	out = mustRunCLI(t, "--config", cfgPath, "eval", "compare", reportPath, reportPath)
	if !strings.Contains(out, "PASS") {
		t.Errorf("self-compare output: %s", out)
	}

	// Missing dataset errors.
	if _, err := runCLI(t, "--config", cfgPath, "eval", filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("missing dataset: expected error")
	}
}

func TestEvalCompareRegressions(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)
	baseline := writeReport(t, "baseline.json", 0.9)

	// A drop beyond tolerance is a regression: exit code 2.
	current := writeReport(t, "current.json", 0.5)
	out, err := runCLI(t, "--config", cfgPath, "eval", "compare", baseline, current)
	var ee *exitError
	if !errors.As(err, &ee) || ee.Code != 2 {
		t.Fatalf("expected exit code 2, got err=%v", err)
	}
	if !strings.Contains(out, "FAIL") || !strings.Contains(out, "regression") {
		t.Errorf("regression output: %s", out)
	}

	// JSON output parses and reports the failure.
	out, err = runCLI(t, "--config", cfgPath, "eval", "compare", baseline, current, "-o", "json")
	if err == nil {
		t.Fatal("regression should fail the command")
	}
	var co compareOutput
	decodeJSON(t, out, &co)
	if co.Passed || len(co.Regressions) == 0 {
		t.Errorf("compare output = %+v", co)
	}

	// An improvement passes.
	better := writeReport(t, "better.json", 0.95)
	out = mustRunCLI(t, "--config", cfgPath, "eval", "compare", baseline, better)
	if !strings.Contains(out, "PASS") {
		t.Errorf("improvement output: %s", out)
	}

	// A missing report file errors.
	if _, err := runCLI(t, "--config", cfgPath, "eval", "compare", baseline, filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("missing report: expected error")
	}
}
