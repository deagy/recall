# Benchmark Guide

Recall ships benchmarks alongside the packages they measure. This guide
covers running them, comparing results between changes, and how CI guards
against performance regressions.

## Running Benchmarks

```bash
# All benchmarks (with memory stats)
go test ./... -run=^$ -bench=. -benchmem

# One package
go test ./index/ -run=^$ -bench=. -benchmem

# One benchmark by name regex
go test ./bm25/ -run=^$ -bench=BenchmarkBM25_Search -benchmem

# Control iteration count / time budget
go test ./... -run=^$ -bench=. -benchtime=100x   # fixed iterations
go test ./... -run=^$ -bench=. -benchtime=500ms  # fixed time per benchmark

# With CPU / memory profiling
go test ./index/ -run=^$ -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
go tool pprof cpu.prof
```

`-run=^$` skips unit tests so only benchmarks run. Where they exist, key
benchmarks include HNSW vs brute-force search (`index/`), BM25 scoring
(`bm25/`), chunking throughput (`chunker/`), and graph traversal (`graph/`).

## Comparing Changes with benchstat

The standard workflow uses [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat):

```bash
# 1. Capture the baseline (before your change)
git switch main
go test ./... -run=^$ -bench=. -benchtime=500ms -count=10 > old.txt

# 2. Capture your change (same flags!)
git switch my-branch
go test ./... -run=^$ -bench=. -benchtime=500ms -count=10 > new.txt

# 3. Compare (multiple -count samples give significance estimates)
go run golang.org/x/perf/cmd/benchstat@latest old.txt new.txt
```

Rules of thumb:

- Use `-count=5` or more per side; single samples are noise.
- Keep flags identical between runs (`-benchtime`, `-benchmem`).
- Close other heavy applications; shared CI machines are noisier than
  dedicated hardware, so treat small (<10%) deltas with skepticism.

## CI Regression Detection

On pull requests, the `benchmark` CI job:

1. Checks out both the PR head and the merge base (`main`).
2. Runs the full benchmark suite on each (`-benchtime=10x -count=1`) into
   `go test` benchmark output files.
3. Compares them with [`scripts/benchcompare.sh`](../scripts/benchcompare.sh),
   which fails the job when a benchmark regresses by more than the threshold.

Defaults (override with arguments/env):

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| threshold | 50% | Relative slowdown considered a regression |
| floor | 1000 ns/op | Absolute delta below which changes are ignored (filters sub-µs noise) |

The generous threshold + floor combination reflects shared-runner noise; it
catches algorithmic regressions (accidental O(n²), lock contention, extra
allocations in hot loops) without failing on jitter. New benchmarks (absent
from the base) are reported but never fail the job. On pushes to `main` the
results are uploaded as a workflow artifact for trend reference.

If a legitimate change slows something down, note it in the PR description;
a maintainer can merge with an acknowledged regression (or the benchmark is
adjusted when the workload definition itself changed).

## Typical Numbers

Reference magnitudes on modern hardware (see README “Performance” for the
latest table):

| Operation | Order of magnitude |
| --------- | ------------------ |
| HNSW search (1K vectors) | low µs/op |
| Brute-force search (1K vectors) | tens of µs/op |
| BM25 search (1K corpus) | ~µs/op |
| Chunking (small doc) | tens of ns/op |
| Graph AddEntity | ~100 ns/op |
| Store Upload (chunk + embed) | ~µs/op |
| ONNX embedding (MiniLM-L6) | ~ms/op per text |
