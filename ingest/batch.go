package ingest

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// RunBatch ingests multiple sources with the same pipeline options. Each
// ref is processed by its own Pipeline (Source is set per ref); the shared
// Progress, Dedup, and Incremental (if configured in opts) are used across
// all refs. Refs run concurrently up to opts.Concurrency. Results are
// returned in input order; a non-nil error is returned when any ref failed
// to load (individual document failures are in each Result).
func RunBatch(ctx context.Context, opts Options, refs []string) ([]*Result, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	workers := opts.Concurrency
	if workers <= 1 {
		workers = 1
	}
	results := make([]*Result, len(refs))
	errs := make([]error, len(refs))

	jobs := make(chan int)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				var err error
				results[idx], err = runRef(ctx, opts, refs[idx])
				errs[idx] = err
			}
		}()
	}
	for i := range refs {
		if ctx.Err() != nil {
			errs[i] = ctx.Err()
			break
		}
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	var failed []string
	for i, err := range errs {
		if err != nil {
			failed = append(failed, refs[i]+": "+err.Error())
		}
	}
	if len(failed) > 0 {
		return results, fmt.Errorf("ingest: %d/%d sources failed: %s", len(failed), len(refs), strings.Join(failed, "; "))
	}
	return results, nil
}

// runRef builds and runs a single-source pipeline.
func runRef(ctx context.Context, opts Options, ref string) (*Result, error) {
	// A batch processes refs concurrently; document-level parallelism
	// within a ref would multiply workers, so refs run sequentially inside.
	opts.Source = ref
	opts.Concurrency = 0
	p, err := NewPipeline(opts)
	if err != nil {
		return &Result{Source: ref}, err
	}
	return p.Run(ctx)
}
