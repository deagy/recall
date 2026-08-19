package testutil

import "testing"

// Harness wraps a *testing.B with consistent warmup and timer handling, so
// benchmarks share one correct pattern and propagate errors cleanly.
type Harness struct {
	b *testing.B
}

// NewHarness creates a Harness for the calling benchmark.
func NewHarness(b *testing.B) *Harness {
	return &Harness{b: b}
}

// Run executes fn a total of warmups + b.N times. Warmup iterations run before
// the timer starts (to exclude cold-start effects); the timed iterations call
// b.StopTimer before failing on error. A nil fn or negative warmups is a
// programming error and fails the benchmark.
func (h *Harness) Run(warmups int, fn func(i int) error) {
	if fn == nil {
		h.b.Fatal("testutil.Harness.Run: nil fn")
		return
	}
	if warmups < 0 {
		h.b.Fatal("testutil.Harness.Run: negative warmups")
		return
	}
	_ = run(warmups, h.b.N, h.b.ResetTimer, h.b.StopTimer, func(err error) { h.b.Fatal(err) }, fn)
}

// run executes warmups+n iterations of fn: it calls reset once before the
// timed iterations, stop both on a failing timed iteration and after the loop,
// and fatal on the first error. It returns the first error encountered (nil if
// none). This is the testable core of Harness.Run.
func run(warmups, n int, reset, stop func(), fatal func(error), fn func(i int) error) error {
	for i := 0; i < warmups; i++ {
		if err := fn(i); err != nil {
			fatal(err)
			return err
		}
	}
	reset()
	for i := 0; i < n; i++ {
		if err := fn(i); err != nil {
			stop()
			fatal(err)
			return err
		}
	}
	stop()
	return nil
}
