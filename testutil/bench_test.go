package testutil_test

import (
	"context"
	"testing"

	"github.com/deagy/recall/index"
	"github.com/deagy/recall/testutil"
)

// BenchmarkFixtureSearch verifies that the Harness and FixtureStore work
// correctly under a real benchmark. Run with:
//
//	go test ./testutil/ -bench=BenchmarkFixtureSearch -benchtime=10x
func BenchmarkFixtureSearch(b *testing.B) {
	f, err := testutil.NewFixtureStore(
		testutil.FixtureDoc{ID: "a", Content: "the quick brown fox jumps"},
		testutil.FixtureDoc{ID: "b", Content: "lazy dogs sleep all day long"},
	)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	h := testutil.NewHarness(b)
	h.Run(1, func(i int) error {
		_, err := f.Store.Search(context.Background(), "quick fox", index.SearchOptions{TopK: 2})
		return err
	})
}
