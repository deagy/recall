package ingest

import (
	"context"
	"strings"
	"testing"

	"github.com/deagy/recall/loader"
)

func loaderNewDir() (*loader.DirectoryLoader, error) {
	return loader.NewDirectoryLoader([]string{".txt"}, true, nil)
}

func TestRunBatch(t *testing.T) {
	dirs := []string{writeDocs(t, 2), writeDocs(t, 3), writeDocs(t, 4)}
	// dirOpts builds its own store per call; a shared store is needed so
	// the batch exercises one pipeline config.
	dl, err := loaderNewDir()
	if err != nil {
		t.Fatal(err)
	}
	s := makeStore(t)
	opts := Options{Store: s, Loader: dl, Concurrency: 2}
	results, err := RunBatch(context.Background(), opts, dirs)
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	total := 0
	for i, r := range results {
		if r.Source != dirs[i] {
			t.Errorf("result %d source: %q", i, r.Source)
		}
		if r.Uploaded != r.Loaded {
			t.Errorf("result %d: %+v", i, r)
		}
		total += r.Uploaded
	}
	if total != 9 {
		t.Errorf("total uploaded: %d", total)
	}
}

func TestRunBatch_Empty(t *testing.T) {
	results, err := RunBatch(context.Background(), Options{Store: makeStore(nil)}, nil)
	if err != nil || results != nil {
		t.Errorf("empty batch: %v %v", err, results)
	}
}

func TestRunBatch_Failure(t *testing.T) {
	dl, _ := loaderNewDir()
	opts := Options{Store: makeStore(t), Loader: dl}
	res, err := RunBatch(context.Background(), opts, []string{writeDocs(t, 1), "/no/such/dir"})
	if err == nil || !strings.Contains(err.Error(), "1/2") {
		t.Errorf("expected partial failure error, got %v", err)
	}
	if len(res) != 2 || res[0].Uploaded != 1 {
		t.Errorf("results: %v", res)
	}
}
