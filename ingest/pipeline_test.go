package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
	"github.com/deagy/recall/store"
)

// makeStore returns a fresh in-memory store with a mock embedder. A nil t
// is allowed for use in table-driven setup; constructor errors are ignored
// there (the memory store cannot actually fail to build).
func makeStore(t *testing.T) store.Store {
	if t != nil {
		t.Helper()
	}
	s, err := store.NewMemoryStore(store.Config{Namespace: "test"})
	if err != nil {
		if t != nil {
			t.Fatalf("store: %v", err)
		}
	}
	return s
}

// writeDocs creates n distinct .txt files in a temp dir and returns its path.
func writeDocs(t *testing.T, n int) string {
	t.Helper()
	dir := t.TempDir()
	for i := 0; i < n; i++ {
		content := "Document number " + string(rune('A'+i)) + " with some body text to chunk over and over again. " +
			"More body text follows for realistic length in this file."
		p := filepath.Join(dir, "doc"+string(rune('0'+i))+".txt")
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func dirOpts(t *testing.T, dir string) Options {
	t.Helper()
	dl, err := loader.NewDirectoryLoader([]string{".txt"}, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	return Options{Store: makeStore(t), Loader: dl, Source: dir}
}

func TestPipeline_Validation(t *testing.T) {
	if _, err := NewPipeline(Options{Loader: &loader.TextLoader{}}); err == nil {
		t.Error("expected missing store error")
	}
	if _, err := NewPipeline(Options{Store: makeStore(nil)}); err == nil {
		t.Error("expected missing source error")
	}
	if _, err := NewPipeline(Options{
		Store: makeStore(nil), Loader: &loader.TextLoader{}, Concurrency: -1,
	}); err == nil {
		t.Error("expected negative concurrency error")
	}
}

func TestPipeline_Run(t *testing.T) {
	dir := writeDocs(t, 3)
	p, err := NewPipeline(dirOpts(t, dir))
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Loaded != 3 || res.Uploaded != 3 || res.Skipped != 0 {
		t.Errorf("result: %+v", res)
	}
	if len(res.Failed) != 0 {
		t.Errorf("unexpected failures: %v", res.Failed)
	}
}

func TestPipeline_Incremental(t *testing.T) {
	dir := writeDocs(t, 3)
	state := filepath.Join(t.TempDir(), "state.json")
	inc, err := NewIncremental(state)
	if err != nil {
		t.Fatal(err)
	}
	opts := dirOpts(t, dir)
	opts.Incremental = inc

	p, _ := NewPipeline(opts)
	res1, err := p.Run(context.Background())
	if err != nil || res1.Uploaded != 3 {
		t.Fatalf("run1: %v %+v", err, res1)
	}
	// Unchanged re-run skips everything.
	res2, err := p.Run(context.Background())
	if err != nil || res2.Uploaded != 0 || res2.Skipped != 3 {
		t.Fatalf("run2: %v %+v", err, res2)
	}
	// A reload from disk must see the same state.
	inc2, err := NewIncremental(state)
	if err != nil {
		t.Fatalf("reload state: %v", err)
	}
	if inc2.Len() != 3 {
		t.Errorf("state size: got %d want 3", inc2.Len())
	}
}

func TestPipeline_Incremental_ChangedContent(t *testing.T) {
	dir := writeDocs(t, 2)
	state := filepath.Join(t.TempDir(), "state.json")
	inc, _ := NewIncremental(state)
	opts := dirOpts(t, dir)
	opts.Incremental = inc
	p, _ := NewPipeline(opts)
	if _, err := p.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "doc0.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "doc0.txt"), append(data, []byte("changed!")...), 0o600); err != nil {
		t.Fatal(err)
	}
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Uploaded != 1 || res.Skipped != 1 {
		t.Errorf("expected 1 upload / 1 skip, got %+v", res)
	}
}

func TestPipeline_Dedup(t *testing.T) {
	dir := t.TempDir()
	content := "identical body text for dedup testing purposes here."
	for _, name := range []string{"a.txt", "b.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	dl, _ := loader.NewDirectoryLoader([]string{".txt"}, true, nil)
	dedup := NewDeduplicator()
	opts := Options{Store: makeStore(t), Loader: dl, Source: dir, Dedup: dedup}
	p, _ := NewPipeline(opts)
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Uploaded != 1 || res.Skipped != 1 {
		t.Errorf("expected 1 upload / 1 skip, got %+v", res)
	}
	if dedup.Len() != 1 {
		t.Errorf("dedup size: %d", dedup.Len())
	}
}

func TestPipeline_ValidationReject(t *testing.T) {
	dir := writeDocs(t, 2)
	dl, _ := loader.NewDirectoryLoader([]string{".txt"}, true, nil)
	opts := Options{
		Store:     makeStore(t),
		Loader:    dl,
		Source:    dir,
		Validator: &Validator{Schema: Schema{MaxContent: 10}},
	}
	p, _ := NewPipeline(opts)
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Uploaded != 0 || res.Skipped != 2 {
		t.Errorf("expected all rejected, got %+v", res)
	}
}

func TestPipeline_Transform(t *testing.T) {
	dir := writeDocs(t, 2)
	dl, _ := loader.NewDirectoryLoader([]string{".txt"}, true, nil)
	seen := make([]string, 0)
	var mu sync.Mutex
	opts := Options{
		Store:  makeStore(t),
		Loader: dl,
		Source: dir,
		Transform: func(d *loader.Document) (*loader.Document, error) {
			mu.Lock()
			seen = append(seen, d.ID)
			mu.Unlock()
			// Drop the first document.
			if strings.HasSuffix(d.ID, "0.txt") {
				return nil, nil
			}
			d.Content = strings.ToUpper(d.Content)
			return d, nil
		},
	}
	p, _ := NewPipeline(opts)
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Uploaded != 1 || res.Skipped != 1 {
		t.Errorf("expected 1 upload / 1 drop, got %+v", res)
	}
	if len(seen) != 2 {
		t.Errorf("transform called %d times", len(seen))
	}
}

func TestPipeline_Concurrency(t *testing.T) {
	dir := writeDocs(t, 8)
	opts := dirOpts(t, dir)
	opts.Concurrency = 4
	p, _ := NewPipeline(opts)
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Loaded != 8 || res.Uploaded != 8 {
		t.Errorf("concurrent run: %+v", res)
	}
}

func TestPipeline_UploadFailure(t *testing.T) {
	dir := writeDocs(t, 3)
	dl, _ := loader.NewDirectoryLoader([]string{".txt"}, true, nil)
	failing := &failingStore{Store: makeStore(t), failID: "doc1.txt"}
	opts := Options{Store: failing, Loader: dl, Source: dir}
	p, _ := NewPipeline(opts)
	res, err := p.Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Uploaded != 2 || len(res.Failed) != 1 {
		t.Errorf("expected 2 uploads / 1 failure, got %+v", res)
	}
	if err := res.Error(); err == nil || !strings.Contains(err.Error(), "doc1.txt") {
		t.Errorf("result error: %v", err)
	}
}

func TestPipeline_LoadError(t *testing.T) {
	p, _ := NewPipeline(Options{
		Store:  makeStore(nil),
		Loader: &loader.TextLoader{},
		Source: "/definitely/missing.txt",
	})
	if _, err := p.Run(context.Background()); err == nil {
		t.Error("expected load error")
	}
}

// failingStore wraps a store and fails Upload for one document id.
type failingStore struct {
	store.Store
	failID string
}

var errBoom = errors.New("boom")

func (f *failingStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	if doc != nil && strings.HasSuffix(doc.ID, f.failID) {
		return errBoom
	}
	return f.Store.Upload(ctx, doc, content)
}
