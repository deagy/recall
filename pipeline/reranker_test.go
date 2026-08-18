package pipeline

import (
	"context"
	"fmt"
	"sort"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

// fakeStore returns a fixed list of results (honoring TopK) so the pipeline's
// two-stage logic can be tested deterministically without an embedder.
type fakeStore struct {
	results  []index.SearchResult
	lastTopK int
}

func (f *fakeStore) Upload(_ context.Context, _ *core.Document, _ string) error { return nil }

func (f *fakeStore) Search(_ context.Context, _ string, opts index.SearchOptions) ([]index.SearchResult, error) {
	f.lastTopK = opts.TopK
	k := opts.TopK
	if k <= 0 || k > len(f.results) {
		k = len(f.results)
	}
	return f.results[:k], nil
}

func (f *fakeStore) SearchHybrid(ctx context.Context, q string, opts index.SearchOptions) ([]index.SearchResult, error) {
	return f.Search(ctx, q, opts)
}

func (f *fakeStore) GetChunk(id string) (*core.Chunk, bool) {
	for _, r := range f.results {
		if r.Chunk != nil && r.Chunk.ID == id {
			return r.Chunk, true
		}
	}
	return nil, false
}

func (f *fakeStore) DeleteChunk(context.Context, string) error    { return nil }
func (f *fakeStore) DeleteDocument(context.Context, string) error { return nil }
func (f *fakeStore) Count() int                                   { return len(f.results) }
func (f *fakeStore) Namespaces() []string                         { return []string{"default"} }
func (f *fakeStore) Close() error                                 { return nil }

// contentReranker orders results by a content-based score, independent of the
// coarse (vector) score, so it visibly reorders the coarse ranking.
type contentReranker struct{}

func (contentReranker) Name() string { return "content-reranker" }

func (contentReranker) Rerank(_ context.Context, _ string, results []index.SearchResult) ([]index.SearchResult, error) {
	scores := map[string]float64{
		"gamma content": 1.0,
		"alpha content": 0.5,
		"beta content":  0.2,
	}
	out := make([]index.SearchResult, len(results))
	for i, r := range results {
		out[i] = r
		out[i].RerankScore = scores[r.Chunk.Content]
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].RerankScore > out[j].RerankScore })
	for i := range out {
		out[i].RerankRank = i + 1
		out[i].Reranker = "content-reranker"
	}
	return out, nil
}

func fakeResults() []index.SearchResult {
	return []index.SearchResult{
		{Chunk: &core.Chunk{ID: "c1", Content: "alpha content"}, Score: 0.9},
		{Chunk: &core.Chunk{ID: "c2", Content: "beta content"}, Score: 0.8},
		{Chunk: &core.Chunk{ID: "c3", Content: "gamma content"}, Score: 0.7},
	}
}

func TestQuery_NoRerankerPreservesCoarse(t *testing.T) {
	s := &fakeStore{results: fakeResults()}
	p := NewRAGPipeline(s, DefaultTemplate())
	resp, err := p.Query(context.Background(), "q")
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if resp.Sources[0].Chunk.ID != "c1" {
		t.Errorf("top = %s, want c1 (coarse)", resp.Sources[0].Chunk.ID)
	}
	for _, r := range resp.Sources {
		if r.RerankRank != 0 || r.Reranker != "" {
			t.Errorf("unexpected rerank attribution on %s: rank=%d name=%q", r.Chunk.ID, r.RerankRank, r.Reranker)
		}
	}
}

func TestQuery_WithRerankerReorders(t *testing.T) {
	s := &fakeStore{results: fakeResults()}
	p := NewRAGPipeline(s, DefaultTemplate()).WithReranker(contentReranker{})
	resp, err := p.Query(context.Background(), "q")
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	want := []string{"c3", "c1", "c2"}
	for i, id := range want {
		if resp.Sources[i].Chunk.ID != id {
			t.Fatalf("position %d = %s, want %s", i, resp.Sources[i].Chunk.ID, id)
		}
	}
	if resp.Sources[0].RerankRank != 1 || resp.Sources[0].Reranker != "content-reranker" {
		t.Errorf("attribution wrong: rank=%d name=%q", resp.Sources[0].RerankRank, resp.Sources[0].Reranker)
	}
}

func TestQuery_WithRerankTopKTruncates(t *testing.T) {
	s := &fakeStore{results: fakeResults()}
	p := NewRAGPipeline(s, DefaultTemplate()).WithReranker(contentReranker{}).WithRerankTopK(2)
	resp, err := p.Query(context.Background(), "q")
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(resp.Sources) != 2 {
		t.Fatalf("expected 2 sources, got %d", len(resp.Sources))
	}
	if resp.Sources[0].Chunk.ID != "c3" || resp.Sources[1].Chunk.ID != "c1" {
		t.Errorf("top-2 = %s,%s, want c3,c1", resp.Sources[0].Chunk.ID, resp.Sources[1].Chunk.ID)
	}
}

func TestQuery_WithCoarseTopKLimitsRetrieval(t *testing.T) {
	s := &fakeStore{results: fakeResults()}
	p := NewRAGPipeline(s, DefaultTemplate()).WithReranker(contentReranker{}).WithCoarseTopK(2)
	resp, err := p.Query(context.Background(), "q")
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if s.lastTopK != 2 {
		t.Errorf("coarseTopK not honored: store saw TopK=%d, want 2", s.lastTopK)
	}
	if len(resp.Sources) != 2 {
		t.Fatalf("expected 2 sources, got %d", len(resp.Sources))
	}
	for _, r := range resp.Sources {
		if r.Chunk.ID == "c3" {
			t.Error("c3 should not have been retrieved under coarseTopK=2")
		}
	}
}

func TestQueryHybrid_WithReranker(t *testing.T) {
	s := &fakeStore{results: fakeResults()}
	p := NewRAGPipeline(s, DefaultTemplate()).WithReranker(contentReranker{})
	resp, err := p.QueryHybrid(context.Background(), "q")
	if err != nil {
		t.Fatalf("queryHybrid: %v", err)
	}
	if resp.Sources[0].Chunk.ID != "c3" {
		t.Errorf("hybrid top = %s, want c3", resp.Sources[0].Chunk.ID)
	}
}

func TestContainsAll(t *testing.T) {
	if err := containsAll("the quick brown fox", "quick", "brown"); err != nil {
		t.Errorf("unexpected: %v", err)
	}
	if err := containsAll("no match here", "zzz"); err == nil {
		t.Error("expected error for missing substring")
	}
}

func containsAll(s string, subs ...string) error {
	for _, sub := range subs {
		found := false
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("substring %q not found", sub)
		}
	}
	return nil
}
