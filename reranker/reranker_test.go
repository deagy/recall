package reranker

import (
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

// mkResult builds an index.SearchResult with the given id/content/score.
func mkResult(id, content string, score float64) index.SearchResult {
	return index.SearchResult{
		Chunk: &core.Chunk{ID: id, Content: content},
		Score: score,
	}
}

func TestFinalize_SortsAndRanks(t *testing.T) {
	in := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "a"}, RerankScore: 0.2, Score: 0.9},
		{Chunk: &core.Chunk{ID: "b"}, RerankScore: 0.8, Score: 0.1},
		{Chunk: &core.Chunk{ID: "c"}, RerankScore: 0.5, Score: 0.5},
	}
	out := finalize("test", in)

	if out[0].Chunk.ID != "b" || out[1].Chunk.ID != "c" || out[2].Chunk.ID != "a" {
		t.Fatalf("unexpected order: %s %s %s", out[0].Chunk.ID, out[1].Chunk.ID, out[2].Chunk.ID)
	}
	for i, r := range out {
		if r.RerankRank != i+1 {
			t.Errorf("rank[%d] = %d, want %d", i, r.RerankRank, i+1)
		}
		if r.Reranker != "test" {
			t.Errorf("reranker name = %q, want test", r.Reranker)
		}
	}
	// Coarse score must be preserved.
	if out[0].Score != 0.1 {
		t.Errorf("coarse score mutated: %f", out[0].Score)
	}
}

func TestFinalize_TieBreakByCoarseThenID(t *testing.T) {
	in := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "z"}, RerankScore: 0.5, Score: 0.3},
		{Chunk: &core.Chunk{ID: "a"}, RerankScore: 0.5, Score: 0.7},
		{Chunk: &core.Chunk{ID: "m"}, RerankScore: 0.5, Score: 0.7},
	}
	out := finalize("t", in)
	// Equal rerank score: higher coarse first; tie on coarse -> id asc.
	want := []string{"a", "m", "z"}
	for i, id := range want {
		if out[i].Chunk.ID != id {
			t.Fatalf("position %d = %s, want %s", i, out[i].Chunk.ID, id)
		}
	}
}

func TestFinalize_DoesNotMutateInput(t *testing.T) {
	in := []index.SearchResult{
		{Chunk: &core.Chunk{ID: "a"}, RerankScore: 0.2},
		{Chunk: &core.Chunk{ID: "b"}, RerankScore: 0.9},
	}
	_ = finalize("t", in)
	if in[0].RerankRank != 0 || in[1].RerankRank != 0 {
		t.Fatal("finalize mutated the input slice")
	}
}

func TestNormalizeToUnit(t *testing.T) {
	n := normalizeToUnit([]float64{1, 2, 3, 4})
	want := []float64{0, 1.0 / 3, 2.0 / 3, 1}
	for i := range n {
		if n[i] != want[i] {
			t.Errorf("normalize[%d] = %f, want %f", i, n[i], want[i])
		}
	}
	// All-equal input maps to all ones.
	eq := normalizeToUnit([]float64{5, 5, 5})
	for _, v := range eq {
		if v != 1 {
			t.Errorf("expected 1 for equal inputs, got %f", v)
		}
	}
	if len(normalizeToUnit(nil)) != 0 {
		t.Error("expected empty output for nil input")
	}
}

func TestClampScore(t *testing.T) {
	cases := []struct{ v, max, want float64 }{
		{-1, 10, 0},
		{5, 10, 5},
		{20, 10, 10},
		{0, 0, 0},
	}
	for _, c := range cases {
		if got := clampScore(c.v, c.max); got != c.want {
			t.Errorf("clampScore(%f,%f) = %f, want %f", c.v, c.max, got, c.want)
		}
	}
}

func TestParseScore(t *testing.T) {
	cases := []struct {
		in   string
		want float64
	}{
		{"7", 7},
		{" 7 ", 7},
		{"The score is 7.", 7},
		{"7.5", 7.5},
		{"I rate it 9 out of 10", 9},
		{"", 0},
		{"no number here", 0},
		{"-3", -3},
	}
	for _, c := range cases {
		if got := parseScore(c.in); got != c.want {
			t.Errorf("parseScore(%q) = %f, want %f", c.in, got, c.want)
		}
	}
}
