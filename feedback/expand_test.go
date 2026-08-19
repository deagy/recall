package feedback

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeEmbedder struct {
	dim     int
	embeds  map[string][]float32
	embedFn func(ctx context.Context, text string) ([]float32, error)
}

func (f *fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if f.embedFn != nil {
		return f.embedFn(ctx, text)
	}
	return f.embeds[text], nil
}
func (f *fakeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	var out [][]float32
	for _, t := range texts {
		v, err := f.Embed(ctx, t)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}
func (f *fakeEmbedder) Dimension() int { return f.dim }

type fakeSearcher struct {
	lastQuery []float32
	results   []index.SearchResult
	err       error
	calls     int
}

func (f *fakeSearcher) Search(ctx context.Context, q []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	f.lastQuery = q
	f.calls++
	return f.results, f.err
}

type fakeGetter struct{ chunks map[string]*core.Chunk }

func (f *fakeGetter) GetChunk(id string) (*core.Chunk, bool) {
	c, ok := f.chunks[id]
	return c, ok
}

func TestExpandAndRetrieve(t *testing.T) {
	getter := &fakeGetter{chunks: map[string]*core.Chunk{
		"A": {ID: "A", Content: "cat content", Embedding: []float32{1, 0}},
		"B": {ID: "B", Content: "fish content", Embedding: []float32{0, 1}},
	}}
	searcher := &fakeSearcher{
		// A is last; boosting should move it to the front.
		results: []index.SearchResult{
			{Chunk: &core.Chunk{ID: "C"}, Score: 0.9},
			{Chunk: &core.Chunk{ID: "B"}, Score: 0.5},
			{Chunk: &core.Chunk{ID: "A"}, Score: 0.4},
		},
	}
	emb := &fakeEmbedder{dim: 2, embeds: map[string][]float32{"q": {1, 1}}}
	rf := NewRelevanceFeedback(searcher, getter, emb)

	fb := NewFeedback("q", map[string]Label{
		"A": LabelRelevant,
		"B": LabelNotRelevant,
	})

	results, adjusted, err := rf.ExpandAndRetrieve(context.Background(), "q", fb, 5)
	require.NoError(t, err)
	require.Len(t, results, 3)

	// The searcher received the Rocchio-adjusted (normalized) query vector.
	// raw = [1,1] + 0.5*[1,0] - 0.3*[0,1] = [1.5,0.7]; normalized ≈ [0.9062,0.4229]
	require.NotNil(t, searcher.lastQuery)
	assert.InDelta(t, 1.5/math.Sqrt(1.5*1.5+0.7*0.7), searcher.lastQuery[0], 1e-4)
	assert.InDelta(t, 0.7/math.Sqrt(1.5*1.5+0.7*0.7), searcher.lastQuery[1], 1e-4)
	assert.Equal(t, searcher.lastQuery, adjusted)

	// Relevant chunk A is boosted to the front, order otherwise preserved.
	assert.Equal(t, []string{"A", "C", "B"}, []string{results[0].Chunk.ID, results[1].Chunk.ID, results[2].Chunk.ID})
	assert.Equal(t, 1, searcher.calls)
}

func TestExpandAndRetrieve_NoJudgment(t *testing.T) {
	rf := NewRelevanceFeedback(&fakeSearcher{}, &fakeGetter{}, &fakeEmbedder{dim: 2})
	fb := NewFeedback("q", map[string]Label{})
	_, _, err := rf.ExpandAndRetrieve(context.Background(), "q", fb, 5)
	assert.ErrorIs(t, err, ErrNoFeedback)
}

func TestExpandAndRetrieve_MissingChunks(t *testing.T) {
	// Feedback references chunks that the getter cannot resolve -> no usable feedback.
	rf := NewRelevanceFeedback(&fakeSearcher{}, &fakeGetter{chunks: map[string]*core.Chunk{}}, &fakeEmbedder{dim: 2})
	fb := NewFeedback("q", map[string]Label{"ghost": LabelRelevant})
	_, _, err := rf.ExpandAndRetrieve(context.Background(), "q", fb, 5)
	assert.ErrorIs(t, err, ErrNoFeedback)
}

func TestExpandAndRetrieve_EmbedError(t *testing.T) {
	emb := &fakeEmbedder{dim: 2, embedFn: func(ctx context.Context, text string) ([]float32, error) {
		return nil, errors.New("embed down")
	}}
	rf := NewRelevanceFeedback(&fakeSearcher{}, &fakeGetter{}, emb)
	fb := NewFeedback("q", map[string]Label{"A": LabelRelevant})
	_, _, err := rf.ExpandAndRetrieve(context.Background(), "q", fb, 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "embed down")
}

func TestExpandAndRetrieve_MissingDeps(t *testing.T) {
	rf := &RelevanceFeedback{Params: DefaultRocchioParams()}
	_, _, err := rf.ExpandAndRetrieve(context.Background(), "q", NewFeedback("q", map[string]Label{"a": LabelRelevant}), 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requires a searcher and an embedder")
}

func TestAdjustText(t *testing.T) {
	rf := NewRelevanceFeedback(nil, nil, nil)
	got := rf.AdjustText("the cat sat", []string{"cat cat"}, []string{"dog"})
	assert.Contains(t, got, "cat")
}
