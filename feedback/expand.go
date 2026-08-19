package feedback

import (
	"context"
	"errors"
	"fmt"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// ErrNoFeedback is returned when a feedback round contains no usable relevance
// judgments or the referenced chunks cannot be resolved to embeddings.
var ErrNoFeedback = errors.New("feedback: no usable relevance feedback")

// VectorSearcher searches a chunk index by embedding vector. It is satisfied
// by index.Index.
type VectorSearcher interface {
	// Search finds the most similar chunks to the given query embedding.
	Search(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error)
}

// ChunkGetter returns a stored chunk by ID. It is satisfied by store.Store.
type ChunkGetter interface {
	// GetChunk returns a chunk by its ID.
	GetChunk(id string) (*core.Chunk, bool)
}

// RelevanceFeedback adjusts queries based on user relevance feedback using the
// Rocchio algorithm. It combines a vector search capability (e.g., an
// index.Index), chunk access (e.g., a store.Store), and an embedder so the
// full retrieve → feedback → re-rank loop can run.
type RelevanceFeedback struct {
	// Searcher retrieves chunks by query embedding.
	Searcher VectorSearcher

	// Getter resolves chunk IDs to chunks (for their embeddings/content).
	Getter ChunkGetter

	// Embedder embeds query text.
	Embedder embedder.Embedder

	// Params configures Rocchio (vector form).
	Params RocchioParams

	// TermParams configures Rocchio (lexical form) for AdjustText.
	TermParams TermRocchioParams

	// BoostRelevant, when true, moves chunks the user marked relevant to the
	// front of the re-ranked results.
	BoostRelevant bool
}

// NewRelevanceFeedback creates a RelevanceFeedback with standard Rocchio
// parameters and relevant-boosting enabled.
func NewRelevanceFeedback(searcher VectorSearcher, getter ChunkGetter, e embedder.Embedder) *RelevanceFeedback {
	return &RelevanceFeedback{
		Searcher:      searcher,
		Getter:        getter,
		Embedder:      e,
		Params:        DefaultRocchioParams(),
		TermParams:    DefaultTermRocchioParams(),
		BoostRelevant: true,
	}
}

// AdjustVector returns a Rocchio-adjusted query embedding given the original
// query embedding and the embeddings of the relevant and not-relevant chunks.
func (r *RelevanceFeedback) AdjustVector(query []float32, relevant, irrelevant [][]float32) []float32 {
	return Rocchio(query, relevant, irrelevant, r.Params)
}

// AdjustText returns a textually expanded query for keyword-based retrieval,
// using lexical Rocchio over the query and the relevant/not-relevant chunk
// texts.
func (r *RelevanceFeedback) AdjustText(query string, relevantTexts, irrelevantTexts []string) string {
	s, _ := RocchioTerms(query, relevantTexts, irrelevantTexts, r.TermParams)
	return s
}

// ExpandAndRetrieve performs one round of iterative relevance feedback:
//
//  1. embed the query,
//  2. gather the embeddings of the chunks judged relevant / not relevant,
//  3. apply Rocchio to obtain an adjusted query vector,
//  4. re-retrieve with the adjusted vector,
//  5. re-rank so user-marked relevant chunks come first (when BoostRelevant).
//
// It returns the re-ranked results and the adjusted query vector.
func (r *RelevanceFeedback) ExpandAndRetrieve(ctx context.Context, query string, fb *Feedback, topK int) ([]index.SearchResult, []float32, error) {
	if r.Searcher == nil || r.Embedder == nil {
		return nil, nil, errors.New("feedback: RelevanceFeedback requires a searcher and an embedder")
	}
	if fb == nil || !fb.HasJudgment() {
		return nil, nil, ErrNoFeedback
	}

	qEmb, err := r.Embedder.Embed(ctx, query)
	if err != nil {
		return nil, nil, fmt.Errorf("feedback: embed query: %w", err)
	}

	var relEmb, irrEmb [][]float32
	if r.Getter != nil {
		for _, id := range fb.Relevant() {
			if c, ok := r.Getter.GetChunk(id); ok && c != nil {
				relEmb = append(relEmb, c.Embedding)
			}
		}
		for _, id := range fb.Irrelevant() {
			if c, ok := r.Getter.GetChunk(id); ok && c != nil {
				irrEmb = append(irrEmb, c.Embedding)
			}
		}
	}
	if len(relEmb) == 0 && len(irrEmb) == 0 {
		return nil, nil, ErrNoFeedback
	}

	adjusted := r.AdjustVector(qEmb, relEmb, irrEmb)

	results, err := r.Searcher.Search(ctx, adjusted, index.DefaultSearchOptions(topK))
	if err != nil {
		return nil, nil, fmt.Errorf("feedback: re-search: %w", err)
	}

	if r.BoostRelevant && len(fb.Relevant()) > 0 {
		results = BoostRelevant(results, fb.Relevant())
	}
	return results, adjusted, nil
}

// BoostRelevant stably partitions results so those whose chunk ID is in
// relevant come first, preserving the original order within each group.
func BoostRelevant(results []index.SearchResult, relevant []string) []index.SearchResult {
	if len(relevant) == 0 {
		return results
	}
	set := make(map[string]struct{}, len(relevant))
	for _, id := range relevant {
		set[id] = struct{}{}
	}
	var front, rest []index.SearchResult
	for _, res := range results {
		if res.Chunk != nil {
			if _, ok := set[res.Chunk.ID]; ok {
				front = append(front, res)
				continue
			}
		}
		rest = append(rest, res)
	}
	out := make([]index.SearchResult, 0, len(results))
	out = append(out, front...)
	out = append(out, rest...)
	return out
}
