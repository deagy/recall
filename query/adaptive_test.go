package query

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

func TestAdaptiveRetriever_Retrieve_Factual(t *testing.T) {
	// Create a mock store
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	// Upload some test documents
	doc := &core.Document{
		ID:    "doc1",
		Title: "Go Programming",
	}
	s.Upload(context.Background(), doc, "Go is a statically typed, compiled programming language.")

	doc2 := &core.Document{
		ID:    "doc2",
		Title: "Python Programming",
	}
	s.Upload(context.Background(), doc2, "Python is a high-level, interpreted programming language.")

	// Create parser and expander
	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	// Retrieve for factual query
	results, err := retriever.Retrieve(context.Background(), "What is Go?", 5)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Error("expected results for factual query")
	}
}

func TestAdaptiveRetriever_Retrieve_Comparative(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	doc := &core.Document{ID: "doc1", Title: "Go"}
	s.Upload(context.Background(), doc, "Go is a compiled language with strong typing.")

	doc2 := &core.Document{ID: "doc2", Title: "Python"}
	s.Upload(context.Background(), doc2, "Python is an interpreted language with dynamic typing.")

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	results, err := retriever.Retrieve(context.Background(), "Go vs Python", 5)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Error("expected results for comparative query")
	}
}

func TestAdaptiveRetriever_Retrieve_EmptyQuery(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	results, err := retriever.Retrieve(context.Background(), "", 5)
	if err != nil {
		t.Fatal(err)
	}

	if results != nil {
		t.Error("expected nil results for empty query")
	}
}

func TestAdaptiveRetriever_DetermineSearchOptions_Factual(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	parsed := &ParsedQuery{
		Original: "What is Go?",
		Intent:   IntentFactual,
		Entities: []ExtractedEntity{{Text: "Go", Type: "language", Confidence: 0.9}},
	}

	opts := retriever.determineSearchOptions(parsed, 10)
	if opts.TopK != 10 {
		t.Errorf("expected topK=10, got %d", opts.TopK)
	}
}

func TestAdaptiveRetriever_DetermineSearchOptions_Comparative(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	parsed := &ParsedQuery{
		Original: "Go vs Python",
		Intent:   IntentComparative,
		Entities: []ExtractedEntity{
			{Text: "Go", Type: "language", Confidence: 0.9},
			{Text: "Python", Type: "language", Confidence: 0.9},
		},
	}

	opts := retriever.determineSearchOptions(parsed, 10)
	// Comparative should have higher topK
	if opts.TopK < 10 {
		t.Errorf("expected topK >= 10 for comparative, got %d", opts.TopK)
	}
	// Should use hybrid search
	if !opts.Hybrid {
		t.Error("expected hybrid search for comparative query")
	}
}

func TestAdaptiveRetriever_DetermineSearchOptions_WithFilters(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	parsed := &ParsedQuery{
		Original: "What happened in 2020?",
		Intent:   IntentTemporal,
		Filters:  []Filter{{Key: "year", Op: "eq", Value: "2020"}},
	}

	opts := retriever.determineSearchOptions(parsed, 10)
	if len(opts.Filters) != 1 {
		t.Errorf("expected 1 filter, got %d", len(opts.Filters))
	}
}

func TestAdaptiveRetriever_IsSufficient_Empty(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	opts := index.DefaultSearchOptions(10)
	if retriever.isSufficient(nil, opts) {
		t.Error("expected false for empty results")
	}
}

func TestAdaptiveRetriever_IsSufficient_BelowThreshold(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	opts := index.DefaultSearchOptions(10)
	opts.MinScore = 0.9
	results := []index.SearchResult{{Score: 0.5}}

	if retriever.isSufficient(results, opts) {
		t.Error("expected false for results below threshold")
	}
}

func TestAdaptiveRetriever_AdjustOptions(t *testing.T) {
	s, err := store.NewMemoryStore(store.Config{
		Embedder: embedder.NewMockEmbedder(32),
	})
	if err != nil {
		t.Fatal(err)
	}

	parser := NewDefaultParser(nil)
	expander := NewGraphExpander(nil)
	retriever := NewAdaptiveRetriever(s, parser, expander)

	opts := index.DefaultSearchOptions(10)
	opts.MinScore = 0.5

	// First adjustment
	opts = retriever.adjustOptions(opts, 0)
	if opts.MinScore >= 0.5 {
		t.Error("expected min score to decrease")
	}
	if opts.TopK <= 10 {
		t.Error("expected topK to increase")
	}
	if !opts.Hybrid {
		t.Error("expected hybrid to be true")
	}
}
