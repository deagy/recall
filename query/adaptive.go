package query

import (
	"context"
	"math"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

// AdaptiveRetriever adjusts retrieval strategies based on query characteristics.
type AdaptiveRetriever struct {
	store    store.Store
	parser   QueryParser
	expander Expander

	// minRelevanceThreshold is the minimum score for a chunk to be included.
	minRelevanceThreshold float64

	// maxRetries is the maximum number of retrieval attempts.
	maxRetries int

	// feedbackWeights are weights for relevance feedback.
	feedbackWeights map[string]float64
}

// NewAdaptiveRetriever creates a new AdaptiveRetriever.
func NewAdaptiveRetriever(s store.Store, p QueryParser, e Expander) *AdaptiveRetriever {
	return &AdaptiveRetriever{
		store:                 s,
		parser:                p,
		expander:              e,
		minRelevanceThreshold: 0.1,
		maxRetries:            3,
		feedbackWeights: map[string]float64{
			"factual":     1.0,
			"comparative": 1.2,
			"temporal":    1.1,
			"causal":      1.3,
			"procedural":  1.0,
			"existential": 0.8,
		},
	}
}

// Retrieve performs adaptive retrieval based on query characteristics.
func (r *AdaptiveRetriever) Retrieve(ctx context.Context, query string, topK int) ([]index.SearchResult, error) {
	// Parse the query
	parsed, err := r.parser.Parse(ctx, query)
	if err != nil {
		return nil, err
	}
	if parsed == nil {
		return nil, nil
	}

	// Expand the query
	expanded, err := r.expander.Expand(ctx, parsed)
	if err != nil {
		return nil, err
	}

	// Determine retrieval strategy based on intent
	opts := r.determineSearchOptions(expanded, topK)

	// Perform retrieval with retries
	var results []index.SearchResult
	for attempt := 0; attempt < r.maxRetries; attempt++ {
		results, err = r.performSearch(ctx, query, opts)
		if err != nil {
			return nil, err
		}

		// Check if we have enough relevant results
		if len(results) > 0 && r.isSufficient(results, opts) {
			break
		}

		// Adjust options for next attempt
		opts = r.adjustOptions(opts, attempt)
	}

	return results, nil
}

// determineSearchOptions determines search options based on query intent.
func (r *AdaptiveRetriever) determineSearchOptions(parsed *ParsedQuery, topK int) index.SearchOptions {
	opts := index.DefaultSearchOptions(topK)

	// Adjust topK based on intent
	if weight, ok := r.feedbackWeights[string(parsed.Intent)]; ok {
		opts.TopK = int(float64(topK) * weight)
		if opts.TopK < 1 {
			opts.TopK = 1
		}
	}

	// Adjust min score based on intent
	switch parsed.Intent {
	case IntentComparative:
		opts.MinScore = 0.3 // Higher threshold for comparisons
	case IntentExistential:
		opts.MinScore = 0.2 // Lower threshold for existence checks
	default:
		opts.MinScore = r.minRelevanceThreshold
	}

	// Use hybrid search for complex queries
	if len(parsed.Entities) > 1 || len(parsed.Relations) > 0 {
		opts.Hybrid = true
		opts.BM25Weight = 0.5
	}

	// Add filters if present
	for _, f := range parsed.Filters {
		opts.Filters = append(opts.Filters, &TermFilter{Key: f.Key, Op: f.Op, Value: f.Value})
	}

	return opts
}

// performSearch performs the actual search.
func (r *AdaptiveRetriever) performSearch(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if opts.Hybrid {
		return r.store.SearchHybrid(ctx, query, opts)
	}
	return r.store.Search(ctx, query, opts)
}

// isSufficient checks if we have enough relevant results.
func (r *AdaptiveRetriever) isSufficient(results []index.SearchResult, opts index.SearchOptions) bool {
	if len(results) == 0 {
		return false
	}

	// Check if top result meets threshold
	if results[0].Score < opts.MinScore {
		return false
	}

	// Check if we have enough results
	return len(results) >= int(float64(opts.TopK)*0.5)
}

// adjustOptions adjusts search options for retry attempts.
func (r *AdaptiveRetriever) adjustOptions(opts index.SearchOptions, attempt int) index.SearchOptions {
	// Lower the threshold on each attempt
	opts.MinScore = math.Max(0.0, opts.MinScore-0.1*float64(attempt+1))

	// Increase topK on each attempt
	opts.TopK = int(float64(opts.TopK) * 1.5)

	// Always use hybrid search on retries
	opts.Hybrid = true

	return opts
}

// TermFilter implements index.Filter for structured filter support.
type TermFilter struct {
	Key   string
	Op    string
	Value interface{}
}

// Match returns true if the chunk's metadata matches this filter.
func (f *TermFilter) Match(chunk *core.Chunk) bool {
	if f.Key == "year" {
		if yearVal, ok := chunk.Metadata["year"]; ok {
			if yearStr, ok := yearVal.(core.String); ok {
				switch f.Op {
				case "eq":
					return yearStr.Value == f.Value.(string)
				case "ne":
					return yearStr.Value != f.Value.(string)
				case "gt":
					return yearStr.Value > f.Value.(string)
				case "lt":
					return yearStr.Value < f.Value.(string)
				}
			}
		}
	}
	return false
}
