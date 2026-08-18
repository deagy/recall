package distributed

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/deagy/recall/index"
)

// ScatterGatherConfig holds configuration for scatter-gather search.
type ScatterGatherConfig struct {
	// FanOut is the number of shards to query in parallel.
	FanOut int

	// MaxResultsPerShard is the maximum number of results to collect from each shard.
	MaxResultsPerShard int

	// TotalResults is the total number of results to return.
	TotalResults int

	// Timeout is the maximum time to wait for all shards to respond.
	Timeout int64 // milliseconds
}

// DefaultScatterGatherConfig returns a default scatter-gather configuration.
func DefaultScatterGatherConfig() *ScatterGatherConfig {
	return &ScatterGatherConfig{
		FanOut:             0, // 0 means query all shards
		MaxResultsPerShard: 100,
		TotalResults:       20,
		Timeout:            5000,
	}
}

// ScatterGatherSearch performs a scatter-gather search across multiple shards.
func ScatterGatherSearch(ctx context.Context, sm *ShardManager, query []float32, opts index.SearchOptions, config *ScatterGatherConfig) ([]index.SearchResult, error) {
	if config == nil {
		config = DefaultScatterGatherConfig()
	}

	activeShards := sm.GetActiveShards()

	if config.FanOut > 0 && len(activeShards) > config.FanOut {
		activeShards = activeShards[:config.FanOut]
	}

	if len(activeShards) == 0 {
		return nil, fmt.Errorf("no active shards available")
	}

	// Fan out to all shards
	type shardResult struct {
		shardID string
		results []index.SearchResult
		err     error
	}

	resultsCh := make(chan shardResult, len(activeShards))
	var wg sync.WaitGroup

	for _, shard := range activeShards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()

			results, err := s.Search(ctx, query, opts)
			resultsCh <- shardResult{
				shardID: s.ID,
				results: results,
				err:     err,
			}
		}(shard)
	}

	// Wait for all shards to respond
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect results
	var allResults []index.SearchResult
	var lastErr error

	for result := range resultsCh {
		if result.err != nil {
			lastErr = result.err
			continue
		}
		allResults = append(allResults, result.results...)
	}

	if lastErr != nil {
		return allResults, lastErr
	}

	// Sort by score (descending)
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	// Limit results
	if len(allResults) > config.TotalResults {
		allResults = allResults[:config.TotalResults]
	}

	return allResults, nil
}

// ScatterGatherSearchHybrid performs a scatter-gather hybrid search across multiple shards.
func ScatterGatherSearchHybrid(ctx context.Context, sm *ShardManager, query []float32, opts index.SearchOptions, config *ScatterGatherConfig) ([]index.SearchResult, error) {
	if config == nil {
		config = DefaultScatterGatherConfig()
	}

	activeShards := sm.GetActiveShards()

	if config.FanOut > 0 && len(activeShards) > config.FanOut {
		activeShards = activeShards[:config.FanOut]
	}

	if len(activeShards) == 0 {
		return nil, fmt.Errorf("no active shards available")
	}

	// Fan out to all shards
	type shardResult struct {
		shardID string
		results []index.SearchResult
		err     error
	}

	resultsCh := make(chan shardResult, len(activeShards))
	var wg sync.WaitGroup

	for _, shard := range activeShards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()

			results, err := s.SearchHybrid(ctx, query, opts)
			resultsCh <- shardResult{
				shardID: s.ID,
				results: results,
				err:     err,
			}
		}(shard)
	}

	// Wait for all shards to respond
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect results
	var allResults []index.SearchResult
	var lastErr error

	for result := range resultsCh {
		if result.err != nil {
			lastErr = result.err
			continue
		}
		allResults = append(allResults, result.results...)
	}

	if lastErr != nil {
		return allResults, lastErr
	}

	// Sort by score (descending)
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	// Limit results
	if len(allResults) > config.TotalResults {
		allResults = allResults[:config.TotalResults]
	}

	return allResults, nil
}
