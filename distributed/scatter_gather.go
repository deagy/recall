package distributed

import (
	"context"
	"fmt"
	"sync"
	"time"

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

// scatterGather fans a per-shard search function out over the active shards
// in parallel and merges the results. When config.Timeout is positive the
// fan-out runs under a derived context that expires after that many
// milliseconds (or earlier if ctx already carries a closer deadline).
func scatterGather(ctx context.Context, sm *ShardManager, config *ScatterGatherConfig, searchShard func(context.Context, *Shard) ([]index.SearchResult, error)) ([]index.SearchResult, error) {
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

	// Apply the configured timeout without shortening an earlier deadline
	// the caller may have set.
	if config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(config.Timeout)*time.Millisecond)
		defer cancel()
	}

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

			results, err := searchShard(ctx, s)
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
		if config.MaxResultsPerShard > 0 && len(result.results) > config.MaxResultsPerShard {
			result.results = result.results[:config.MaxResultsPerShard]
		}
		allResults = append(allResults, result.results...)
	}

	if lastErr != nil {
		return allResults, lastErr
	}

	// Sort by score (descending) with a deterministic tie-break so the merged
	// ordering does not depend on shard iteration order.
	sortSearchResults(allResults)

	// Limit results
	if len(allResults) > config.TotalResults {
		allResults = allResults[:config.TotalResults]
	}

	return allResults, nil
}

// ScatterGatherSearch performs a scatter-gather search across multiple shards.
func ScatterGatherSearch(ctx context.Context, sm *ShardManager, query []float32, opts index.SearchOptions, config *ScatterGatherConfig) ([]index.SearchResult, error) {
	return scatterGather(ctx, sm, config, func(ctx context.Context, s *Shard) ([]index.SearchResult, error) {
		return s.Search(ctx, query, opts)
	})
}

// ScatterGatherSearchHybrid performs a scatter-gather hybrid search across
// multiple shards.
func ScatterGatherSearchHybrid(ctx context.Context, sm *ShardManager, query []float32, opts index.SearchOptions, config *ScatterGatherConfig) ([]index.SearchResult, error) {
	return scatterGather(ctx, sm, config, func(ctx context.Context, s *Shard) ([]index.SearchResult, error) {
		return s.SearchHybrid(ctx, query, opts)
	})
}
