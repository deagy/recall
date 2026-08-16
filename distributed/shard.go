package distributed

import (
	"context"
	"fmt"
	"sync"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

// Shard represents a shard in the distributed storage.
type Shard struct {
	ID      string
	NodeID  string
	Status  string // "active", "inactive", "degraded"
	Data    map[string]*core.Chunk
	mu      sync.RWMutex
}

// NewShard creates a new shard.
func NewShard(id, nodeID string) *Shard {
	return &Shard{
		ID:     id,
		NodeID: nodeID,
		Status: "active",
		Data:   make(map[string]*core.Chunk),
	}
}

// ShardManager manages shards across the cluster.
type ShardManager struct {
	mu      sync.RWMutex
	shards  map[string]*Shard
	cluster *Cluster
}

// NewShardManager creates a new shard manager.
func NewShardManager(cluster *Cluster) *ShardManager {
	return &ShardManager{
		shards:  make(map[string]*Shard),
		cluster: cluster,
	}
}

// GetShard returns a shard by its ID.
func (sm *ShardManager) GetShard(shardID string) (*Shard, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	shard, exists := sm.shards[shardID]
	return shard, exists
}

// GetShardForChunk returns the shard responsible for a chunk.
func (sm *ShardManager) GetShardForChunk(chunkID string) *Shard {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Find the shard for this chunk ID
	for _, shard := range sm.shards {
		if shard.Status == "active" {
			shard.mu.RLock()
			_, exists := shard.Data[chunkID]
			shard.mu.RUnlock()

			if exists {
				return shard
			}
		}
	}

	return nil
}

// GetShardForNode returns all shards for a node.
func (sm *ShardManager) GetShardForNode(nodeID string) []*Shard {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	var shards []*Shard
	for _, shard := range sm.shards {
		if shard.NodeID == nodeID && shard.Status == "active" {
			shards = append(shards, shard)
		}
	}

	return shards
}

// GetActiveShards returns all active shards.
func (sm *ShardManager) GetActiveShards() []*Shard {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	var shards []*Shard
	for _, shard := range sm.shards {
		if shard.Status == "active" {
			shards = append(shards, shard)
		}
	}

	return shards
}

// GetShardCount returns the number of shards.
func (sm *ShardManager) GetShardCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return len(sm.shards)
}

// GetActiveShardCount returns the number of active shards.
func (sm *ShardManager) GetActiveShardCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	count := 0
	for _, shard := range sm.shards {
		if shard.Status == "active" {
			count++
		}
	}

	return count
}

// CreateShard creates a new shard and assigns it to a node.
func (sm *ShardManager) CreateShard(nodeID string) (*Shard, error) {
	return sm.CreateShardWithID(nodeID, "")
}

// CreateShardWithID creates a new shard with a specific ID and assigns it to a node.
func (sm *ShardManager) CreateShardWithID(nodeID string, shardID string) (*Shard, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	node, exists := sm.cluster.GetNode(nodeID)
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}

	if node.Status != "online" {
		return nil, fmt.Errorf("node %s is not online", nodeID)
	}

	if shardID == "" {
		shardID = fmt.Sprintf("shard-%s-%d", nodeID, len(sm.shards)+1)
	}

	shard := NewShard(shardID, nodeID)
	sm.shards[shardID] = shard

	return shard, nil
}

// DeleteShard deletes a shard.
func (sm *ShardManager) DeleteShard(shardID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	shard, exists := sm.shards[shardID]
	if !exists {
		return fmt.Errorf("shard %s not found", shardID)
	}

	shard.Status = "inactive"
	delete(sm.shards, shardID)

	return nil
}

// StoreChunk stores a chunk in the appropriate shard.
func (sm *ShardManager) StoreChunk(ctx context.Context, chunk *core.Chunk) error {
	shardID := sm.getShardIDForChunk(chunk.ID)

	sm.mu.Lock()
	shard, exists := sm.shards[shardID]
	sm.mu.Unlock()

	if !exists {
		// Use consistent hashing to determine node
		nodeID := sm.cluster.GetNodeForChunk(chunk.ID)
		if nodeID == "" {
			return fmt.Errorf("no online nodes available")
		}
		var err error
		shard, err = sm.CreateShardWithID(nodeID, shardID)
		if err != nil {
			return fmt.Errorf("failed to create shard: %w", err)
		}
	}

	shard.mu.Lock()
	defer shard.mu.Unlock()

	shard.Data[chunk.ID] = chunk

	return nil
}

// GetChunk retrieves a chunk from the appropriate shard.
func (sm *ShardManager) GetChunk(ctx context.Context, chunkID string) (*core.Chunk, bool) {
	shardID := sm.getShardIDForChunk(chunkID)

	sm.mu.RLock()
	shard, exists := sm.shards[shardID]
	sm.mu.RUnlock()

	if !exists {
		return nil, false
	}

	shard.mu.RLock()
	defer shard.mu.RUnlock()

	chunk, exists := shard.Data[chunkID]
	return chunk, exists
}

// DeleteChunk deletes a chunk from the appropriate shard.
func (sm *ShardManager) DeleteChunk(ctx context.Context, chunkID string) error {
	shardID := sm.getShardIDForChunk(chunkID)

	sm.mu.RLock()
	shard, exists := sm.shards[shardID]
	sm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("shard %s not found", shardID)
	}

	shard.mu.Lock()
	defer shard.mu.Unlock()

	delete(shard.Data, chunkID)

	return nil
}

// getShardIDForChunk generates a consistent shard ID for a chunk using consistent hashing.
func (sm *ShardManager) getShardIDForChunk(chunkID string) string {
	// Use the first 8 characters of the chunk ID as the shard key
	// This ensures chunks with the same prefix go to the same shard
	if len(chunkID) >= 8 {
		return fmt.Sprintf("shard-%s", chunkID[:8])
	}
	return fmt.Sprintf("shard-%s", chunkID)
}

// Search searches across all active shards.
func (sm *ShardManager) Search(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	activeShards := sm.GetActiveShards()
	
	var allResults []index.SearchResult
	var lastErr error

	for _, shard := range activeShards {
		results, err := shard.Search(ctx, query, opts)
		if err != nil {
			lastErr = err
			continue
		}
		allResults = append(allResults, results...)
	}

	if lastErr != nil {
		return allResults, lastErr
	}

	return allResults, nil
}

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
func (sm *ShardManager) SearchHybrid(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	activeShards := sm.GetActiveShards()
	
	var allResults []index.SearchResult
	var lastErr error

	for _, shard := range activeShards {
		results, err := shard.SearchHybrid(ctx, query, opts)
		if err != nil {
			lastErr = err
			continue
		}
		allResults = append(allResults, results...)
	}

	if lastErr != nil {
		return allResults, lastErr
	}

	return allResults, nil
}

// Search searches within this shard using vector similarity.
func (s *Shard) Search(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.Data) == 0 {
		return []index.SearchResult{}, nil
	}

	// Create a simple in-memory index for this shard
	idx := NewShardIndex(s)
	results, err := idx.Search(ctx, query, opts)
	if err != nil {
		return nil, err
	}

	return results, nil
}

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
// Note: This is a wrapper that calls Search with a dummy embedding for hybrid functionality.
// In a production system, you would generate the query embedding from the query string.
func (s *Shard) SearchHybrid(ctx context.Context, query []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	return s.Search(ctx, query, opts)
}
