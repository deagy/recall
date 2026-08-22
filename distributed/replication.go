package distributed

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/deagy/recall/core"
)

// ReplicationStrategy defines the strategy for data replication.
type ReplicationStrategy string

const (
	// StrategyPrimaryReplica uses a primary node with replica nodes.
	StrategyPrimaryReplica ReplicationStrategy = "primary_replica"

	// StrategyQuorum uses a quorum-based approach.
	StrategyQuorum ReplicationStrategy = "quorum"

	// StrategyAllNodes replicates to all nodes.
	StrategyAllNodes ReplicationStrategy = "all_nodes"
)

// ReplicationResult represents the result of a replication operation.
type ReplicationResult struct {
	ShardID   string `json:"shard_id"`
	NodeID    string `json:"node_id"`
	Success   bool   `json:"success"`
	Error     string `json:"error,omitempty"`
	ReplicaID string `json:"replica_id,omitempty"`
}

// ReplicationManager manages data replication across the cluster.
type ReplicationManager struct {
	mu                sync.RWMutex
	cluster           *Cluster
	shardManager      *ShardManager
	strategy          ReplicationStrategy
	replicationFactor int
}

// NewReplicationManager creates a new replication manager.
func NewReplicationManager(cluster *Cluster, shardManager *ShardManager, strategy ReplicationStrategy, replicationFactor int) *ReplicationManager {
	if strategy == "" {
		strategy = StrategyPrimaryReplica
	}
	if replicationFactor <= 0 {
		replicationFactor = 3
	}

	return &ReplicationManager{
		cluster:           cluster,
		shardManager:      shardManager,
		strategy:          strategy,
		replicationFactor: replicationFactor,
	}
}

// ReplicateData replicates data to the appropriate nodes based on the strategy.
func (rm *ReplicationManager) ReplicateData(ctx context.Context, shardID string, data map[string]*core.Chunk) ([]ReplicationResult, error) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	var results []ReplicationResult

	switch rm.strategy {
	case StrategyPrimaryReplica:
		results = rm.replicatePrimaryReplica(ctx, shardID, data)
	case StrategyQuorum:
		results = rm.replicateQuorum(ctx, shardID, data)
	case StrategyAllNodes:
		results = rm.replicateAllNodes(ctx, shardID, data)
	default:
		return nil, fmt.Errorf("unknown replication strategy: %s", rm.strategy)
	}

	return results, nil
}

// getOrCreateReplicaShard returns the deterministic replica shard for nodeID,
// creating it on first use. Reusing an existing replica shard makes
// replication idempotent: repeated ReplicateData calls update the same shard
// instead of accumulating unbounded duplicates, and the reported ReplicaID
// always resolves to a real shard.
func (rm *ReplicationManager) getOrCreateReplicaShard(nodeID, replicaShardID string) (*Shard, error) {
	if shard, exists := rm.shardManager.GetShard(replicaShardID); exists {
		return shard, nil
	}
	shard, err := rm.shardManager.CreateShardWithID(nodeID, replicaShardID)
	if err != nil {
		if !errors.Is(err, ErrShardExists) {
			return nil, err
		}
		// A concurrent replication created the same shard first; reuse it.
		shard, exists := rm.shardManager.GetShard(replicaShardID)
		if !exists {
			return nil, err
		}
		return shard, nil
	}
	return shard, nil
}

// storeInShard copies data into shard under its write lock.
func storeInShard(shard *Shard, data map[string]*core.Chunk) {
	shard.mu.Lock()
	for k, v := range data {
		shard.Data[k] = v
	}
	shard.mu.Unlock()
}

// replicatePrimaryReplica replicates data to a primary node and its replicas.
func (rm *ReplicationManager) replicatePrimaryReplica(ctx context.Context, shardID string, data map[string]*core.Chunk) []ReplicationResult {
	var results []ReplicationResult

	// Get the primary shard
	primaryShard, exists := rm.shardManager.GetShard(shardID)
	if !exists {
		results = append(results, ReplicationResult{
			ShardID: shardID,
			Error:   "primary shard not found",
		})
		return results
	}

	// Store data in primary shard
	primaryShard.mu.Lock()
	for k, v := range data {
		primaryShard.Data[k] = v
	}
	primaryShard.mu.Unlock()

	results = append(results, ReplicationResult{
		ShardID: shardID,
		NodeID:  primaryShard.NodeID,
		Success: true,
	})

	// Get replica nodes. The ring's first node for this key is not
	// necessarily the node hosting the primary shard, so the primary is
	// skipped by node ID rather than by list position.
	replicaNodes := rm.cluster.GetReplicaNodes(shardID)
	for _, node := range replicaNodes {
		if node.ID == primaryShard.NodeID {
			continue // primary shard already holds the data
		}
		replicaShardID := fmt.Sprintf("%s-replica-%s", shardID, node.ID)
		replicaShard, err := rm.getOrCreateReplicaShard(node.ID, replicaShardID)
		if err != nil {
			results = append(results, ReplicationResult{
				ShardID: shardID,
				NodeID:  node.ID,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		storeInShard(replicaShard, data)

		results = append(results, ReplicationResult{
			ShardID:   shardID,
			NodeID:    node.ID,
			Success:   true,
			ReplicaID: replicaShardID,
		})
	}

	return results
}

// replicateQuorum replicates data using a quorum-based approach.
func (rm *ReplicationManager) replicateQuorum(ctx context.Context, shardID string, data map[string]*core.Chunk) []ReplicationResult {
	var results []ReplicationResult

	// Get all nodes
	allNodes := rm.cluster.GetOnlineNodes()
	if len(allNodes) == 0 {
		results = append(results, ReplicationResult{
			ShardID: shardID,
			Error:   "no online nodes available",
		})
		return results
	}

	// Determine quorum size (majority)
	quorumSize := len(allNodes)/2 + 1

	// Replicate to quorum nodes
	for i := 0; i < quorumSize && i < len(allNodes); i++ {
		node := allNodes[i]
		replicaShardID := fmt.Sprintf("%s-quorum-%s", shardID, node.ID)
		replicaShard, err := rm.getOrCreateReplicaShard(node.ID, replicaShardID)
		if err != nil {
			results = append(results, ReplicationResult{
				ShardID: shardID,
				NodeID:  node.ID,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		storeInShard(replicaShard, data)

		results = append(results, ReplicationResult{
			ShardID:   shardID,
			NodeID:    node.ID,
			Success:   true,
			ReplicaID: replicaShardID,
		})
	}

	return results
}

// replicateAllNodes replicates data to all nodes.
func (rm *ReplicationManager) replicateAllNodes(ctx context.Context, shardID string, data map[string]*core.Chunk) []ReplicationResult {
	var results []ReplicationResult

	allNodes := rm.cluster.GetOnlineNodes()
	if len(allNodes) == 0 {
		results = append(results, ReplicationResult{
			ShardID: shardID,
			Error:   "no online nodes available",
		})
		return results
	}

	for _, node := range allNodes {
		replicaShardID := fmt.Sprintf("%s-all-%s", shardID, node.ID)
		replicaShard, err := rm.getOrCreateReplicaShard(node.ID, replicaShardID)
		if err != nil {
			results = append(results, ReplicationResult{
				ShardID: shardID,
				NodeID:  node.ID,
				Success: false,
				Error:   err.Error(),
			})
			continue
		}

		storeInShard(replicaShard, data)

		results = append(results, ReplicationResult{
			ShardID:   shardID,
			NodeID:    node.ID,
			Success:   true,
			ReplicaID: replicaShardID,
		})
	}

	return results
}

// GetReplicationStatus returns the replication status for a shard.
func (rm *ReplicationManager) GetReplicationStatus(ctx context.Context, shardID string) (int, int, error) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	_, exists := rm.shardManager.GetShard(shardID)
	if !exists {
		return 0, rm.replicationFactor, fmt.Errorf("shard %s not found", shardID)
	}

	// Count replicas using the deterministic ID scheme of the active
	// strategy: <shard>-<kind>-<node>.
	kind := "replica"
	switch rm.strategy {
	case StrategyQuorum:
		kind = "quorum"
	case StrategyAllNodes:
		kind = "all"
	}

	replicaCount := 0
	allNodes := rm.cluster.GetOnlineNodes()
	for _, node := range allNodes {
		replicaShardID := fmt.Sprintf("%s-%s-%s", shardID, kind, node.ID)
		if _, exists := rm.shardManager.GetShard(replicaShardID); exists {
			replicaCount++
		}
	}

	return replicaCount, rm.replicationFactor, nil
}
