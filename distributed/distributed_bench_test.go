package distributed

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

func BenchmarkCluster_AddNode(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.AddNode(node)
	}
}

func BenchmarkCluster_RemoveNode(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.RemoveNode("node-1")
	}
}

func BenchmarkCluster_GetAllNodes(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node1 := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	node2 := &Node{
		ID:      "node-2",
		Address: "localhost:8081",
	}
	cluster.AddNode(node1)
	cluster.AddNode(node2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.GetAllNodes()
	}
}

func BenchmarkCluster_GetOnlineNodes(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node1 := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	node2 := &Node{
		ID:      "node-2",
		Address: "localhost:8081",
	}
	cluster.AddNode(node1)
	cluster.AddNode(node2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.GetOnlineNodes()
	}
}

func BenchmarkCluster_GetReplicaNodes(b *testing.B) {
	config := DefaultClusterConfig()
	config.ReplicationFactor = 3
	cluster := NewCluster(config)

	node1 := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	node2 := &Node{
		ID:      "node-2",
		Address: "localhost:8081",
	}
	node3 := &Node{
		ID:      "node-3",
		Address: "localhost:8082",
	}
	cluster.AddNode(node1)
	cluster.AddNode(node2)
	cluster.AddNode(node3)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster.GetReplicaNodes("test-key")
	}
}

func BenchmarkShardManager_CreateShard(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		shardMgr.CreateShard("node-1")
	}
}

func BenchmarkShardManager_StoreChunk(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)
	shard, _ := shardMgr.CreateShard("node-1")

	chunk := &core.Chunk{
		ID:      "chunk-1",
		Content: "Test chunk",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		shard.mu.Lock()
		shard.Data[chunk.ID] = chunk
		shard.mu.Unlock()
	}
}

func BenchmarkShardManager_GetChunk(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)
	shard, _ := shardMgr.CreateShard("node-1")

	chunk := &core.Chunk{
		ID:      "chunk-1",
		Content: "Test chunk",
	}

	shard.mu.Lock()
	shard.Data[chunk.ID] = chunk
	shard.mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		shardMgr.GetChunk(context.Background(), "chunk-1")
	}
}

func BenchmarkScatterGatherSearch(b *testing.B) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node1 := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	node2 := &Node{
		ID:      "node-2",
		Address: "localhost:8081",
	}
	cluster.AddNode(node1)
	cluster.AddNode(node2)

	shardMgr := NewShardManager(cluster)
	shardMgr.CreateShard("node-1")
	shardMgr.CreateShard("node-2")

	opts := index.SearchOptions{
		TopK: 10,
	}

	scgConfig := &ScatterGatherConfig{
		FanOut:             0,
		MaxResultsPerShard: 100,
		TotalResults:       20,
		Timeout:            5000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScatterGatherSearch(context.Background(), shardMgr, nil, opts, scgConfig)
	}
}

func BenchmarkReplicationManager_ReplicateData(b *testing.B) {
	config := DefaultClusterConfig()
	config.ReplicationFactor = 2
	cluster := NewCluster(config)

	node1 := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	node2 := &Node{
		ID:      "node-2",
		Address: "localhost:8081",
	}
	cluster.AddNode(node1)
	cluster.AddNode(node2)

	shardMgr := NewShardManager(cluster)
	shardMgr.CreateShard("node-1")

	replicationMgr := NewReplicationManager(cluster, shardMgr, StrategyPrimaryReplica, 2)

	data := map[string]*core.Chunk{
		"chunk-1": {ID: "chunk-1", Content: "Test chunk"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		replicationMgr.ReplicateData(context.Background(), "shard-1", data)
	}
}
