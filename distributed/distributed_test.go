package distributed

import (
	"context"
	"fmt"
	"testing"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

func TestCluster_AddNode(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}

	err := cluster.AddNode(node)
	if err != nil {
		t.Fatal(err)
	}

	if cluster.GetNodeCount() != 1 {
		t.Errorf("expected 1 node, got %d", cluster.GetNodeCount())
	}

	if cluster.GetOnlineNodeCount() != 1 {
		t.Errorf("expected 1 online node, got %d", cluster.GetOnlineNodeCount())
	}
}

func TestCluster_AddDuplicateNode(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}

	err := cluster.AddNode(node)
	if err != nil {
		t.Fatal(err)
	}

	err = cluster.AddNode(node)
	if err == nil {
		t.Error("expected error for duplicate node")
	}
}

func TestCluster_RemoveNode(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}

	err := cluster.AddNode(node)
	if err != nil {
		t.Fatal(err)
	}

	err = cluster.RemoveNode("node-1")
	if err != nil {
		t.Fatal(err)
	}

	if cluster.GetNodeCount() != 0 {
		t.Errorf("expected 0 nodes, got %d", cluster.GetNodeCount())
	}
}

func TestCluster_RemoveNonexistentNode(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	err := cluster.RemoveNode("node-1")
	if err == nil {
		t.Error("expected error for nonexistent node")
	}
}

func TestCluster_GetAllNodes(t *testing.T) {
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

	nodes := cluster.GetAllNodes()
	if len(nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(nodes))
	}
}

func TestCluster_GetOnlineNodes(t *testing.T) {
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

	nodes := cluster.GetOnlineNodes()
	if len(nodes) != 2 {
		t.Errorf("expected 2 online nodes, got %d", len(nodes))
	}
}

func TestCluster_GetReplicaNodes(t *testing.T) {
	clusterConfig := DefaultClusterConfig()
	clusterConfig.ReplicationFactor = 2
	cluster := NewCluster(clusterConfig)

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

	replicas := cluster.GetReplicaNodes("test-key")
	if len(replicas) == 0 {
		t.Error("expected at least one replica")
	}
}

func TestShardManager_CreateShard(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)
	shard, err := shardMgr.CreateShard("node-1")
	if err != nil {
		t.Fatal(err)
	}

	if shard == nil {
		t.Error("expected shard to be created")
	}

	if shardMgr.GetShardCount() != 1 {
		t.Errorf("expected 1 shard, got %d", shardMgr.GetShardCount())
	}
}

func TestShardManager_CreateShardForOfflineNode(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	// Manually set status to offline
	node.Status = "offline"

	shardMgr := NewShardManager(cluster)
	_, err := shardMgr.CreateShard("node-1")
	if err == nil {
		t.Error("expected error for offline node")
	}
}

func TestShardManager_DeleteShard(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)
	shard, _ := shardMgr.CreateShard("node-1")

	err := shardMgr.DeleteShard(shard.ID)
	if err != nil {
		t.Fatal(err)
	}

	if shardMgr.GetShardCount() != 0 {
		t.Errorf("expected 0 shards, got %d", shardMgr.GetShardCount())
	}
}

func TestShardManager_StoreAndGetChunk(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)

	chunk := &core.Chunk{
		ID:      "chunk-00000001",
		Content: "Test chunk",
	}

	// Store chunk (this will create the shard automatically)
	err := shardMgr.StoreChunk(context.Background(), chunk)
	if err != nil {
		t.Fatal(err)
	}

	retrieved, exists := shardMgr.GetChunk(context.Background(), "chunk-00000001")
	if !exists {
		t.Error("expected chunk to exist")
	}

	if retrieved.Content != "Test chunk" {
		t.Errorf("expected 'Test chunk', got %q", retrieved.Content)
	}
}

func TestShardManager_DeleteChunk(t *testing.T) {
	config := DefaultClusterConfig()
	cluster := NewCluster(config)

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	cluster.AddNode(node)

	shardMgr := NewShardManager(cluster)

	chunk := &core.Chunk{
		ID:      "chunk-00000001",
		Content: "Test chunk",
	}

	// Store chunk
	err := shardMgr.StoreChunk(context.Background(), chunk)
	if err != nil {
		t.Fatal(err)
	}

	// Verify it exists
	_, exists := shardMgr.GetChunk(context.Background(), "chunk-00000001")
	if !exists {
		t.Fatal("expected chunk to exist before deletion")
	}

	// Delete chunk
	err = shardMgr.DeleteChunk(context.Background(), "chunk-00000001")
	if err != nil {
		t.Fatal(err)
	}

	// Verify it's deleted
	_, exists = shardMgr.GetChunk(context.Background(), "chunk-00000001")
	if exists {
		t.Error("expected chunk to be deleted")
	}
}

func TestScatterGatherSearch(t *testing.T) {
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
	shard1, _ := shardMgr.CreateShard("node-1")
	shard2, _ := shardMgr.CreateShard("node-2")

	// Add some test data with longer IDs and embeddings
	shard1.mu.Lock()
	shard1.Data["chunk-00000001"] = &core.Chunk{
		ID:        "chunk-00000001",
		Content:   "Test chunk 1",
		Embedding: []float32{0.1, 0.2, 0.3},
	}
	shard1.mu.Unlock()

	shard2.mu.Lock()
	shard2.Data["chunk-00000002"] = &core.Chunk{
		ID:        "chunk-00000002",
		Content:   "Test chunk 2",
		Embedding: []float32{0.4, 0.5, 0.6},
	}
	shard2.mu.Unlock()

	opts := index.SearchOptions{
		TopK: 10,
	}

	results, err := ScatterGatherSearch(context.Background(), shardMgr, []float32{0.1, 0.2, 0.3}, opts, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Error("expected results")
	}
}

func TestScatterGatherSearchWithFanOut(t *testing.T) {
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
	node3 := &Node{
		ID:      "node-3",
		Address: "localhost:8082",
	}
	cluster.AddNode(node1)
	cluster.AddNode(node2)
	cluster.AddNode(node3)

	shardMgr := NewShardManager(cluster)
	shardMgr.CreateShard("node-1")
	shardMgr.CreateShard("node-2")
	shardMgr.CreateShard("node-3")

	// Add test data with embeddings
	shards := shardMgr.GetActiveShards()
	for i, shard := range shards {
		shard.mu.Lock()
		shard.Data[fmt.Sprintf("chunk-%08d", i)] = &core.Chunk{
			ID:        fmt.Sprintf("chunk-%08d", i),
			Content:   fmt.Sprintf("Test chunk %d", i),
			Embedding: []float32{float32(i + 1), float32(i + 2), float32(i + 3)},
		}
		shard.mu.Unlock()
	}

	opts := index.SearchOptions{
		TopK: 10,
	}

	scgConfig := &ScatterGatherConfig{
		FanOut:             2,
		MaxResultsPerShard: 100,
		TotalResults:       20,
		Timeout:            5000,
	}

	results, err := ScatterGatherSearch(context.Background(), shardMgr, []float32{1, 2, 3}, opts, scgConfig)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Error("expected results")
	}
}

func TestReplicationManager_ReplicateData(t *testing.T) {
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

	results, err := replicationMgr.ReplicateData(context.Background(), "shard-1", data)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Error("expected replication results")
	}
}

func TestDistributedStore_AddNode(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	ds := NewDistributedStore(nil, mockEmbedder, nil, "test-namespace")

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}

	err := ds.AddNode(node)
	if err != nil {
		t.Fatal(err)
	}

	if ds.GetCluster().GetNodeCount() != 1 {
		t.Errorf("expected 1 node, got %d", ds.GetCluster().GetNodeCount())
	}
}

func TestDistributedStore_RemoveNode(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	ds := NewDistributedStore(nil, mockEmbedder, nil, "test-namespace")

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	ds.AddNode(node)

	err := ds.RemoveNode("node-1")
	if err != nil {
		t.Fatal(err)
	}

	if ds.GetCluster().GetNodeCount() != 0 {
		t.Errorf("expected 0 nodes, got %d", ds.GetCluster().GetNodeCount())
	}
}

func TestDistributedStore_Upload(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	// Create a simple chunker factory
	chunkerFactory := func(cfg chunker.Config) chunker.Chunker {
		return &mockChunker{
			chunks: []core.Chunk{
				{ID: "chunk-00000001", Content: "Test chunk 1"},
				{ID: "chunk-00000002", Content: "Test chunk 2"},
			},
		}
	}

	ds := NewDistributedStore(nil, mockEmbedder, chunkerFactory, "test-namespace")

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	ds.AddNode(node)

	doc := &core.Document{
		ID:    "doc-1",
		Title: "Test Document",
	}

	err := ds.Upload(context.Background(), doc, "Test content")
	if err != nil {
		t.Fatal(err)
	}

	if ds.Count() != 2 {
		t.Errorf("expected 2 chunks, got %d", ds.Count())
	}
}

func TestDistributedStore_Search(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	chunkerFactory := func(cfg chunker.Config) chunker.Chunker {
		return &mockChunker{
			chunks: []core.Chunk{
				{ID: "chunk-00000001", Content: "Test chunk 1"},
				{ID: "chunk-00000002", Content: "Test chunk 2"},
			},
		}
	}

	ds := NewDistributedStore(nil, mockEmbedder, chunkerFactory, "test-namespace")

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	ds.AddNode(node)
	ds.Upload(context.Background(), &core.Document{ID: "doc-1"}, "Test content")

	opts := index.SearchOptions{
		TopK: 10,
	}

	results, err := ds.Search(context.Background(), "test", opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) == 0 {
		t.Error("expected results")
	}
}

func TestDistributedStore_GetChunk(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	chunkerFactory := func(cfg chunker.Config) chunker.Chunker {
		return &mockChunker{
			chunks: []core.Chunk{
				{ID: "chunk-00000001", Content: "Test chunk 1"},
			},
		}
	}

	ds := NewDistributedStore(nil, mockEmbedder, chunkerFactory, "test-namespace")

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	ds.AddNode(node)
	ds.Upload(context.Background(), &core.Document{ID: "doc-1"}, "Test content")

	chunk, exists := ds.GetChunk("chunk-00000001")
	if !exists {
		t.Error("expected chunk to exist")
	}

	if chunk.Content != "Test chunk 1" {
		t.Errorf("expected 'Test chunk 1', got %q", chunk.Content)
	}
}

func TestDistributedStore_DeleteChunk(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	chunkerFactory := func(cfg chunker.Config) chunker.Chunker {
		return &mockChunker{
			chunks: []core.Chunk{
				{ID: "chunk-00000001", Content: "Test chunk 1"},
			},
		}
	}

	ds := NewDistributedStore(nil, mockEmbedder, chunkerFactory, "test-namespace")

	node := &Node{
		ID:      "node-1",
		Address: "localhost:8080",
	}
	ds.AddNode(node)
	ds.Upload(context.Background(), &core.Document{ID: "doc-1"}, "Test content")

	err := ds.DeleteChunk(context.Background(), "chunk-00000001")
	if err != nil {
		t.Fatal(err)
	}

	_, exists := ds.GetChunk("chunk-00000001")
	if exists {
		t.Error("expected chunk to be deleted")
	}
}

func TestDistributedStore_Namespaces(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	ds := NewDistributedStore(nil, mockEmbedder, nil, "test-namespace")

	namespaces := ds.Namespaces()
	if len(namespaces) != 1 || namespaces[0] != "test-namespace" {
		t.Errorf("expected [test-namespace], got %v", namespaces)
	}
}

func TestDistributedStore_Close(t *testing.T) {
	mockEmbedder := embedder.NewMockEmbedder(3)

	ds := NewDistributedStore(nil, mockEmbedder, nil, "test-namespace")

	err := ds.Close()
	if err != nil {
		t.Fatal(err)
	}
}

// mockChunker is a simple mock chunker for testing.
type mockChunker struct {
	chunks []core.Chunk
}

func (m *mockChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	result := make([]*core.Chunk, len(m.chunks))
	for i, c := range m.chunks {
		chunk := c
		result[i] = &chunk
	}
	return result, nil
}
