// Package distributed provides a sharded, replicated store facade over
// multiple in-process nodes: consistent hashing for shard assignment,
// scatter-gather search with partial-failure tolerance, and cluster
// diagnostics.
package distributed

import (
	"context"
	"fmt"
	"sync"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// DistributedStore implements the store.Store interface for distributed storage.
type DistributedStore struct {
	mu             sync.RWMutex
	config         *ClusterConfig
	cluster        *Cluster
	shardManager   *ShardManager
	replicationMgr *ReplicationManager
	embedder       embedder.Embedder
	chunkerFactory chunker.Factory
	namespace      string
}

// NewDistributedStore creates a new distributed store.
func NewDistributedStore(config *ClusterConfig, embedder embedder.Embedder, chunkerFactory chunker.Factory, namespace string) *DistributedStore {
	if config == nil {
		config = DefaultClusterConfig()
	}

	cluster := NewCluster(config)
	shardManager := NewShardManager(cluster)
	replicationMgr := NewReplicationManager(cluster, shardManager, StrategyPrimaryReplica, config.ReplicationFactor)

	return &DistributedStore{
		config:         config,
		cluster:        cluster,
		shardManager:   shardManager,
		replicationMgr: replicationMgr,
		embedder:       embedder,
		chunkerFactory: chunkerFactory,
		namespace:      namespace,
	}
}

// AddNode adds a node to the distributed cluster.
func (ds *DistributedStore) AddNode(node *Node) error {
	return ds.cluster.AddNode(node)
}

// RemoveNode removes a node from the distributed cluster.
func (ds *DistributedStore) RemoveNode(nodeID string) error {
	return ds.cluster.RemoveNode(nodeID)
}

// Upload processes a document: chunks it, embeds the chunks, and indexes them.
func (ds *DistributedStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Create chunker and chunk the document
	chunkerCfg := chunker.Config{
		MaxTokens:     512,
		MinChunkSize:  50,
		OverlapTokens: 50,
		Separator:     "\n\n",
	}
	chunker := ds.chunkerFactory(chunkerCfg)
	chunks, err := chunker.Chunk(doc, content)
	if err != nil {
		return fmt.Errorf("failed to chunk document: %w", err)
	}

	// Embed and store chunks
	for _, chunk := range chunks {
		// Generate embedding
		embedding, err := ds.embedder.Embed(ctx, chunk.Content)
		if err != nil {
			return fmt.Errorf("failed to embed chunk: %w", err)
		}

		// Store in shard
		chunk.Embedding = embedding
		if err := ds.shardManager.StoreChunk(ctx, chunk); err != nil {
			return fmt.Errorf("failed to store chunk: %w", err)
		}

		// Replicate data
		data := map[string]*core.Chunk{
			chunk.ID: chunk,
		}
		shardID := fmt.Sprintf("shard-%s", chunk.ID[:8])
		_, err = ds.replicationMgr.ReplicateData(ctx, shardID, data)
		if err != nil {
			return fmt.Errorf("failed to replicate data: %w", err)
		}
	}

	return nil
}

// Search finds the most relevant chunks for a query string (vector similarity only).
func (ds *DistributedStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}

	// Generate query embedding from the query string
	queryEmbedding := generateQueryEmbedding(query)

	scgConfig := &ScatterGatherConfig{
		FanOut:             0,
		MaxResultsPerShard: opts.TopK * 2,
		TotalResults:       opts.TopK,
		Timeout:            5000,
	}

	return ScatterGatherSearch(ctx, ds.shardManager, queryEmbedding, opts, scgConfig)
}

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
func (ds *DistributedStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}

	// Generate query embedding from the query string
	queryEmbedding := generateQueryEmbedding(query)

	scgConfig := &ScatterGatherConfig{
		FanOut:             0,
		MaxResultsPerShard: opts.TopK * 2,
		TotalResults:       opts.TopK,
		Timeout:            5000,
	}

	return ScatterGatherSearchHybrid(ctx, ds.shardManager, queryEmbedding, opts, scgConfig)
}

// GetChunk returns a chunk by its ID.
func (ds *DistributedStore) GetChunk(id string) (*core.Chunk, bool) {
	return ds.shardManager.GetChunk(context.Background(), id)
}

// DeleteChunk removes a chunk from the store.
func (ds *DistributedStore) DeleteChunk(ctx context.Context, id string) error {
	return ds.shardManager.DeleteChunk(ctx, id)
}

// DeleteDocument removes all chunks belonging to a document across every
// active shard. It returns core.ErrNotFound if the document has no chunks
// in any shard.
func (ds *DistributedStore) DeleteDocument(ctx context.Context, docID string) error {
	deleted, err := ds.shardManager.DeleteDocument(ctx, docID)
	if err != nil {
		return err
	}
	if deleted == 0 {
		return core.ErrNotFound
	}
	return nil
}

// Count returns the total number of chunks across all namespaces.
func (ds *DistributedStore) Count() int {
	return ds.shardManager.Count()
}

// Namespaces returns the list of namespaces in the store.
func (ds *DistributedStore) Namespaces() []string {
	return []string{ds.namespace}
}

// Close cleans up any resources held by the store.
func (ds *DistributedStore) Close() error {
	return nil
}

// GetCluster returns the underlying cluster.
func (ds *DistributedStore) GetCluster() *Cluster {
	return ds.cluster
}

// GetShardManager returns the underlying shard manager.
func (ds *DistributedStore) GetShardManager() *ShardManager {
	return ds.shardManager
}

// GetReplicationManager returns the underlying replication manager.
func (ds *DistributedStore) GetReplicationManager() *ReplicationManager {
	return ds.replicationMgr
}
