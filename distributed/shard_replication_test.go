package distributed

import (
	"context"
	"testing"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// newTestShardCluster builds a cluster with n online nodes plus a shard
// manager ready for use.
func newTestShardCluster(n int) (*Cluster, *ShardManager) {
	cluster := NewCluster(DefaultClusterConfig())
	for i := 0; i < n; i++ {
		if err := cluster.AddNode(&Node{ID: fixedNodeID(i), Address: "localhost:9000"}); err != nil {
			panic(err)
		}
	}
	return cluster, NewShardManager(cluster)
}

func fixedNodeID(i int) string {
	return []string{"node-1", "node-2", "node-3", "node-4"}[i]
}

func chunkWithEmbed(id, content string, v ...float32) *core.Chunk {
	return &core.Chunk{ID: id, Content: content, Embedding: v}
}

// newTestDistributedStore builds a DistributedStore backed by n online
// nodes with a fixed two-chunk mock chunker (see mockChunker in
// distributed_test.go). The chunks carry DocumentRef "doc-1" so document
// deletion can be exercised (real chunkers set it from the document).
func newTestDistributedStore(t *testing.T, nodes int) *DistributedStore {
	t.Helper()
	factory := func(cfg chunker.Config) chunker.Chunker {
		return &mockChunker{chunks: []core.Chunk{
			{ID: "chunk-00000001", Content: "Test chunk 1", DocumentRef: "doc-1"},
			{ID: "chunk-00000002", Content: "Test chunk 2", DocumentRef: "doc-1"},
		}}
	}
	ds := NewDistributedStore(nil, embedder.NewMockEmbedder(3), factory, "test-ns")
	for i := 0; i < nodes; i++ {
		if err := ds.AddNode(&Node{ID: fixedNodeID(i), Address: "localhost:9000"}); err != nil {
			t.Fatal(err)
		}
	}
	return ds
}

// --- Cluster ---

func TestNewCluster_NilConfig(t *testing.T) {
	c := NewCluster(nil)
	if c.config.ReplicationFactor != 3 || c.config.ConsistentHashingVirtualNodes != 150 {
		t.Errorf("nil config must fall back to defaults, got %+v", c.config)
	}
}

// --- ShardManager lookups ---

func TestShardManager_GetShardForChunk(t *testing.T) {
	_, sm := newTestShardCluster(1)

	shardA, _ := sm.CreateShardWithID(fixedNodeID(0), "shard-aaaa")
	shardA.Status = "inactive"
	shardB, _ := sm.CreateShardWithID(fixedNodeID(0), "shard-bbbb")

	chunk := chunkWithEmbed("cccc-1", "content", 1, 0, 0)
	shardB.Data[chunk.ID] = chunk

	if got := sm.GetShardForChunk(chunk.ID); got != shardB {
		t.Errorf("expected the active shard holding the chunk, got %v", got)
	}
	if got := sm.GetShardForChunk("nope"); got != nil {
		t.Errorf("unknown chunk must return nil, got %v", got)
	}
}

func TestShardManager_GetShardForNode_AndActiveCount(t *testing.T) {
	_, sm := newTestShardCluster(2)

	sm.CreateShardWithID(fixedNodeID(0), "shard-n1a")
	sm.CreateShardWithID(fixedNodeID(0), "shard-n1b")
	sm.CreateShardWithID(fixedNodeID(1), "shard-n2a")

	if got := sm.GetShardForNode(fixedNodeID(0)); len(got) != 2 {
		t.Errorf("expected 2 shards for node-1, got %d", len(got))
	}
	if got := sm.GetShardForNode("ghost"); len(got) != 0 {
		t.Errorf("unknown node must have no shards, got %d", len(got))
	}

	if got := sm.GetActiveShardCount(); got != 3 {
		t.Errorf("expected 3 active shards, got %d", got)
	}

	shard, _ := sm.GetShard("shard-n1a")
	shard.Status = "inactive"
	if got := sm.GetActiveShardCount(); got != 2 {
		t.Errorf("expected 2 active shards after deactivating one, got %d", got)
	}
}

// --- Shard / ShardManager search ---

func TestShard_Search(t *testing.T) {
	shard := NewShard("shard-1", fixedNodeID(0))
	// Query [1,0,0] should rank the aligned chunk first.
	shard.Data["c1"] = chunkWithEmbed("c1", "aligned", 1, 0, 0)
	shard.Data["c2"] = chunkWithEmbed("c2", "perpendicular", 0, 1, 0)
	shard.Data["c3"] = &core.Chunk{ID: "c3", Content: "no embedding"}

	results, err := shard.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results (embedding-less chunk skipped), got %d", len(results))
	}
	if results[0].Chunk.ID != "c1" {
		t.Errorf("expected aligned chunk first, got %s", results[0].Chunk.ID)
	}

	empty := NewShard("shard-2", fixedNodeID(0))
	res, err := empty.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{})
	if err != nil || len(res) != 0 {
		t.Errorf("empty shard must return no results, got %v err=%v", res, err)
	}
}

func TestShardManager_SearchAndHybrid(t *testing.T) {
	_, sm := newTestShardCluster(2)
	if err := sm.StoreChunk(context.Background(), chunkWithEmbed("aaaa-1", "first chunk", 1, 0, 0)); err != nil {
		t.Fatal(err)
	}
	if err := sm.StoreChunk(context.Background(), chunkWithEmbed("bbbb-1", "second chunk", 0, 1, 0)); err != nil {
		t.Fatal(err)
	}

	res, err := sm.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{TopK: 5})
	if err != nil || len(res) != 2 {
		t.Fatalf("expected 2 results across shards, got %d err=%v", len(res), err)
	}
	if res[0].Chunk.ID != "aaaa-1" {
		t.Errorf("expected the aligned chunk first, got %s", res[0].Chunk.ID)
	}

	hybrid, err := sm.SearchHybrid(context.Background(), []float32{1, 0, 0}, index.SearchOptions{TopK: 5})
	if err != nil || len(hybrid) != 2 {
		t.Fatalf("hybrid search expected 2 results, got %d err=%v", len(hybrid), err)
	}

	// No shards at all ⇒ no iterations ⇒ nil results, nil error.
	emptySM := NewShardManager(NewCluster(DefaultClusterConfig()))
	res, err = emptySM.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{})
	if err != nil || len(res) != 0 {
		t.Errorf("shardless manager must return no results, got %v err=%v", res, err)
	}
}

// --- ShardIndex ---

func TestShardIndex_Search_WithFiltersAndLimits(t *testing.T) {
	shard := NewShard("shard-1", fixedNodeID(0))
	shard.Data["c1"] = &core.Chunk{
		ID: "c1", Content: "one", Embedding: []float32{1, 0, 0},
		Metadata: map[string]core.Value{"category": core.String{Value: "docs"}},
	}
	shard.Data["c2"] = &core.Chunk{
		ID: "c2", Content: "two", Embedding: []float32{0.9, 0.1, 0},
		Metadata: map[string]core.Value{"category": core.String{Value: "notes"}},
	}
	si := NewShardIndex(shard)

	// Term filter keeps only c1.
	res, err := si.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{
		TopK:    5,
		Filters: []index.Filter{&index.TermFilter{Key: "category", Value: "docs"}},
	})
	if err != nil || len(res) != 1 || res[0].Chunk.ID != "c1" {
		t.Fatalf("term filter expected only c1, got %v err=%v", res, err)
	}

	// MinScore above c2's similarity (~0.9939) drops it.
	res, err = si.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{TopK: 5, MinScore: 0.995})
	if err != nil || len(res) != 1 || res[0].Chunk.ID != "c1" {
		t.Fatalf("MinScore expected only c1, got %v err=%v", res, err)
	}

	// TopK=1 limits results.
	res, err = si.Search(context.Background(), []float32{1, 0, 0}, index.SearchOptions{TopK: 1})
	if err != nil || len(res) != 1 {
		t.Fatalf("TopK=1 expected 1 result, got %v err=%v", res, err)
	}
}

func TestShardIndex_SearchHybrid(t *testing.T) {
	shard := NewShard("shard-1", fixedNodeID(0))
	shard.Data["c1"] = &core.Chunk{ID: "c1", Content: "quantum fluoroscope zephyr", Embedding: []float32{0.5, 0.5, 0}}
	shard.Data["c2"] = &core.Chunk{ID: "c2", Content: "unrelated mundane text", Embedding: []float32{0.9, 0.1, 0}}
	si := NewShardIndex(shard)

	// The query shares rare tokens with c1, so BM25 should lift it above
	// c2 despite c2 having the better vector alignment.
	res, err := si.SearchHybrid(context.Background(), "quantum fluoroscope zephyr", index.SearchOptions{TopK: 5})
	if err != nil || len(res) != 2 {
		t.Fatalf("expected 2 results, got %v err=%v", res, err)
	}
	if res[0].Chunk.ID != "c1" {
		t.Errorf("keyword match expected first, got %s", res[0].Chunk.ID)
	}

	// Nil-shard guard.
	res, err = (&ShardIndex{}).SearchHybrid(context.Background(), "anything", index.SearchOptions{})
	if err != nil || len(res) != 0 {
		t.Errorf("nil-shard index must return no results, got %v err=%v", res, err)
	}
}

func TestShardIndex_AdapterMethods(t *testing.T) {
	shard := NewShard("shard-7", fixedNodeID(0))
	shard.Data["c1"] = chunkWithEmbed("c1", "x", 1, 0, 0)
	si := NewShardIndex(shard)

	ctx := context.Background()
	if err := si.Add(ctx, shard.Data["c1"]); err == nil {
		t.Error("Add must fail: ShardIndex is read-only")
	}
	if err := si.AddBatch(ctx, nil); err == nil {
		t.Error("AddBatch must fail: ShardIndex is read-only")
	}
	if err := si.Delete(ctx, "c1"); err == nil {
		t.Error("Delete must fail: ShardIndex is read-only")
	}
	if si.Count() != 1 {
		t.Errorf("Count expected 1, got %d", si.Count())
	}
	if si.Dimension() != 3 {
		t.Errorf("Dimension expected 3, got %d", si.Dimension())
	}
	if si.Namespace() != "shard-7" {
		t.Errorf("Namespace expected shard-7, got %q", si.Namespace())
	}

	// Nil-shard guards (shard field unset, not a nil receiver).
	empty := &ShardIndex{}
	if empty.Count() != 0 || empty.Dimension() != 0 || empty.Namespace() != "" {
		t.Error("nil-shard guards must return zero values")
	}

	noEmbed := NewShard("shard-8", fixedNodeID(0))
	noEmbed.Data["c1"] = &core.Chunk{ID: "c1", Content: "x"}
	if si2 := NewShardIndex(noEmbed); si2.Dimension() != 0 {
		t.Errorf("Dimension without embeddings expected 0, got %d", si2.Dimension())
	}
}

// --- Replication strategies ---

func TestNewReplicationManager_Defaults(t *testing.T) {
	cluster, sm := newTestShardCluster(1)
	rm := NewReplicationManager(cluster, sm, "", 0)
	if rm.strategy != StrategyPrimaryReplica {
		t.Errorf("empty strategy must default to primary_replica, got %q", rm.strategy)
	}
	if rm.replicationFactor != 3 {
		t.Errorf("non-positive factor must default to 3, got %d", rm.replicationFactor)
	}
}

func TestReplicationManager_Quorum(t *testing.T) {
	cluster, sm := newTestShardCluster(3)
	rm := NewReplicationManager(cluster, sm, StrategyQuorum, 3)

	data := map[string]*core.Chunk{"chunk-q1": {ID: "chunk-q1", Content: "q"}}
	results, err := rm.ReplicateData(context.Background(), "shard-q", data)
	if err != nil {
		t.Fatal(err)
	}
	// Quorum of 3 online nodes = 2.
	if len(results) != 2 {
		t.Fatalf("quorum replication expected 2 results, got %d", len(results))
	}
	for _, r := range results {
		if !r.Success {
			t.Errorf("expected success, got %+v", r)
		}
	}

	// No online nodes → error result.
	emptyCluster := NewCluster(DefaultClusterConfig())
	emptyRM := NewReplicationManager(emptyCluster, NewShardManager(emptyCluster), StrategyQuorum, 1)
	results, err = emptyRM.ReplicateData(context.Background(), "shard-q", data)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Success || results[0].Error == "" {
		t.Errorf("expected a single failure result with error, got %+v", results)
	}
}

func TestReplicationManager_AllNodes(t *testing.T) {
	cluster, sm := newTestShardCluster(3)
	rm := NewReplicationManager(cluster, sm, StrategyAllNodes, 3)

	data := map[string]*core.Chunk{"chunk-a1": {ID: "chunk-a1", Content: "a"}}
	results, err := rm.ReplicateData(context.Background(), "shard-a", data)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("all-nodes replication expected 3 results, got %d", len(results))
	}
	for _, r := range results {
		if !r.Success {
			t.Errorf("expected success, got %+v", r)
		}
	}

	emptyCluster := NewCluster(DefaultClusterConfig())
	emptyRM := NewReplicationManager(emptyCluster, NewShardManager(emptyCluster), StrategyAllNodes, 1)
	results, _ = emptyRM.ReplicateData(context.Background(), "shard-a", data)
	if len(results) != 1 || results[0].Success {
		t.Errorf("expected a single failure result, got %+v", results)
	}
}

func TestReplicationManager_PrimaryReplica_MissingShard(t *testing.T) {
	cluster, sm := newTestShardCluster(2)
	rm := NewReplicationManager(cluster, sm, StrategyPrimaryReplica, 2)

	results, err := rm.ReplicateData(context.Background(), "shard-ghost", map[string]*core.Chunk{})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Success || results[0].Error == "" {
		t.Errorf("expected failure result for missing primary shard, got %+v", results)
	}
}

func TestReplicationManager_Status(t *testing.T) {
	cluster, sm := newTestShardCluster(3)
	sm.CreateShardWithID(fixedNodeID(0), "shard-s")
	rm := NewReplicationManager(cluster, sm, StrategyPrimaryReplica, 3)

	// No replicas created yet.
	got, want, err := rm.GetReplicationStatus(context.Background(), "shard-s")
	if err != nil || got != 0 || want != 3 {
		t.Errorf("expected (0, 3, nil), got (%d, %d, %v)", got, want, err)
	}

	// Unknown shard.
	if _, _, err := rm.GetReplicationStatus(context.Background(), "shard-ghost"); err == nil {
		t.Error("expected error for unknown shard")
	}
}

// --- DistributedStore hybrid + accessors ---

func TestDistributedStore_SearchHybrid_DeleteDocument_Getters(t *testing.T) {
	ds := newTestDistributedStore(t, 2)
	if err := ds.Upload(context.Background(), &core.Document{ID: "doc-1"}, "Test content"); err != nil {
		t.Fatal(err)
	}

	results, err := ds.SearchHybrid(context.Background(), "test", index.SearchOptions{TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Error("expected hybrid results after upload")
	}

	// Note: Count() includes replica copies, so expect at least one copy per
	// chunk (2 chunks × 2 nodes = 4 with the default primary-replica strategy).
	if got := ds.Count(); got < 2 {
		t.Errorf("expected at least 2 chunk copies before deletion, got %d", got)
	}

	// DeleteDocument must remove every copy of the document's chunks across
	// all shards, including replicas.
	if err := ds.DeleteDocument(context.Background(), "doc-1"); err != nil {
		t.Fatalf("DeleteDocument must succeed for an uploaded document: %v", err)
	}
	if got := ds.Count(); got != 0 {
		t.Errorf("expected 0 chunks after DeleteDocument, got %d", got)
	}
	if _, ok := ds.GetChunk("chunk-00000001"); ok {
		t.Error("chunk must not be retrievable after DeleteDocument")
	}
	if err := ds.DeleteDocument(context.Background(), "doc-1"); err != core.ErrNotFound {
		t.Errorf("second DeleteDocument must return core.ErrNotFound, got %v", err)
	}
	if err := ds.DeleteDocument(context.Background(), "doc-never-uploaded"); err != core.ErrNotFound {
		t.Errorf("DeleteDocument of unknown document must return core.ErrNotFound, got %v", err)
	}

	if ds.GetShardManager() == nil || ds.GetReplicationManager() == nil || ds.GetCluster() == nil {
		t.Error("manager accessors must return non-nil values")
	}
}

// TestShardManager_DeleteDocument verifies selective, multi-shard document
// deletion: only chunks whose DocumentRef matches are removed, the return
// count is accurate, and unrelated documents survive.
func TestShardManager_DeleteDocument(t *testing.T) {
	_, sm := newTestShardCluster(2)
	ctx := context.Background()

	docs := map[string][]string{
		"doc-a": {"aaaa-1", "aaaa-2"},
		"doc-b": {"bbbb-1"},
		"doc-c": {"cccc-1", "cccc-2", "cccc-3"},
	}
	for docID, ids := range docs {
		for _, id := range ids {
			chunk := chunkWithEmbed(id, "content of "+id, 1, 0, 0)
			chunk.DocumentRef = docID
			if err := sm.StoreChunk(ctx, chunk); err != nil {
				t.Fatal(err)
			}
		}
	}
	if got := sm.Count(); got != 6 {
		t.Fatalf("expected 6 chunks seeded, got %d", got)
	}

	deleted, err := sm.DeleteDocument(ctx, "doc-b")
	if err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}
	if deleted != 1 {
		t.Errorf("expected 1 chunk deleted for doc-b, got %d", deleted)
	}
	if got := sm.Count(); got != 5 {
		t.Errorf("expected 5 chunks after deletion, got %d", got)
	}
	if _, ok := sm.GetChunk(ctx, "bbbb-1"); ok {
		t.Error("deleted chunk must not be retrievable")
	}

	deleted, err = sm.DeleteDocument(ctx, "doc-c")
	if err != nil || deleted != 3 {
		t.Errorf("expected 3 chunks deleted for doc-c, got %d (err=%v)", deleted, err)
	}

	// doc-a survives the unrelated deletions, then deletes cleanly, and a
	// repeat deletion reports zero.
	if deleted, err = sm.DeleteDocument(ctx, "doc-a"); err != nil || deleted != 2 {
		t.Errorf("expected 2 chunks deleted for doc-a, got %d (err=%v)", deleted, err)
	}
	if deleted, err = sm.DeleteDocument(ctx, "doc-a"); err != nil || deleted != 0 {
		t.Errorf("second deletion of doc-a must report 0, got %d (err=%v)", deleted, err)
	}
}

func TestScatterGatherSearchHybrid(t *testing.T) {
	_, sm := newTestShardCluster(2)
	if err := sm.StoreChunk(context.Background(), chunkWithEmbed("aaaa-1", "alpha chunk", 1, 0, 0)); err != nil {
		t.Fatal(err)
	}
	if err := sm.StoreChunk(context.Background(), chunkWithEmbed("bbbb-1", "beta chunk", 0, 1, 0)); err != nil {
		t.Fatal(err)
	}

	res, err := ScatterGatherSearchHybrid(context.Background(), sm, []float32{1, 0, 0},
		index.SearchOptions{TopK: 5}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
	if res[0].Score < res[1].Score {
		t.Error("results must be sorted by score descending")
	}

	// FanOut=1 queries at most one shard.
	cfg := DefaultScatterGatherConfig()
	cfg.FanOut = 1
	res, err = ScatterGatherSearchHybrid(context.Background(), sm, []float32{1, 0, 0},
		index.SearchOptions{TopK: 5}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) > 1 {
		t.Errorf("FanOut=1 expected at most 1 result, got %d", len(res))
	}

	// No active shards → error.
	emptySM := NewShardManager(NewCluster(DefaultClusterConfig()))
	if _, err := ScatterGatherSearchHybrid(context.Background(), emptySM, []float32{1, 0, 0},
		index.SearchOptions{}, nil); err == nil {
		t.Error("expected error with no active shards")
	}
}
