package distributed

import (
	"context"
	"fmt"
	"sort"
	"testing"
)

func addClusterNodes(t *testing.T, c *Cluster, ids ...string) {
	t.Helper()
	for _, id := range ids {
		if err := c.AddNode(&Node{ID: id, Address: "localhost:8080"}); err != nil {
			t.Fatalf("AddNode(%s): %v", id, err)
		}
	}
}

func TestCluster_GetNodeForChunk_DeterministicAcrossInsertionOrder(t *testing.T) {
	ids := []string{"node-1", "node-2", "node-3", "node-4"}

	mappings := make([]map[string]string, 2)
	orderings := [][]string{ids, {ids[3], ids[1], ids[0], ids[2]}}
	for k, order := range orderings {
		cfg := DefaultClusterConfig()
		cluster := NewCluster(cfg)
		addClusterNodes(t, cluster, order...)

		m := make(map[string]string)
		for i := 0; i < 200; i++ {
			key := fmt.Sprintf("chunk-%d", i)
			nodeID := cluster.GetNodeForChunk(key)
			if nodeID == "" {
				t.Fatalf("order %d: key %q resolved to no node", k, key)
			}
			m[key] = nodeID
		}
		mappings[k] = m
	}

	// The same key must land on the same node regardless of insertion order.
	for i := 0; i < 200; i++ {
		key := fmt.Sprintf("chunk-%d", i)
		if mappings[0][key] != mappings[1][key] {
			t.Fatalf("key %q: node %q (order 0) != %q (order 1)", key, mappings[0][key], mappings[1][key])
		}
	}
}

func TestCluster_RingStaysSortedAndConsistentAfterChurn(t *testing.T) {
	cfg := DefaultClusterConfig()
	cluster := NewCluster(cfg)
	addClusterNodes(t, cluster, "node-a", "node-b", "node-c")

	assert := func(label string) {
		t.Helper()
		cluster.mu.RLock()
		defer cluster.mu.RUnlock()
		if !sort.SliceIsSorted(cluster.hashRing, func(i, j int) bool { return cluster.hashRing[i] < cluster.hashRing[j] }) {
			t.Errorf("%s: hash ring is not sorted", label)
		}
		wantLen := len(cluster.nodes) * cfg.ConsistentHashingVirtualNodes
		if len(cluster.hashRing) != wantLen {
			t.Errorf("%s: ring has %d entries, want %d", label, len(cluster.hashRing), wantLen)
		}
		for _, hash := range cluster.hashRing {
			id, ok := cluster.virtualNodes[hash]
			if !ok {
				t.Errorf("%s: ring hash %d has no virtualNodes entry (stale)", label, hash)
				continue
			}
			if cluster.nodes[id] == nil {
				t.Errorf("%s: ring resolves to removed node %q", label, id)
			}
		}
		if len(cluster.virtualNodes) != wantLen {
			t.Errorf("%s: virtualNodes map has %d entries, want %d", label, len(cluster.virtualNodes), wantLen)
		}
	}

	assert("after add")

	if err := cluster.RemoveNode("node-b"); err != nil {
		t.Fatalf("RemoveNode: %v", err)
	}
	assert("after remove")

	// All keys must still resolve to a surviving node.
	for i := 0; i < 200; i++ {
		nodeID := cluster.GetNodeForChunk(fmt.Sprintf("chunk-%d", i))
		if nodeID != "node-a" && nodeID != "node-c" {
			t.Fatalf("key chunk-%d resolved to removed/unknown node %q", i, nodeID)
		}
	}
}
func TestCluster_GetReplicaNodes_FactorSatisfiedWithWrapAround(t *testing.T) {
	cfg := DefaultClusterConfig()
	cfg.ReplicationFactor = 3
	cluster := NewCluster(cfg)
	addClusterNodes(t, cluster, "node-a", "node-b", "node-c", "node-d", "node-e")

	for i := 0; i < 300; i++ {
		key := fmt.Sprintf("shard-%d", i)
		replicas := cluster.GetReplicaNodes(key)
		if len(replicas) != 3 {
			t.Fatalf("key %q: got %d replicas, want 3", key, len(replicas))
		}
		seen := make(map[string]bool)
		for _, r := range replicas {
			if r == nil {
				t.Fatalf("key %q: nil replica node", key)
			}
			if seen[r.ID] {
				t.Fatalf("key %q: duplicate replica %q", key, r.ID)
			}
			seen[r.ID] = true
		}
		// The primary must be the first clockwise node, matching GetNodeForChunk.
		if replicas[0].ID != cluster.GetNodeForChunk(key) {
			t.Fatalf("key %q: replicas[0]=%s but GetNodeForChunk=%s", key, replicas[0].ID, cluster.GetNodeForChunk(key))
		}
	}
}

func TestCluster_GetReplicaNodes_MoreNodesThanFactorAndFewer(t *testing.T) {
	cfg := DefaultClusterConfig()
	cfg.ReplicationFactor = 2
	cluster := NewCluster(cfg)
	addClusterNodes(t, cluster, "node-a", "node-b", "node-c")

	for i := 0; i < 100; i++ {
		if replicas := cluster.GetReplicaNodes(fmt.Sprintf("k-%d", i)); len(replicas) != 2 {
			t.Fatalf("3 nodes, rf=2: got %d replicas", len(replicas))
		}
	}

	if err := cluster.RemoveNode("node-c"); err != nil {
		t.Fatal(err)
	}
	if err := cluster.RemoveNode("node-b"); err != nil {
		t.Fatal(err)
	}
	// Single node: replicas capped at 1.
	if replicas := cluster.GetReplicaNodes("k-1"); len(replicas) != 1 || replicas[0].ID != "node-a" {
		t.Fatalf("1 node, rf=2: got %v", replicas)
	}

	if err := cluster.RemoveNode("node-a"); err != nil {
		t.Fatal(err)
	}
	if replicas := cluster.GetReplicaNodes("k-1"); replicas != nil {
		t.Fatalf("empty cluster: got %v replicas, want nil", replicas)
	}
}

func TestCluster_Rebalance_IdempotentAndSelfHealing(t *testing.T) {
	cfg := DefaultClusterConfig()
	cluster := NewCluster(cfg)
	addClusterNodes(t, cluster, "node-a", "node-b", "node-c")

	before := make(map[string]string)
	for i := 0; i < 100; i++ {
		before[fmt.Sprintf("k-%d", i)] = cluster.GetNodeForChunk(fmt.Sprintf("k-%d", i))
	}

	// Idempotent: an already-consistent ring maps identically after Rebalance.
	if err := cluster.Rebalance(context.Background()); err != nil {
		t.Fatalf("Rebalance: %v", err)
	}
	cluster.mu.RLock()
	wantLen := 3 * cfg.ConsistentHashingVirtualNodes
	if len(cluster.hashRing) != wantLen || len(cluster.virtualNodes) != wantLen {
		cluster.mu.RUnlock()
		t.Fatalf("after Rebalance: ring=%d map=%d, want %d", len(cluster.hashRing), len(cluster.virtualNodes), wantLen)
	}
	cluster.mu.RUnlock()
	for k, v := range before {
		if got := cluster.GetNodeForChunk(k); got != v {
			t.Fatalf("Rebalance changed mapping for %q: %q -> %q", k, v, got)
		}
	}

	// A cancelled context must be honored.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := cluster.Rebalance(ctx); err == nil {
		t.Error("Rebalance with cancelled context: expected error")
	}

	// Self-healing: corrupt the ring, then rebuild.
	cluster.mu.Lock()
	cluster.hashRing = nil
	cluster.mu.Unlock()
	if err := cluster.Rebalance(context.Background()); err != nil {
		t.Fatalf("Rebalance after corruption: %v", err)
	}
	cluster.mu.RLock()
	if len(cluster.hashRing) != wantLen {
		cluster.mu.RUnlock()
		t.Fatalf("self-heal: ring len %d, want %d", len(cluster.hashRing), wantLen)
	}
	cluster.mu.RUnlock()
	for k, v := range before {
		if got := cluster.GetNodeForChunk(k); got != v {
			t.Fatalf("self-heal changed mapping for %q: %q -> %q", k, v, got)
		}
	}
}

func TestCluster_Distribution_Sanity(t *testing.T) {
	cfg := DefaultClusterConfig()
	cluster := NewCluster(cfg)
	addClusterNodes(t, cluster, "node-a", "node-b", "node-c")

	counts := make(map[string]int)
	for i := 0; i < 1000; i++ {
		counts[cluster.GetNodeForChunk(fmt.Sprintf("key-%d", i))]++
	}
	// 3 nodes x 150 vnodes: expect ~33% each; allow a wide band, but every
	// node must receive a meaningful share (an unsorted ring would not).
	for _, id := range []string{"node-a", "node-b", "node-c"} {
		if counts[id] < 100 {
			t.Errorf("node %s got %d/1000 keys, expected ~330", id, counts[id])
		}
	}
}
