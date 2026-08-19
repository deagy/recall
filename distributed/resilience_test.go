package distributed

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// newClusterWithNodes builds a cluster with the given node IDs, all online.
func newClusterWithNodes(t *testing.T, ids ...string) *Cluster {
	t.Helper()
	c := NewCluster(DefaultClusterConfig())
	for _, id := range ids {
		require.NoError(t, c.AddNode(&Node{ID: id, Address: id + ".local"}))
	}
	return c
}

func TestNodeHealth_MarksOfflineAfterThreshold(t *testing.T) {
	c := newClusterWithNodes(t, "n1", "n2")
	down := map[string]bool{"n1": true}
	h := NewNodeHealth(c, NodeHealthConfig{
		Interval:         time.Hour,
		FailureThreshold: 2,
		Probe:            probeFor(down),
	})
	ctx := context.Background()

	// First failure: below threshold, still online.
	n, err := h.Check(ctx)
	require.NoError(t, err)
	require.Equal(t, 0, n)
	node, _ := c.GetNode("n1")
	require.Equal(t, "online", node.Status)
	require.Equal(t, 1, h.Failures("n1"))

	// Second failure: reaches threshold, goes offline.
	n, err = h.Check(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, n)
	node, _ = c.GetNode("n1")
	require.Equal(t, "offline", node.Status)

	// n2 stays online.
	node2, _ := c.GetNode("n2")
	require.Equal(t, "online", node2.Status)
}

func TestNodeHealth_Recovery(t *testing.T) {
	c := newClusterWithNodes(t, "n1")
	down := true
	h := NewNodeHealth(c, NodeHealthConfig{
		Interval:         time.Hour,
		FailureThreshold: 1,
		Probe: func(ctx context.Context, node *Node) error {
			if down && node.ID == "n1" {
				return fmt.Errorf("n1 down")
			}
			return nil
		},
	})
	ctx := context.Background()

	_, _ = h.Check(ctx)
	node, _ := c.GetNode("n1")
	require.Equal(t, "offline", node.Status)

	// Recover the node.
	down = false
	_, _ = h.Check(ctx)
	node, _ = c.GetNode("n1")
	require.Equal(t, "online", node.Status)
	require.Equal(t, 0, h.Failures("n1"))
}

func TestNodeHealth_StartStop(t *testing.T) {
	c := newClusterWithNodes(t, "n1")
	h := NewNodeHealth(c, NodeHealthConfig{
		Interval: time.Millisecond,
		Probe:    probeFor(map[string]bool{}),
	})
	h.Start(context.Background())
	time.Sleep(5 * time.Millisecond) // let a tick or two run
	node, ok := c.GetNode("n1")
	require.True(t, ok)
	require.Equal(t, "online", node.Status)
	h.Stop() // must not hang
}

func TestNodeHealth_RequiresProbe(t *testing.T) {
	c := newClusterWithNodes(t, "n1")
	h := NewNodeHealth(c, NodeHealthConfig{Interval: time.Hour})
	_, err := h.Check(context.Background())
	require.Error(t, err)
}

func TestAutoRebalancer_RebalancesOnChange(t *testing.T) {
	c := newClusterWithNodes(t, "n1")
	r := NewAutoRebalancer(c, time.Hour)
	ctx := context.Background()

	changed, err := r.MaybeRebalance(ctx)
	require.NoError(t, err)
	require.True(t, changed, "first check should rebalance (empty -> {n1})")
	require.Equal(t, 1, r.Rebalances())

	changed, err = r.MaybeRebalance(ctx)
	require.NoError(t, err)
	require.False(t, changed, "no membership change should not rebalance")
	require.Equal(t, 1, r.Rebalances())

	require.NoError(t, c.AddNode(&Node{ID: "n2"}))
	changed, err = r.MaybeRebalance(ctx)
	require.NoError(t, err)
	require.True(t, changed, "adding a node should rebalance")
	require.Equal(t, 2, r.Rebalances())
}

func TestAutoRebalancer_OfflineNodeExcludedFromRing(t *testing.T) {
	c := newClusterWithNodes(t, "n1", "n2", "n3")
	r := NewAutoRebalancer(c, time.Hour)
	ctx := context.Background()

	vn := c.config.ConsistentHashingVirtualNodes
	_, _ = r.MaybeRebalance(ctx)
	require.Equal(t, 3*vn, ringLen(t, c))

	// Mark n2 offline and rebalance; the ring should drop to 2 nodes.
	require.NoError(t, c.SetNodeStatus("n2", "offline"))
	_, _ = r.MaybeRebalance(ctx)
	require.Equal(t, 2*vn, ringLen(t, c))

	// No chunk should route to the offline node n2.
	for i := 0; i < 200; i++ {
		require.NotEqual(t, "n2", c.GetNodeForChunk(fmt.Sprintf("chunk-%d", i)))
	}
}

func TestAutoRebalancer_StartStop(t *testing.T) {
	c := newClusterWithNodes(t, "n1")
	r := NewAutoRebalancer(c, time.Millisecond)
	r.Start(context.Background())
	time.Sleep(5 * time.Millisecond)
	r.Stop() // must not hang
	require.GreaterOrEqual(t, r.Rebalances(), 1)
}

// probeFor returns a HealthProbe that fails for the node IDs in the set.
func probeFor(failing map[string]bool) HealthProbe {
	return func(ctx context.Context, node *Node) error {
		if failing[node.ID] {
			return fmt.Errorf("node %s unhealthy", node.ID)
		}
		return nil
	}
}

// ringLen returns the number of virtual nodes in the hash ring.
func ringLen(t *testing.T, c *Cluster) int {
	t.Helper()
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.hashRing)
}

func TestQuorum(t *testing.T) {
	require.Equal(t, 0, Quorum(0))
	require.Equal(t, 1, Quorum(1))
	require.Equal(t, 2, Quorum(2))
	require.Equal(t, 2, Quorum(3))
	require.Equal(t, 3, Quorum(4))
	require.Equal(t, 3, Quorum(5))
}

func TestQuorumMet(t *testing.T) {
	require.True(t, QuorumMet(2, 3))
	require.False(t, QuorumMet(1, 3))
	require.True(t, QuorumMet(1, 1))
	require.False(t, QuorumMet(0, 1))
}

func TestReplicateOp(t *testing.T) {
	nodes := []*Node{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	failed := map[string]bool{"b": true}
	var seen []string
	succeeded, firstErr := ReplicateOp(context.Background(), nodes, func(ctx context.Context, node *Node) error {
		seen = append(seen, node.ID)
		if failed[node.ID] {
			return fmt.Errorf("node %s failed", node.ID)
		}
		return nil
	})
	require.Equal(t, 2, succeeded)
	require.NotNil(t, firstErr)
	require.Contains(t, firstErr.Error(), "b")
	require.Len(t, seen, 3)
	// 2 of 3 succeed meets quorum.
	require.True(t, QuorumMet(succeeded, len(nodes)))
}

func TestClusterHealth(t *testing.T) {
	ctx := context.Background()

	// All online -> healthy.
	c := newClusterWithNodes(t, "n1", "n2", "n3")
	h := c.Health()
	require.Equal(t, "healthy", h.Overall)
	require.Equal(t, 3, h.Online)

	// One offline -> degraded (still has quorum).
	require.NoError(t, c.SetNodeStatus("n1", "offline"))
	h = c.Health()
	require.Equal(t, "degraded", h.Overall)
	require.Equal(t, 1, h.Offline)

	// Two of three offline -> below quorum -> down.
	require.NoError(t, c.SetNodeStatus("n2", "offline"))
	h = c.Health()
	require.Equal(t, "down", h.Overall)
	_ = ctx
}

func TestConsensus_ElectsSmallestOnlineID(t *testing.T) {
	c := newClusterWithNodes(t, "c3", "c1", "c2")
	cs := NewConsensus(c)
	leader, term, err := cs.Elect(context.Background())
	require.NoError(t, err)
	require.Equal(t, "c1", leader)
	require.Equal(t, 1, term)
	require.True(t, cs.IsLeader("c1"))
	require.False(t, cs.IsLeader("c2"))
	require.Equal(t, "c1", cs.Leader())
}

func TestConsensus_TermBumpsOnLeaderChange(t *testing.T) {
	c := newClusterWithNodes(t, "c1", "c2")
	cs := NewConsensus(c)
	ctx := context.Background()

	leader, term, _ := cs.Elect(ctx)
	require.Equal(t, "c1", leader)
	require.Equal(t, 1, term)

	// c1 fails; c2 becomes leader and the term bumps.
	require.NoError(t, c.SetNodeStatus("c1", "offline"))
	leader, term, _ = cs.Elect(ctx)
	require.Equal(t, "c2", leader)
	require.Equal(t, 2, term)

	// No change -> term stays.
	leader, term, _ = cs.Elect(ctx)
	require.Equal(t, "c2", leader)
	require.Equal(t, 2, term)
}

func TestConsensus_NoLeaderWhenNoneOnline(t *testing.T) {
	c := newClusterWithNodes(t, "c1")
	cs := NewConsensus(c)
	require.NoError(t, c.SetNodeStatus("c1", "offline"))
	leader, _, err := cs.Elect(context.Background())
	require.NoError(t, err)
	require.Equal(t, "", leader)
	require.False(t, cs.IsLeader("c1"))
}
