package distributed

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// HealthProbe checks a single node's health, returning nil when healthy and a
// non-nil error otherwise.
type HealthProbe func(ctx context.Context, node *Node) error

// NodeHealthConfig controls NodeHealth behavior.
type NodeHealthConfig struct {
	// Interval between health-check rounds. Zero uses 5s.
	Interval time.Duration

	// FailureThreshold is the number of consecutive failed probes that marks a
	// node offline. Zero uses 3.
	FailureThreshold int

	// Probe is the per-node health check. Required.
	Probe HealthProbe
}

// NodeHealth periodically probes cluster nodes and updates their status based
// on consecutive failures, letting the cluster detect node failures and react
// (via AutoRebalancer) while a node is down.
type NodeHealth struct {
	cluster  *Cluster
	cfg      NodeHealthConfig
	mu       sync.Mutex
	failures map[string]int
	cancel   context.CancelFunc
	stopped  chan struct{}
}

// NewNodeHealth creates a NodeHealth monitor for the cluster.
func NewNodeHealth(cluster *Cluster, cfg NodeHealthConfig) *NodeHealth {
	if cfg.Interval <= 0 {
		cfg.Interval = 5 * time.Second
	}
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 3
	}
	return &NodeHealth{
		cluster:  cluster,
		cfg:      cfg,
		failures: make(map[string]int),
	}
}

// Failures returns the current consecutive-failure count for a node.
func (h *NodeHealth) Failures(nodeID string) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.failures[nodeID]
}

// Check runs a single health-check round over all nodes, updating each node's
// status. It returns the number of nodes that transitioned to offline in this
// round.
func (h *NodeHealth) Check(ctx context.Context) (int, error) {
	if h.cfg.Probe == nil {
		return 0, fmt.Errorf("distributed: NodeHealth requires a Probe")
	}
	nodes := h.cluster.GetAllNodes()
	newlyOffline := 0
	for _, node := range nodes {
		probeErr := h.cfg.Probe(ctx, node)

		h.mu.Lock()
		var action string // "online", "offline", or "" (no change)
		if probeErr == nil {
			wasDown := h.failures[node.ID] >= h.cfg.FailureThreshold
			h.failures[node.ID] = 0
			if wasDown {
				action = "online"
			}
		} else {
			h.failures[node.ID]++
			if h.failures[node.ID] >= h.cfg.FailureThreshold {
				action = "offline"
			}
		}
		h.mu.Unlock()

		switch action {
		case "offline":
			if cur, ok := h.cluster.GetNode(node.ID); ok && cur.Status != "offline" {
				if h.cluster.SetNodeStatus(node.ID, "offline") == nil {
					newlyOffline++
				}
			}
		case "online":
			if cur, ok := h.cluster.GetNode(node.ID); ok && cur.Status != "online" {
				_ = h.cluster.SetNodeStatus(node.ID, "online")
			}
		}
	}
	return newlyOffline, nil
}

// Start begins the background health-check loop. It is a no-op if already
// running. The loop stops when ctx is cancelled or Stop is called.
func (h *NodeHealth) Start(ctx context.Context) {
	h.mu.Lock()
	if h.cancel != nil {
		h.mu.Unlock()
		return
	}
	cctx, cancel := context.WithCancel(ctx)
	h.cancel = cancel
	h.stopped = make(chan struct{})
	stopped := h.stopped
	h.mu.Unlock()

	go func() {
		defer close(stopped)
		ticker := time.NewTicker(h.cfg.Interval)
		defer ticker.Stop()
		_, _ = h.Check(cctx)
		for {
			select {
			case <-cctx.Done():
				return
			case <-ticker.C:
				_, _ = h.Check(cctx)
			}
		}
	}()
}

// Stop halts the background loop and waits for it to finish. It is safe to
// call when not running.
func (h *NodeHealth) Stop() {
	h.mu.Lock()
	cancel := h.cancel
	h.cancel = nil
	stopped := h.stopped
	h.mu.Unlock()
	if cancel != nil {
		cancel()
		<-stopped
	}
}

// AutoRebalancer watches cluster membership and re-runs an active rebalance
// whenever the set of active (non-offline) nodes changes, so the hash ring
// tracks node availability automatically.
type AutoRebalancer struct {
	cluster    *Cluster
	interval   time.Duration
	mu         sync.Mutex
	lastActive string // sorted, comma-joined active node IDs
	rebalances int
	cancel     context.CancelFunc
	stopped    chan struct{}
}

// NewAutoRebalancer creates an AutoRebalancer that watches the cluster's
// active node set. A non-positive interval uses 2s.
func NewAutoRebalancer(cluster *Cluster, interval time.Duration) *AutoRebalancer {
	if interval <= 0 {
		interval = 2 * time.Second
	}
	return &AutoRebalancer{cluster: cluster, interval: interval}
}

// Rebalances returns how many rebalances have been performed.
func (r *AutoRebalancer) Rebalances() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rebalances
}

// MaybeRebalance checks whether the active node set changed since the last
// check and, if so, runs an active rebalance. It returns true when a rebalance
// occurred.
func (r *AutoRebalancer) MaybeRebalance(ctx context.Context) (bool, error) {
	key := strings.Join(r.cluster.ActiveNodeIDs(), ",")
	r.mu.Lock()
	changed := key != r.lastActive
	if changed {
		r.lastActive = key
	}
	r.mu.Unlock()
	if !changed {
		return false, nil
	}
	if err := r.cluster.RebalanceActive(ctx); err != nil {
		return false, err
	}
	r.mu.Lock()
	r.rebalances++
	r.mu.Unlock()
	return true, nil
}

// Start begins the background rebalance watcher. It is a no-op if already
// running and stops when ctx is cancelled or Stop is called.
func (r *AutoRebalancer) Start(ctx context.Context) {
	r.mu.Lock()
	if r.cancel != nil {
		r.mu.Unlock()
		return
	}
	cctx, cancel := context.WithCancel(ctx)
	r.cancel = cancel
	r.stopped = make(chan struct{})
	stopped := r.stopped
	r.mu.Unlock()

	go func() {
		defer close(stopped)
		ticker := time.NewTicker(r.interval)
		defer ticker.Stop()
		for {
			select {
			case <-cctx.Done():
				return
			case <-ticker.C:
				_, _ = r.MaybeRebalance(cctx)
			}
		}
	}()
}

// Stop halts the watcher and waits for it to finish. Safe to call when not
// running.
func (r *AutoRebalancer) Stop() {
	r.mu.Lock()
	cancel := r.cancel
	r.cancel = nil
	stopped := r.stopped
	r.mu.Unlock()
	if cancel != nil {
		cancel()
		<-stopped
	}
}

// Quorum returns the majority quorum size for n replicas.
func Quorum(n int) int {
	if n <= 0 {
		return 0
	}
	return n/2 + 1
}

// QuorumMet reports whether succeeded replicas meet the quorum for n replicas.
func QuorumMet(succeeded, n int) bool {
	return succeeded >= Quorum(n)
}

// ReplicateOp applies op to each node, returning how many succeeded and the
// first error encountered. A replicated write is considered durable when at
// least Quorum(len(nodes)) nodes succeed (see QuorumMet).
func ReplicateOp(ctx context.Context, nodes []*Node, op func(ctx context.Context, node *Node) error) (succeeded int, firstErr error) {
	for _, node := range nodes {
		if err := ctx.Err(); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if err := op(ctx, node); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		succeeded++
	}
	return succeeded, firstErr
}

// ClusterHealth summarizes the health of a cluster.
type ClusterHealth struct {
	Total    int
	Online   int
	Degraded int
	Offline  int

	// Overall is "healthy", "degraded", or "down". A cluster is "down" when it
	// cannot reach a write quorum; "degraded" when operating with some nodes
	// offline; otherwise "healthy".
	Overall string
}

// Health returns a summary of cluster node health and the cluster's overall
// operability.
func (c *Cluster) Health() ClusterHealth {
	c.mu.RLock()
	defer c.mu.RUnlock()

	h := ClusterHealth{Total: len(c.nodes)}
	for _, node := range c.nodes {
		switch node.Status {
		case "online":
			h.Online++
		case "degraded":
			h.Degraded++
		case "offline":
			h.Offline++
		}
	}
	switch {
	case h.Online+h.Degraded < Quorum(h.Total):
		h.Overall = "down"
	case h.Offline > 0:
		h.Overall = "degraded"
	default:
		h.Overall = "healthy"
	}
	return h
}

// Consensus provides deterministic leader election for writes over the
// cluster's online nodes. The leader is the online node with the
// lexicographically smallest ID, which is stable across calls and changes only
// when node availability changes. A monotonically increasing term is bumped
// whenever the leader changes so writers can detect leadership transitions.
type Consensus struct {
	cluster *Cluster
	mu      sync.Mutex
	leader  string
	term    int
}

// NewConsensus creates a Consensus for the cluster.
func NewConsensus(cluster *Cluster) *Consensus {
	return &Consensus{cluster: cluster}
}

// Term returns the current leadership term.
func (c *Consensus) Term() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.term
}

// Leader returns the current leader ID ("" when there is no online node).
func (c *Consensus) Leader() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.leader
}

// IsLeader reports whether nodeID is the current leader.
func (c *Consensus) IsLeader(nodeID string) bool {
	return nodeID != "" && c.Leader() == nodeID
}

// Elect runs leader election among online nodes and updates the recorded
// leader and term. It returns the leader ID and the current term. When no node
// is online it returns ("", term).
func (c *Consensus) Elect(ctx context.Context) (string, int, error) {
	if err := ctx.Err(); err != nil {
		c.mu.Lock()
		term := c.term
		c.mu.Unlock()
		return "", term, err
	}
	var leader string
	for _, node := range c.cluster.GetOnlineNodes() {
		if leader == "" || node.ID < leader {
			leader = node.ID
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if leader != c.leader {
		c.leader = leader
		c.term++
	}
	return c.leader, c.term, nil
}
