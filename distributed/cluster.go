package distributed

import (
	"context"
	"crypto/md5"
	"fmt"
	"sort"
	"sync"
)

// Node represents a node in the distributed cluster.
type Node struct {
	ID      string `json:"id"`
	Address string `json:"address"`
	Status  string `json:"status"` // "online", "offline", "degraded"
}

// ClusterConfig holds configuration for the distributed cluster.
type ClusterConfig struct {
	// ReplicationFactor is the number of replicas for each shard.
	ReplicationFactor int `json:"replication_factor"`

	// ConsistentHashingVirtualNodes is the number of virtual nodes for consistent hashing.
	ConsistentHashingVirtualNodes int `json:"consistent_hashing_virtual_nodes"`
}

// DefaultClusterConfig returns a default cluster configuration.
func DefaultClusterConfig() *ClusterConfig {
	return &ClusterConfig{
		ReplicationFactor:             3,
		ConsistentHashingVirtualNodes: 150,
	}
}

// Cluster manages the distributed cluster of nodes.
type Cluster struct {
	mu           sync.RWMutex
	config       *ClusterConfig
	nodes        map[string]*Node
	virtualNodes map[uint64]string // hash -> node ID
	hashRing     []uint64
}

// NewCluster creates a new distributed cluster.
func NewCluster(config *ClusterConfig) *Cluster {
	if config == nil {
		config = DefaultClusterConfig()
	}
	return &Cluster{
		config:       config,
		nodes:        make(map[string]*Node),
		virtualNodes: make(map[uint64]string),
	}
}

// AddNode adds a node to the cluster.
func (c *Cluster) AddNode(node *Node) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.nodes[node.ID]; exists {
		return fmt.Errorf("node %s already exists", node.ID)
	}

	node.Status = "online"
	c.nodes[node.ID] = node

	// Add virtual nodes to the hash ring
	c.addVirtualNodes(node.ID)

	return nil
}

// RemoveNode removes a node from the cluster.
func (c *Cluster) RemoveNode(nodeID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	_, exists := c.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	// Remove virtual nodes from the hash ring
	c.removeVirtualNodes(nodeID)

	// Remove the node from the map
	delete(c.nodes, nodeID)

	return nil
}

// GetNode returns a node by its ID.
func (c *Cluster) GetNode(nodeID string) (*Node, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	node, exists := c.nodes[nodeID]
	return node, exists
}

// GetAllNodes returns all nodes in the cluster.
func (c *Cluster) GetAllNodes() []*Node {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodes := make([]*Node, 0, len(c.nodes))
	for _, node := range c.nodes {
		nodes = append(nodes, node)
	}

	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].ID < nodes[j].ID
	})

	return nodes
}

// GetOnlineNodes returns all online nodes.
func (c *Cluster) GetOnlineNodes() []*Node {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodes := make([]*Node, 0)
	for _, node := range c.nodes {
		if node.Status == "online" {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// GetNodeCount returns the number of nodes in the cluster.
func (c *Cluster) GetNodeCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.nodes)
}

// GetOnlineNodeCount returns the number of online nodes.
func (c *Cluster) GetOnlineNodeCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	count := 0
	for _, node := range c.nodes {
		if node.Status == "online" {
			count++
		}
	}

	return count
}

// GetReplicaNodes returns the nodes responsible for a given key: the primary
// (the first virtual node clockwise from the key's hash) followed by up to
// ReplicationFactor-1 further distinct nodes, walking the ring with wrap-around.
// The result contains at most len(nodes) distinct nodes and is empty only when
// the cluster has no nodes.
func (c *Cluster) GetReplicaNodes(key string) []*Node {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.hashRing) == 0 {
		return nil
	}

	rf := c.config.ReplicationFactor
	if rf < 1 {
		rf = 1
	}
	if rf > len(c.nodes) {
		rf = len(c.nodes)
	}

	hash := c.hashKey(key)
	seen := make(map[string]bool)
	replicas := make([]*Node, 0, rf)
	for i := 0; i < len(c.hashRing) && len(replicas) < rf; i++ {
		idx := c.ringIndex(hash, i)
		nodeID := c.virtualNodes[c.hashRing[idx]]
		if seen[nodeID] {
			continue
		}
		seen[nodeID] = true
		if node := c.nodes[nodeID]; node != nil {
			replicas = append(replicas, node)
		}
	}
	return replicas
}

// ringIndex returns the position of the (i+1)-th virtual node clockwise from
// hash in the sorted ring, wrapping around. Callers must hold c.mu (at least
// for read).
func (c *Cluster) ringIndex(hash uint64, i int) int {
	pos := c.searchRing(hash)
	if pos >= len(c.hashRing) {
		pos = 0
	}
	return (pos + i) % len(c.hashRing)
}

// searchRing returns the index of the first ring entry >= hash.
// Callers must hold c.mu (at least for read).
func (c *Cluster) searchRing(hash uint64) int {
	return sort.Search(len(c.hashRing), func(i int) bool { return c.hashRing[i] >= hash })
}

// hashKey computes the hash for a key.
func (c *Cluster) hashKey(key string) uint64 {
	h := md5.Sum([]byte(key))
	return uint64(h[0])<<56 | uint64(h[1])<<48 | uint64(h[2])<<40 | uint64(h[3])<<32 |
		uint64(h[4])<<24 | uint64(h[5])<<16 | uint64(h[6])<<8 | uint64(h[7])
}

// addVirtualNodes adds the virtual node hashes for nodeID into the sorted ring.
// Callers must hold c.mu.
func (c *Cluster) addVirtualNodes(nodeID string) {
	for i := 0; i < c.config.ConsistentHashingVirtualNodes; i++ {
		hash := c.hashKey(fmt.Sprintf("%s:%d", nodeID, i))
		pos := c.searchRing(hash)
		c.hashRing = append(c.hashRing, 0)
		copy(c.hashRing[pos+1:], c.hashRing[pos:])
		c.hashRing[pos] = hash
		c.virtualNodes[hash] = nodeID
	}
}

// removeVirtualNodes removes nodeID's virtual node hashes from both the ring
// slice and the hash-to-node map. Callers must hold c.mu.
func (c *Cluster) removeVirtualNodes(nodeID string) {
	for i := 0; i < c.config.ConsistentHashingVirtualNodes; i++ {
		hash := c.hashKey(fmt.Sprintf("%s:%d", nodeID, i))
		pos := c.searchRing(hash)
		if pos < len(c.hashRing) && c.hashRing[pos] == hash {
			c.hashRing = append(c.hashRing[:pos], c.hashRing[pos+1:]...)
		}
		delete(c.virtualNodes, hash)
	}
}

// Rebalance rebuilds the hash ring from the current node set. It is
// idempotent — with a consistent ring it is a no-op — but it also self-heals
// if the ring ever drifts out of sync with the node map, so it is safe to call
// after node churn.
func (c *Cluster) Rebalance(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	vn := make(map[uint64]string, len(c.nodes)*c.config.ConsistentHashingVirtualNodes)
	for id := range c.nodes {
		for i := 0; i < c.config.ConsistentHashingVirtualNodes; i++ {
			hash := c.hashKey(fmt.Sprintf("%s:%d", id, i))
			vn[hash] = id
		}
	}
	ring := make([]uint64, 0, len(vn))
	for hash := range vn {
		ring = append(ring, hash)
	}
	sort.Slice(ring, func(i, j int) bool { return ring[i] < ring[j] })

	c.virtualNodes = vn
	c.hashRing = ring
	return nil
}

// GetNodeForChunk returns the node responsible for storing a chunk with the
// given ID: the first virtual node clockwise from the chunk ID's hash.
// Returns "" when the cluster has no nodes.
func (c *Cluster) GetNodeForChunk(chunkID string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.hashRing) == 0 {
		return ""
	}

	hash := c.hashKey(chunkID)
	pos := c.searchRing(hash)
	if pos >= len(c.hashRing) {
		pos = 0
	}
	return c.virtualNodes[c.hashRing[pos]]
}
