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
		ReplicationFactor:               3,
		ConsistentHashingVirtualNodes: 150,
	}
}

// Cluster manages the distributed cluster of nodes.
type Cluster struct {
	mu             sync.RWMutex
	config         *ClusterConfig
	nodes          map[string]*Node
	virtualNodes   map[uint64]string // hash -> node ID
	hashRing       []uint64
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

// GetReplicaNodes returns the nodes responsible for a given key.
func (c *Cluster) GetReplicaNodes(key string) []*Node {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.virtualNodes) == 0 {
		return nil
	}

	// Find the primary node
	hash := c.hashKey(key)
	var primaryNode *Node
	var replicas []*Node

	// Walk the hash ring to find the primary node
	for _, vHash := range c.hashRing {
		if vHash >= hash {
			primaryID := c.virtualNodes[vHash]
			primaryNode = c.nodes[primaryID]
			break
		}
	}

	if primaryNode == nil {
		// Wrap around to the first node
		primaryID := c.virtualNodes[c.hashRing[0]]
		primaryNode = c.nodes[primaryID]
	}

	replicas = append(replicas, primaryNode)

	// Find replication factor - 1 additional replicas
	for len(replicas) < c.config.ReplicationFactor {
		// Continue walking the hash ring
		found := false
		for _, vHash := range c.hashRing {
			if vHash > hash {
				nodeID := c.virtualNodes[vHash]
				node := c.nodes[nodeID]
				
				// Check if already in replicas
				duplicate := false
				for _, r := range replicas {
					if r.ID == node.ID {
						duplicate = true
						break
					}
				}

				if !duplicate {
					replicas = append(replicas, node)
					found = true
					break
				}
			}
		}

		if !found {
			break
		}
	}

	return replicas
}

// hashKey computes the hash for a key.
func (c *Cluster) hashKey(key string) uint64 {
	h := md5.Sum([]byte(key))
	return uint64(h[0])<<56 | uint64(h[1])<<48 | uint64(h[2])<<40 | uint64(h[3])<<32 |
		uint64(h[4])<<24 | uint64(h[5])<<16 | uint64(h[6])<<8 | uint64(h[7])
}

// addVirtualNodes adds virtual nodes for a real node.
func (c *Cluster) addVirtualNodes(nodeID string) {
	for i := 0; i < c.config.ConsistentHashingVirtualNodes; i++ {
		virtualKey := fmt.Sprintf("%s:%d", nodeID, i)
		hash := c.hashKey(virtualKey)
		c.hashRing = append(c.hashRing, hash)
		c.virtualNodes[hash] = nodeID
	}
}

// removeVirtualNodes removes virtual nodes for a real node.
func (c *Cluster) removeVirtualNodes(nodeID string) {
	for i := 0; i < c.config.ConsistentHashingVirtualNodes; i++ {
		virtualKey := fmt.Sprintf("%s:%d", nodeID, i)
		hash := c.hashKey(virtualKey)
		delete(c.virtualNodes, hash)
	}
}

// Rebalance triggers a rebalancing of the cluster (placeholder for future implementation).
func (c *Cluster) Rebalance(ctx context.Context) error {
	// TODO: Implement actual rebalancing logic
	return nil
}

// GetNodeForChunk returns the node responsible for storing a chunk with the given ID.
func (c *Cluster) GetNodeForChunk(chunkID string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.nodes) == 0 {
		return ""
	}

	// Use consistent hashing to find the node
	hash := c.hashKey(chunkID)

	// Find the first node in the hash ring that has a hash >= our key
	for _, vHash := range c.hashRing {
		if vHash >= hash {
			return c.virtualNodes[vHash]
		}
	}

	// Wrap around to the first node
	if len(c.hashRing) > 0 {
		return c.virtualNodes[c.hashRing[0]]
	}

	// Fallback to first node
	for _, node := range c.nodes {
		return node.ID
	}

	return ""
}
