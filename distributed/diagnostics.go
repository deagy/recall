package distributed

import (
	"encoding/json"
	"net/http"
	"time"
)

// ShardStats summarizes shard distribution across the cluster.
type ShardStats struct {
	// Total is the number of shards of any status.
	Total int `json:"total"`
	// Active is the number of active shards.
	Active int `json:"active"`
	// Inactive is the number of inactive shards.
	Inactive int `json:"inactive"`
	// Degraded is the number of degraded shards.
	Degraded int `json:"degraded"`
	// Chunks is the total number of chunks across all shards.
	Chunks int `json:"chunks"`
	// PerNode maps a node ID to its shard count.
	PerNode map[string]int `json:"per_node,omitempty"`
}

// ShardDistribution returns statistics about the shards managed by sm. A nil
// sm yields zero stats.
func ShardDistribution(sm *ShardManager) ShardStats {
	var stats ShardStats
	if sm == nil {
		return stats
	}
	stats.PerNode = make(map[string]int)
	for _, shard := range sm.GetAllShards() {
		stats.Total++
		switch shard.Status {
		case "active":
			stats.Active++
		case "degraded":
			stats.Degraded++
		default:
			stats.Inactive++
		}
		stats.PerNode[shard.NodeID]++
	}
	stats.Chunks = sm.Count()
	return stats
}

// ClusterDiagnostics combines cluster node health with shard distribution into
// a single operator-facing snapshot.
type ClusterDiagnostics struct {
	// Health summarizes node status and overall cluster operability.
	Health ClusterHealth `json:"health"`
	// Shards summarizes shard distribution.
	Shards ShardStats `json:"shards"`
	// GeneratedAt is when the snapshot was taken.
	GeneratedAt time.Time `json:"generated_at"`
}

// Diagnostics builds a ClusterDiagnostics snapshot from the cluster and its
// shard manager. A nil sm yields zero shard stats.
func Diagnostics(c *Cluster, sm *ShardManager) *ClusterDiagnostics {
	return &ClusterDiagnostics{
		Health:      c.Health(),
		Shards:      ShardDistribution(sm),
		GeneratedAt: time.Now().UTC(),
	}
}

// HealthHandler returns an http.Handler exposing cluster health and
// diagnostics:
//
//	GET /healthz      -> 200 when healthy or degraded, 503 when down (JSON body)
//	GET /diagnostics  -> 200 with a JSON cluster diagnostics snapshot
//
// Other paths return 404.
func HealthHandler(c *Cluster, sm *ShardManager) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			d := Diagnostics(c, sm)
			code := http.StatusOK
			if d.Health.Overall == "down" {
				code = http.StatusServiceUnavailable
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(code)
			_ = json.NewEncoder(w).Encode(d.Health)
		case "/diagnostics":
			d := Diagnostics(c, sm)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(d)
		default:
			http.NotFound(w, r)
		}
	})
}
