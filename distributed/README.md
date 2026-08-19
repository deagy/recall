# distributed

A sharded, replicated store facade over multiple in-process nodes:
consistent hashing for shard assignment, scatter-gather search with
partial-failure tolerance, replication, and cluster health.

## Core

```go
cluster := distributed.NewCluster(distributed.DefaultClusterConfig())
sm := /* ShardManager: shards keyed to nodes via consistent hash ring */
store := distributed.NewDistributedStore(clusterCfg, embedder, ...) // Store-compatible
```

- `Cluster` / `Node` — node membership on a consistent-hash ring
  (MD5-based key hashing — uniformity, not security).
- `ScatterGatherSearch` / `ScatterGatherSearchHybrid` — fan out a query to
  the responsible shards and merge; tolerate partial node failure.
- `Quorum(n)` / `QuorumMet(succeeded, n)` — quorum math for reads/writes.

## Replication & operations

- `ReplicationManager` + `ReplicationStrategy` (sync/async) —
  `ReplicateOp` writes to replicas, reports per-node results.
- `AutoRebalancer` — periodic shard rebalancing when nodes join/leave.
- `NodeHealth` / `Consensus` — liveness probing and simple consensus
  helpers.
- `Diagnostics(c, sm)` / `HealthHandler` — cluster-wide health for the
  `cluster status` CLI command.
