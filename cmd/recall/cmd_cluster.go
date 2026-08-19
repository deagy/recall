package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
)

// clusterNodeOut is one probed node in `recall cluster status` output.
type clusterNodeOut struct {
	Node      string `json:"node" yaml:"node"`
	Status    string `json:"status" yaml:"status"` // healthy | degraded | down | unreachable
	Nodes     string `json:"nodes" yaml:"nodes"`
	Shards    string `json:"shards" yaml:"shards"`
	Chunks    int    `json:"chunks" yaml:"chunks"`
	LatencyMS int64  `json:"latency_ms" yaml:"latency_ms"`
	Error     string `json:"error,omitempty" yaml:"error,omitempty"`
}

// clusterStatusOutput is the result of `recall cluster status`.
type clusterStatusOutput struct {
	Nodes []clusterNodeOut `json:"nodes" yaml:"nodes"`
	OK    bool             `json:"ok" yaml:"ok"`
}

func newClusterCmd(o *globalOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Distributed cluster operations",
	}
	cmd.AddCommand(newClusterStatusCmd(o))
	return cmd
}

func newClusterStatusCmd(o *globalOptions) *cobra.Command {
	var (
		nodes        []string
		probeTimeout time.Duration
	)
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Check distributed cluster health",
		Long: `Probe each cluster node's /diagnostics endpoint (served by
distributed.HealthHandler) and summarize node health and shard distribution.
Nodes come from --node (repeatable) or cli.cluster_nodes in the config.

Exit code: 0 when every node is healthy or degraded, 1 when any node is
down or unreachable.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runClusterStatus(cmd, o, nodes, probeTimeout)
		},
	}
	cmd.Flags().StringSliceVar(&nodes, "node", nil, "cluster node base URL (repeatable; default: cli.cluster_nodes from config)")
	cmd.Flags().DurationVar(&probeTimeout, "probe-timeout", 5*time.Second, "per-node probe timeout")
	return cmd
}

func runClusterStatus(cmd *cobra.Command, o *globalOptions, nodes []string, probeTimeout time.Duration) error {
	ctx := cmd.Context()

	nodeURLs := nodes
	if len(nodeURLs) == 0 {
		nodeURLs = o.cfg.CLI.ClusterNodes
	}
	if len(nodeURLs) == 0 {
		return fmt.Errorf("no cluster nodes: use --node or set cli.cluster_nodes in the config")
	}
	cleaned := make([]string, 0, len(nodeURLs))
	for _, u := range nodeURLs {
		if u = strings.TrimSpace(u); u != "" {
			cleaned = append(cleaned, u)
		}
	}
	nodeURLs = cleaned

	out := &clusterStatusOutput{Nodes: make([]clusterNodeOut, 0, len(nodeURLs)), OK: true}
	for _, u := range nodeURLs {
		node := clusterNodeOut{Node: u, Status: "unreachable"}
		start := time.Now()
		d, err := client.ProbeClusterNode(ctx, u, probeTimeout)
		node.LatencyMS = time.Since(start).Milliseconds()
		if err != nil {
			node.Error = err.Error()
			out.OK = false
		} else {
			node.Status = d.Health.Overall
			node.Nodes = fmt.Sprintf("%d online, %d degraded, %d offline (%d total)",
				d.Health.Online, d.Health.Degraded, d.Health.Offline, d.Health.Total)
			node.Shards = fmt.Sprintf("%d active, %d degraded, %d inactive (%d total)",
				d.Shards.Active, d.Shards.Degraded, d.Shards.Inactive, d.Shards.Total)
			node.Chunks = d.Shards.Chunks
			if d.Health.Overall == "down" {
				out.OK = false
			}
		}
		out.Nodes = append(out.Nodes, node)
	}

	p := newPrinter(cmd, o.output)
	err := p.emit(out, func(p *printer) {
		tw := p.table("node", "status", "nodes", "shards", "chunks", "latency", "error")
		for _, n := range out.Nodes {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%dms\t%s\n", n.Node, n.Status, n.Nodes, n.Shards, n.Chunks, n.LatencyMS, n.Error)
		}
		tw.Flush()
		if out.OK {
			fmt.Fprintln(p.w, "cluster OK")
		} else {
			fmt.Fprintln(p.w, "cluster NOT OK (down or unreachable nodes)")
		}
	})
	if err != nil {
		return err
	}
	if !out.OK {
		return &exitError{Code: 1, Message: "cluster status: down or unreachable node(s) detected"}
	}
	return nil
}
