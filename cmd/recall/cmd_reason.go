package main

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
	"github.com/deagy/recall/reasoning"
)

// reasonInference is an inferred relation in `recall reason` output.
type reasonInference struct {
	From       string  `json:"from" yaml:"from"`
	To         string  `json:"to" yaml:"to"`
	Type       string  `json:"type" yaml:"type"`
	Confidence float64 `json:"confidence" yaml:"confidence"`
	Rule       string  `json:"rule" yaml:"rule"`
	Hops       int     `json:"hops" yaml:"hops"`
}

// reasonPath is a discovered path in `recall reason` output.
type reasonPath struct {
	Entities  []string `json:"entities" yaml:"entities"`
	Relations []string `json:"relations" yaml:"relations"`
}

// reasonOutput is the result of `recall reason`.
type reasonOutput struct {
	Mode       string            `json:"mode" yaml:"mode"`
	Query      string            `json:"query,omitempty" yaml:"query,omitempty"`
	From       string            `json:"from,omitempty" yaml:"from,omitempty"`
	To         string            `json:"to,omitempty" yaml:"to,omitempty"`
	Inferences []reasonInference `json:"inferences" yaml:"inferences"`
	Paths      []reasonPath      `json:"paths" yaml:"paths"`
}

func newReasonCmd(o *globalOptions) *cobra.Command {
	var (
		from    string
		to      string
		maxHops int
	)
	cmd := &cobra.Command{
		Use:   "reason [query]",
		Short: "Run multi-hop reasoning over the knowledge graph",
		Long: `Run multi-hop reasoning over the knowledge graph, either as a natural
language query (inferences are derived from the graph via inference rules)
or as a path exploration between two entities:

  recall reason "Is Alice connected to Berlin?"
  recall reason --from alice --to berlin --max-hops 4`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := ""
			if len(args) == 1 {
				query = args[0]
			}
			if query == "" && (from == "" || to == "") {
				return fmt.Errorf("provide a query argument or both --from and --to")
			}
			return runReason(cmd, o, reasonArgs{query: query, from: from, to: to, maxHops: maxHops})
		},
	}
	cmd.Flags().StringVar(&from, "from", "", "start entity ID (path exploration, requires --to)")
	cmd.Flags().StringVar(&to, "to", "", "end entity ID (path exploration, requires --from)")
	cmd.Flags().IntVar(&maxHops, "max-hops", 3, "maximum number of hops")
	return cmd
}

type reasonArgs struct {
	query   string
	from    string
	to      string
	maxHops int
}

func runReason(cmd *cobra.Command, o *globalOptions, a reasonArgs) error {
	ctx := cmd.Context()
	out := &reasonOutput{Mode: "local", Query: a.query, From: a.from, To: a.to, Inferences: []reasonInference{}, Paths: []reasonPath{}}

	if o.cli != nil {
		resp, err := o.cli.Reason(ctx, client.ReasonRequest{Query: a.query, From: a.from, To: a.to, MaxHops: a.maxHops})
		if err != nil {
			return err
		}
		out.Mode = "server"
		for _, ir := range resp.Inferences {
			out.Inferences = append(out.Inferences, reasonInference{From: ir.From, To: ir.To, Type: ir.Type, Confidence: ir.Confidence, Rule: ir.Rule, Hops: ir.Hops})
		}
		for _, p := range resp.Paths {
			out.Paths = append(out.Paths, reasonPath{Entities: p.Entities, Relations: p.Relations})
		}
		return emitReason(cmd, o, out)
	}

	gs, err := o.openLocalGraph()
	if err != nil {
		return err
	}
	defer gs.Close()

	engine := reasoning.NewEngine(gs.Graph(), reasoning.DefaultConfig())
	if a.query != "" {
		for _, ir := range engine.Reason(a.query, a.maxHops) {
			if ir == nil {
				continue
			}
			out.Inferences = append(out.Inferences, reasonInference{
				From: ir.From, To: ir.To, Type: ir.Type,
				Confidence: ir.Confidence, Rule: ir.Rule, Hops: len(ir.Path),
			})
		}
	} else {
		for _, p := range engine.ExplorePaths(a.from, a.to) {
			if p == nil {
				continue
			}
			rp := reasonPath{Entities: make([]string, 0, len(p.Entities)), Relations: make([]string, 0, len(p.Relations))}
			for _, e := range p.Entities {
				if e != nil {
					rp.Entities = append(rp.Entities, e.ID)
				}
			}
			for _, r := range p.Relations {
				if r != nil {
					rp.Relations = append(rp.Relations, fmt.Sprintf("%s->%s:%s", r.From, r.To, r.Type))
				}
			}
			out.Paths = append(out.Paths, rp)
		}
	}
	return emitReason(cmd, o, out)
}

func emitReason(cmd *cobra.Command, o *globalOptions, out *reasonOutput) error {
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		if len(out.Inferences) > 0 {
			tw := p.table("from", "to", "type", "confidence", "rule", "hops")
			for _, ir := range out.Inferences {
				fmt.Fprintf(tw, "%s\t%s\t%s\t%.3f\t%s\t%d\n", ir.From, ir.To, ir.Type, ir.Confidence, ir.Rule, ir.Hops)
			}
			tw.Flush()
			fmt.Fprintf(p.w, "%d inference(s)\n", len(out.Inferences))
		}
		if len(out.Paths) > 0 {
			fmt.Fprintf(p.w, "paths (%d):\n", len(out.Paths))
			for _, path := range out.Paths {
				fmt.Fprintf(p.w, "  %s\n", renderPath(path))
			}
		}
		if len(out.Inferences) == 0 && len(out.Paths) == 0 {
			fmt.Fprintln(p.w, "no inferences or paths found")
		}
	})
}

// renderPath renders a path as "a ->[type] b ->[type] c".
func renderPath(p reasonPath) string {
	out := ""
	for i, e := range p.Entities {
		if i > 0 && i-1 < len(p.Relations) {
			out += " --[" + p.Relations[i-1] + "]--> "
		}
		out += e
	}
	return out
}
