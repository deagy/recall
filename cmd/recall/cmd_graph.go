package main

import (
	"fmt"
	"sort"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/store"
)

// graphEntityOut is an entity in graph command output.
type graphEntityOut struct {
	ID      string            `json:"id" yaml:"id"`
	Label   string            `json:"label" yaml:"label"`
	Type    string            `json:"type" yaml:"type"`
	Props   map[string]string `json:"properties,omitempty" yaml:"properties,omitempty"`
	Sources []string          `json:"source_chunks,omitempty" yaml:"source_chunks,omitempty"`
}

// graphRelationOut is a relation in graph command output.
type graphRelationOut struct {
	From   string  `json:"from" yaml:"from"`
	To     string  `json:"to" yaml:"to"`
	Type   string  `json:"type" yaml:"type"`
	Weight float64 `json:"weight" yaml:"weight"`
}

// graphEntityOutput is the result of `recall graph <entity>`.
type graphEntityOutput struct {
	Mode      string             `json:"mode" yaml:"mode"`
	Entity    graphEntityOut     `json:"entity" yaml:"entity"`
	Neighbors []graphEntityOut   `json:"neighbors" yaml:"neighbors"`
	Relations []graphRelationOut `json:"relations" yaml:"relations"`
}

// graphListOutput is the result of `recall graph list`.
type graphListOutput struct {
	Mode       string           `json:"mode" yaml:"mode"`
	Entities   int              `json:"entities" yaml:"entities"`
	Relations  int              `json:"relations" yaml:"relations"`
	EntityList []graphEntityOut `json:"entity_list" yaml:"entity_list"`
}

func newGraphCmd(o *globalOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "graph <entity>",
		Short: "Query the knowledge graph",
		Long: `Query the knowledge graph.

  recall graph <entity>   show an entity (by ID or unique label) with its
                           neighbors and relations
  recall graph list       list all entities (local mode only)`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGraphEntity(cmd, o, args[0])
		},
	}
	cmd.AddCommand(newGraphListCmd(o))
	return cmd
}

func runGraphEntity(cmd *cobra.Command, o *globalOptions, id string) error {
	ctx := cmd.Context()

	if o.cli != nil {
		detail, err := o.cli.GraphEntity(ctx, id)
		if err != nil {
			return err
		}
		out := &graphEntityOutput{
			Mode:      "server",
			Entity:    toGraphEntityOutFromClient(&detail.Entity),
			Neighbors: make([]graphEntityOut, 0, len(detail.Neighbors)),
			Relations: make([]graphRelationOut, 0, len(detail.Relations)),
		}
		for _, n := range detail.Neighbors {
			out.Neighbors = append(out.Neighbors, toGraphEntityOutFromClient(&n))
		}
		for _, r := range detail.Relations {
			out.Relations = append(out.Relations, graphRelationOut{From: r.From, To: r.To, Type: r.Type, Weight: r.Weight})
		}
		return emitGraphEntity(cmd, o, out)
	}

	gs, err := o.openLocalGraph()
	if err != nil {
		return err
	}
	defer gs.Close()

	g := gs.Graph()
	entity, ok := g.GetEntity(id)
	if !ok {
		if matches := g.FindEntitiesByLabel(id); len(matches) == 1 {
			entity = matches[0]
		} else {
			return fmt.Errorf("entity not found: %s", id)
		}
	}

	out := &graphEntityOutput{
		Mode:      "local",
		Entity:    toGraphEntityOut(entity),
		Neighbors: make([]graphEntityOut, 0),
		Relations: make([]graphRelationOut, 0),
	}
	for _, n := range g.Neighbors(entity.ID) {
		if n != nil {
			out.Neighbors = append(out.Neighbors, toGraphEntityOut(n))
		}
	}
	for _, r := range g.Relations() {
		if r.From == entity.ID || r.To == entity.ID {
			out.Relations = append(out.Relations, graphRelationOut{From: r.From, To: r.To, Type: r.Type, Weight: r.Weight})
		}
	}
	return emitGraphEntity(cmd, o, out)
}

func newGraphListCmd(o *globalOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all graph entities (local mode only)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			gs, err := o.openLocalGraph()
			if err != nil {
				return err
			}
			defer gs.Close()

			ents := gs.Graph().Entities()
			sort.Slice(ents, func(i, j int) bool { return ents[i].ID < ents[j].ID })
			out := &graphListOutput{
				Mode:       "local",
				Entities:   len(ents),
				Relations:  gs.Graph().RelationCount(),
				EntityList: make([]graphEntityOut, 0, len(ents)),
			}
			for _, e := range ents {
				out.EntityList = append(out.EntityList, toGraphEntityOut(e))
			}
			p := newPrinter(cmd, o.output)
			return p.emit(out, func(p *printer) {
				if len(ents) == 0 {
					fmt.Fprintln(p.w, "no entities in the graph")
					return
				}
				tw := p.table("id", "label", "type")
				for _, e := range ents {
					fmt.Fprintf(tw, "%s\t%s\t%s\n", e.ID, e.Label, e.Type)
				}
				tw.Flush()
				fmt.Fprintf(p.w, "%d entit(y/ies), %d relation(s)\n", len(ents), out.Relations)
			})
		},
	}
}

func emitGraphEntity(cmd *cobra.Command, o *globalOptions, out *graphEntityOutput) error {
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		e := out.Entity
		fmt.Fprintf(p.w, "entity: %s (%s, type=%s)\n", e.Label, e.ID, e.Type)
		if len(e.Props) > 0 {
			keys := make([]string, 0, len(e.Props))
			for k := range e.Props {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				fmt.Fprintf(p.w, "  %s = %s\n", k, e.Props[k])
			}
		}
		fmt.Fprintf(p.w, "neighbors (%d):\n", len(out.Neighbors))
		for _, n := range out.Neighbors {
			fmt.Fprintf(p.w, "  - %s (%s, type=%s)\n", n.Label, n.ID, n.Type)
		}
		fmt.Fprintf(p.w, "relations (%d):\n", len(out.Relations))
		for _, r := range out.Relations {
			fmt.Fprintf(p.w, "  %s -> %s [%s] (weight %.2f)\n", r.From, r.To, r.Type, r.Weight)
		}
	})
}

func toGraphEntityOut(e *graph.Entity) graphEntityOut {
	if e == nil {
		return graphEntityOut{}
	}
	return graphEntityOut{ID: e.ID, Label: e.Label, Type: string(e.Type), Props: e.Properties, Sources: e.SourceChunks}
}

func toGraphEntityOutFromClient(e *client.Entity) graphEntityOut {
	return graphEntityOut{ID: e.ID, Label: e.Label, Type: e.Type, Props: e.Properties, Sources: e.SourceChunks}
}

// openLocalGraph opens the SQLite-backed graph store at the configured store
// path (the graph shares the store's database file).
func (o *globalOptions) openLocalGraph() (*store.SQLiteGraphStore, error) {
	if err := o.requireLocal("graph"); err != nil {
		return nil, err
	}
	path, err := o.requireSQLiteLocal("")
	if err != nil {
		return nil, err
	}
	gs, err := store.NewSQLiteGraphStore(path)
	if err != nil {
		return nil, err
	}
	if err := gs.LoadFromDB(); err != nil {
		_ = gs.Close()
		return nil, fmt.Errorf("loading graph: %w", err)
	}
	return gs, nil
}
