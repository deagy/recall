package main

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
	"github.com/deagy/recall/index"
)

// searchHit is a single ranked result.
type searchHit struct {
	Rank     int            `json:"rank" yaml:"rank"`
	ID       string         `json:"id" yaml:"id"`
	Document string         `json:"document" yaml:"document"`
	Chunk    int            `json:"chunk" yaml:"chunk"`
	Score    float64        `json:"score" yaml:"score"`
	Content  string         `json:"content" yaml:"content"`
	Metadata map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// searchOutput is the result of `recall search` / `recall hybrid-search`.
type searchOutput struct {
	Mode    string      `json:"mode" yaml:"mode"`
	Hybrid  bool        `json:"hybrid" yaml:"hybrid"`
	Query   string      `json:"query" yaml:"query"`
	Count   int         `json:"count" yaml:"count"`
	Results []searchHit `json:"results" yaml:"results"`
}

type searchFlags struct {
	topK       int
	minScore   float64
	efSearch   int
	bm25Weight float64 // hybrid only
}

func newSearchCmd(o *globalOptions) *cobra.Command {
	f := &searchFlags{}
	cmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Perform vector similarity search",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runSearch(cmd, o, args[0], false, f)
		},
	}
	addSearchFlags(cmd, f, false)
	return cmd
}

func newHybridSearchCmd(o *globalOptions) *cobra.Command {
	f := &searchFlags{bm25Weight: 0.5}
	cmd := &cobra.Command{
		Use:   "hybrid-search <query>",
		Short: "Perform hybrid search (vector + BM25 keyword fusion)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runSearch(cmd, o, args[0], true, f)
		},
	}
	addSearchFlags(cmd, f, true)
	return cmd
}

func addSearchFlags(cmd *cobra.Command, f *searchFlags, hybrid bool) {
	cmd.Flags().IntVar(&f.topK, "top-k", 10, "maximum number of results")
	cmd.Flags().Float64Var(&f.minScore, "min-score", 0, "minimum relevance score")
	cmd.Flags().IntVar(&f.efSearch, "ef-search", 0, "HNSW search width (0 = default)")
	if hybrid {
		cmd.Flags().Float64Var(&f.bm25Weight, "bm25-weight", 0.5, "BM25 keyword weight in [0,1] (0 = pure vector, 1 = pure keyword)")
	}
}

func runSearch(cmd *cobra.Command, o *globalOptions, query string, hybrid bool, f *searchFlags) error {
	ctx := cmd.Context()

	if o.cli != nil {
		opts := clientSearchOptions(f)
		var res *client.SearchResults
		var err error
		if hybrid {
			res, err = o.cli.HybridSearch(ctx, query, opts)
		} else {
			res, err = o.cli.Search(ctx, query, opts)
		}
		if err != nil {
			return err
		}
		return emitSearch(cmd, o, searchFromAPI(query, hybrid, res))
	}

	st, err := o.openLocalStore()
	if err != nil {
		return err
	}
	defer st.Close()

	opts := index.DefaultSearchOptions(f.topK)
	opts.MinScore = f.minScore
	opts.EfSearch = f.efSearch
	opts.Filters = namespaceFilter(namespaceFilterValue(cmd, o))
	var results []index.SearchResult
	if hybrid {
		opts.Hybrid = true
		opts.BM25Weight = f.bm25Weight
		results, err = st.SearchHybrid(ctx, query, opts)
	} else {
		results, err = st.Search(ctx, query, opts)
	}
	if err != nil {
		return err
	}
	return emitSearch(cmd, o, searchFromStore(query, hybrid, results))
}

// namespaceFilterValue returns the namespace to filter by in local mode:
// the --namespace flag only when explicitly set, so an unset flag searches
// all namespaces in the store.
func namespaceFilterValue(cmd *cobra.Command, o *globalOptions) string {
	if cmd.Flags().Changed("namespace") {
		return o.namespace
	}
	return ""
}

func clientSearchOptions(f *searchFlags) client.SearchOptions {
	return client.SearchOptions{
		TopK:       f.topK,
		MinScore:   f.minScore,
		BM25Weight: f.bm25Weight,
		EfSearch:   f.efSearch,
	}
}

func searchFromAPI(query string, hybrid bool, res *client.SearchResults) *searchOutput {
	out := &searchOutput{Mode: "server", Hybrid: hybrid, Query: query, Count: res.Count, Results: make([]searchHit, 0, len(res.Results))}
	for i, r := range res.Results {
		out.Results = append(out.Results, searchHit{
			Rank: i + 1, ID: r.ID, Document: r.Document, Chunk: r.ChunkIndex,
			Score: r.Score, Content: r.Content, Metadata: r.Metadata,
		})
	}
	return out
}

func searchFromStore(query string, hybrid bool, results []index.SearchResult) *searchOutput {
	out := &searchOutput{Mode: "local", Hybrid: hybrid, Query: query, Count: len(results), Results: make([]searchHit, 0, len(results))}
	for i, r := range results {
		if r.Chunk == nil {
			continue
		}
		out.Results = append(out.Results, searchHit{
			Rank: i + 1, ID: r.Chunk.ID, Document: r.Chunk.DocumentRef, Chunk: r.Chunk.ChunkIndex,
			Score: r.Score, Content: r.Chunk.Content, Metadata: metadataToAny(r.Chunk.Metadata),
		})
	}
	return out
}

func emitSearch(cmd *cobra.Command, o *globalOptions, out *searchOutput) error {
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		if out.Count == 0 {
			fmt.Fprintln(p.w, "no results")
			return
		}
		tw := p.table("rank", "score", "id", "document", "chunk", "snippet")
		for _, h := range out.Results {
			fmt.Fprintf(tw, "%d\t%.4f\t%s\t%s\t%d\t%s\n", h.Rank, h.Score, h.ID, h.Document, h.Chunk, snippet(h.Content, 100))
		}
		tw.Flush()
		fmt.Fprintf(p.w, "%d result(s) for %q\n", out.Count, out.Query)
	})
}
