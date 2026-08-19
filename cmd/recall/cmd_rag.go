package main

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
	"github.com/deagy/recall/pipeline"
)

// ragSource is a retrieved source chunk in `recall rag` output.
type ragSource struct {
	ID       string  `json:"id" yaml:"id"`
	Document string  `json:"document" yaml:"document"`
	Score    float64 `json:"score" yaml:"score"`
}

// ragCitation is a ranked citation reference.
type ragCitation struct {
	Number   int     `json:"number" yaml:"number"`
	ChunkID  string  `json:"chunk_id" yaml:"chunk_id"`
	Document string  `json:"document,omitempty" yaml:"document,omitempty"`
	Score    float64 `json:"score" yaml:"score"`
	Snippet  string  `json:"snippet,omitempty" yaml:"snippet,omitempty"`
}

// ragOutput is the result of `recall rag`.
type ragOutput struct {
	Mode      string        `json:"mode" yaml:"mode"`
	Hybrid    bool          `json:"hybrid" yaml:"hybrid"`
	Query     string        `json:"query" yaml:"query"`
	Answer    string        `json:"answer" yaml:"answer"`
	Context   string        `json:"context" yaml:"context"`
	Tokens    int           `json:"tokens" yaml:"tokens"`
	Sources   []ragSource   `json:"sources" yaml:"sources"`
	Citations []ragCitation `json:"citations,omitempty" yaml:"citations,omitempty"`
}

func newRAGCmd(o *globalOptions) *cobra.Command {
	var (
		hybrid       bool
		topK         int
		minScore     float64
		maxTokens    int
		smartContext bool
	)
	cmd := &cobra.Command{
		Use:   "rag <query>",
		Short: "Run a RAG query (retrieve, assemble context, render the prompt)",
		Long: `Run a RAG query: retrieve relevant chunks, assemble the context window,
and render the prompt (the "answer" field is the rendered prompt, ready to
send to an LLM of your choice). Citations are included by default.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRAG(cmd, o, args[0], ragOpts{hybrid: hybrid, topK: topK, min: minScore, maxTok: maxTokens, smart: smartContext})
		},
	}
	cmd.Flags().BoolVar(&hybrid, "hybrid", false, "use hybrid retrieval (vector + BM25)")
	cmd.Flags().IntVar(&topK, "top-k", 10, "number of chunks to retrieve")
	cmd.Flags().Float64Var(&minScore, "min-score", 0, "minimum relevance score")
	cmd.Flags().IntVar(&maxTokens, "max-tokens", 4096, "context window token budget")
	cmd.Flags().BoolVar(&smartContext, "smart-context", false, "select chunks by score within the token budget")
	return cmd
}

type ragOpts struct {
	hybrid bool
	topK   int
	min    float64
	maxTok int
	smart  bool
}

func runRAG(cmd *cobra.Command, o *globalOptions, query string, f ragOpts) error {
	ctx := cmd.Context()

	if o.cli != nil {
		resp, err := o.cli.RAG(ctx, query, f.hybrid)
		if err != nil {
			return err
		}
		return emitRAG(cmd, o, ragFromAPI(query, f.hybrid, resp))
	}

	st, err := o.openLocalStore()
	if err != nil {
		return err
	}
	defer st.Close()

	p := pipeline.NewRAGPipeline(st, nil).
		WithCitations().
		WithTopK(f.topK).
		WithMinScore(f.min).
		WithMaxTokens(f.maxTok)
	if f.smart {
		p = p.WithSmartContext()
	}
	if ns := namespaceFilterValue(cmd, o); ns != "" {
		p = p.WithSearchFilters(namespaceFilter(ns)...)
	}

	var resp *pipeline.RAGResponse
	if f.hybrid {
		resp, err = p.QueryHybrid(ctx, query)
	} else {
		resp, err = p.Query(ctx, query)
	}
	if err != nil {
		return err
	}
	return emitRAG(cmd, o, ragFromLocal(query, f.hybrid, resp))
}

func ragFromAPI(query string, hybrid bool, resp *client.RAGResponse) *ragOutput {
	out := &ragOutput{Mode: "server", Hybrid: hybrid, Query: resp.Query, Answer: resp.Answer, Context: resp.Context, Tokens: resp.Tokens}
	for _, s := range resp.Sources {
		out.Sources = append(out.Sources, ragSource{ID: s.ID, Document: s.Document, Score: s.Score})
	}
	for _, c := range resp.Citations {
		out.Citations = append(out.Citations, ragCitation{Number: c.Number, ChunkID: c.ChunkID, Document: c.Document, Score: c.Score, Snippet: c.Snippet})
	}
	return out
}

func ragFromLocal(query string, hybrid bool, resp *pipeline.RAGResponse) *ragOutput {
	out := &ragOutput{Mode: "local", Hybrid: hybrid, Query: query, Answer: resp.Answer, Context: resp.Context, Tokens: resp.Tokens}
	for _, s := range resp.Sources {
		if s.Chunk == nil {
			continue
		}
		out.Sources = append(out.Sources, ragSource{ID: s.Chunk.ID, Document: s.Chunk.DocumentRef, Score: s.Score})
	}
	for _, c := range resp.Citations {
		out.Citations = append(out.Citations, ragCitation{Number: c.Number, ChunkID: c.ChunkID, Document: c.DocumentRef, Score: c.Score, Snippet: c.Snippet})
	}
	return out
}

func emitRAG(cmd *cobra.Command, o *globalOptions, out *ragOutput) error {
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		fmt.Fprintf(p.w, "query:   %s\n", out.Query)
		fmt.Fprintf(p.w, "tokens:  %d\n", out.Tokens)
		fmt.Fprintf(p.w, "sources: %d\n\n", len(out.Sources))
		fmt.Fprintln(p.w, "answer:")
		fmt.Fprintln(p.w, out.Answer)
		fmt.Fprintln(p.w, "\ncontext:")
		fmt.Fprintln(p.w, out.Context)
		if len(out.Sources) > 0 {
			fmt.Fprintln(p.w, "\nsources:")
			tw := p.table("rank", "score", "id", "document")
			for i, s := range out.Sources {
				fmt.Fprintf(tw, "%d\t%.4f\t%s\t%s\n", i+1, s.Score, s.ID, s.Document)
			}
			tw.Flush()
		}
		if len(out.Citations) > 0 {
			fmt.Fprintln(p.w, "citations:")
			for _, c := range out.Citations {
				fmt.Fprintf(p.w, "[%d] %s (score %.3f): %s\n", c.Number, c.ChunkID, c.Score, snippet(c.Snippet, 80))
			}
		}
	})
}
