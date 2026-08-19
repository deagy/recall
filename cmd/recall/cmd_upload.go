package main

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/loader"
	"github.com/deagy/recall/store"
)

// uploadDocResult is the outcome of uploading one document.
type uploadDocResult struct {
	ID     string `json:"id" yaml:"id"`
	Title  string `json:"title,omitempty" yaml:"title,omitempty"`
	Source string `json:"source,omitempty" yaml:"source,omitempty"`
	Chunks int    `json:"chunks,omitempty" yaml:"chunks,omitempty"`
	Status string `json:"status" yaml:"status"` // ok | failed
	Error  string `json:"error,omitempty" yaml:"error,omitempty"`
}

// uploadOutput is the aggregate result of `recall upload`.
type uploadOutput struct {
	Mode      string            `json:"mode" yaml:"mode"`
	Namespace string            `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	Documents []uploadDocResult `json:"documents" yaml:"documents"`
	Uploaded  int               `json:"uploaded" yaml:"uploaded"`
	Failed    int               `json:"failed" yaml:"failed"`
}

func newUploadCmd(o *globalOptions) *cobra.Command {
	var recursive bool
	cmd := &cobra.Command{
		Use:   "upload <path>...",
		Short: "Upload documents (files or directories) to the store",
		Long: `Upload documents to the store. Each argument may be a file (text,
markdown, CSV, JSON, HTML, PDF, DOCX) or a directory (scanned recursively
by default for supported file types).

In local mode the documents are ingested in-process into the configured
store; in server mode they are loaded locally and POSTed to the server.

With the default in-memory backend, data does not persist between CLI
runs — set store.backend: sqlite and store.path in your config for a
persistent store.`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runUpload(cmd, o, args, recursive)
		},
	}
	cmd.Flags().BoolVar(&recursive, "recursive", true, "recurse into subdirectories when a directory is given")
	return cmd
}

func runUpload(cmd *cobra.Command, o *globalOptions, paths []string, recursive bool) error {
	ctx := cmd.Context()
	out := &uploadOutput{Mode: "local", Namespace: o.effectiveNamespace(), Documents: []uploadDocResult{}}

	var localStore store.Store
	if o.cli == nil {
		if o.cfg.Store.Backend != config.BackendSQLite {
			fmt.Fprintln(os.Stderr, "warning: store backend is not sqlite; uploaded data will not persist between CLI runs (set store.backend: sqlite for a persistent store)")
		}
		st, err := o.openLocalStore()
		if err != nil {
			return err
		}
		defer st.Close()
		localStore = st
	}

	uploadOne := func(ctx context.Context, d *loader.Document) (int, error) {
		if o.cli != nil {
			up, err := o.cli.Upload(ctx, *clientUploadRequest(d, out.Namespace))
			if err != nil {
				return 0, err
			}
			return up.Chunks, nil
		}
		cd := toCoreDocument(d, out.Namespace)
		if err := localStore.Upload(ctx, cd, d.Content); err != nil {
			return 0, err
		}
		return cd.ChunkCount, nil
	}

	for _, path := range paths {
		ld, err := loaderForPath(path, recursive)
		if err != nil {
			out.Failed++
			out.Documents = append(out.Documents, uploadDocResult{Source: path, Status: "failed", Error: err.Error()})
			continue
		}
		docs, err := ld.Load(ctx, path)
		if err != nil {
			out.Failed++
			out.Documents = append(out.Documents, uploadDocResult{Source: path, Status: "failed", Error: err.Error()})
			continue
		}
		for _, d := range docs {
			res := uploadDocResult{ID: d.ID, Title: d.Title, Source: d.Source}
			chunks, err := uploadOne(ctx, d)
			if err != nil {
				res.Status = "failed"
				res.Error = err.Error()
				out.Failed++
			} else {
				res.Status = "ok"
				res.Chunks = chunks
				out.Uploaded++
			}
			out.Documents = append(out.Documents, res)
		}
	}

	p := newPrinter(cmd, o.output)
	if o.cli != nil {
		out.Mode = "server"
	}
	return p.emit(out, func(p *printer) {
		tw := p.table("id", "title", "source", "chunks", "status", "error")
		for _, d := range out.Documents {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\n", d.ID, d.Title, d.Source, d.Chunks, d.Status, d.Error)
		}
		tw.Flush()
		fmt.Fprintf(p.w, "uploaded %d document(s), %d failed (mode: %s, namespace: %q)\n", out.Uploaded, out.Failed, out.Mode, out.Namespace)
	})
}

// clientUploadRequest maps a loaded document to the API upload request.
func clientUploadRequest(d *loader.Document, namespace string) *client.UploadRequest {
	return &client.UploadRequest{
		ID:        d.ID,
		Title:     d.Title,
		Source:    d.Source,
		Namespace: namespace,
		Metadata:  metadataToAny(d.Metadata),
		Content:   d.Content,
	}
}
