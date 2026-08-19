package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"unicode/utf8"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

// printer renders command results in the configured format (table, json,
// or yaml).
type printer struct {
	w      io.Writer
	format string
}

func newPrinter(cmd *cobra.Command, format string) *printer {
	return &printer{w: cmd.OutOrStdout(), format: format}
}

// emit renders v as JSON or YAML, or invokes table for the table format.
func (p *printer) emit(v any, table func(p *printer)) error {
	switch p.format {
	case "json":
		return p.writeJSON(v)
	case "yaml":
		data, err := yaml.Marshal(v)
		if err != nil {
			return fmt.Errorf("encoding yaml: %w", err)
		}
		_, err = p.w.Write(data)
		return err
	default:
		if table != nil {
			table(p)
			return nil
		}
		return p.writeJSON(v)
	}
}

func (p *printer) writeJSON(v any) error {
	enc := json.NewEncoder(p.w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return fmt.Errorf("encoding json: %w", err)
	}
	return nil
}

// table starts a tabwriter table with an upper-cased header row. Call
// Flush on the returned writer when done.
func (p *printer) table(header ...string) *tabwriter.Writer {
	tw := tabwriter.NewWriter(p.w, 0, 4, 2, ' ', 0)
	if len(header) > 0 {
		upper := make([]string, len(header))
		for i, h := range header {
			upper[i] = strings.ToUpper(h)
		}
		fmt.Fprintln(tw, strings.Join(upper, "\t"))
	}
	return tw
}

// snippet truncates s to at most n runes, appending an ellipsis when
// truncated. Newlines are flattened so table rows stay single-line.
func snippet(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n]) + "…"
}

// truncate reports s limited to n runes (no ellipsis), for blocks like RAG
// contexts that the user asked to see in full.
func truncate(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return string([]rune(s)[:n]) + "\n… (truncated)"
}
