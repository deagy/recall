package query

import (
	"context"
	"fmt"
	"strings"

	"github.com/deagy/recall/llm"
)

// SubQueryDecomposer decomposes a complex question into independent
// sub-queries for retrieval. Heuristic decomposition (conjunctions,
// multiple question marks) is always available; when an LLM backend is
// configured it is tried first and the heuristic result is used as a
// fallback.
type SubQueryDecomposer struct {
	// Backend optionally decomposes via LLM. Nil = heuristic only.
	Backend llm.Backend

	// MaxSubQueries caps the number of sub-queries. Default 5.
	MaxSubQueries int
}

const defaultDecomposePrompt = `Decompose the user's question into up to %d independent retrieval
sub-queries, one per line, each answerable on its own from a single
document section. If the question is already atomic, reply with it
unchanged. Reply with the lines only.`

// NewSubQueryDecomposer creates a heuristic-only decomposer.
func NewSubQueryDecomposer() *SubQueryDecomposer {
	return &SubQueryDecomposer{MaxSubQueries: 5}
}

// WithBackend enables LLM-based decomposition with the given backend.
func (d *SubQueryDecomposer) WithBackend(b llm.Backend) *SubQueryDecomposer {
	d.Backend = b
	return d
}

// Decompose returns the sub-queries for the input. The result is
// non-empty whenever the input is non-empty.
func (d *SubQueryDecomposer) Decompose(ctx context.Context, query string) ([]string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Errorf("query: empty query cannot be decomposed")
	}
	limit := d.MaxSubQueries
	if limit <= 0 {
		limit = 5
	}

	if d.Backend != nil {
		prompt := fmt.Sprintf(defaultDecomposePrompt, limit)
		if out, err := chatSystemUser(ctx, d.Backend, prompt, query); err == nil {
			if subs := parseLineList(out, limit); len(subs) > 1 {
				return subs, nil
			}
		}
		// LLM unavailable or degenerate output: fall back to heuristics.
	}
	return heuristicDecompose(query, limit), nil
}

// heuristicDecompose splits on multiple question marks and common
// conjunctions.
func heuristicDecompose(query string, limit int) []string {
	var candidates []string
	// Multiple explicit questions.
	if strings.Count(query, "?") >= 2 {
		parts := strings.Split(query, "?")
		for i, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			if i < len(parts)-1 {
				p += "?"
			}
			candidates = append(candidates, p)
		}
	} else {
		// Single conjunction-based split.
		for _, sep := range []string{" and ", " or ", "; ", " what about ", " also "} {
			lower := strings.ToLower(query)
			if idx := strings.Index(lower, sep); idx > 0 {
				a := strings.TrimSpace(query[:idx])
				b := strings.TrimSpace(query[idx+len(sep):])
				if a != "" && b != "" {
					candidates = append(candidates, a, b)
					break
				}
			}
		}
	}
	if len(candidates) <= 1 {
		return []string{query}
	}
	return parseLineList(strings.Join(candidates, "\n"), limit)
}

// parseLineList trims lines, drops empties and duplicates, caps at limit.
func parseLineList(raw string, limit int) []string {
	if limit <= 0 {
		limit = 5
	}
	seen := make(map[string]bool)
	var out []string
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "-"))
		line = strings.TrimSpace(line)
		if line == "" || seen[line] {
			continue
		}
		seen[line] = true
		out = append(out, line)
		if len(out) >= limit {
			break
		}
	}
	return out
}
