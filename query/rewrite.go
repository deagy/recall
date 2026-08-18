package query

import (
	"context"
	"strings"

	"github.com/deagy/recall/llm"
)

// Rewriter performs LLM-powered query rewriting: it turns a raw user
// question into a retrieval-optimized query — expanded acronyms,
// removed filler, key terms made explicit — improving match rates
// against indexed content.
type Rewriter struct {
	// Backend is the LLM used for rewriting. Required.
	Backend llm.Backend

	// SystemPrompt overrides the default rewriting instructions.
	SystemPrompt string
}

const defaultRewritePrompt = `Rewrite the user's question as a concise retrieval query for a
document search engine. Preserve meaning, expand abbreviations, drop
conversational filler, and list the key concepts. Reply with the query
only, no explanation.`

// NewRewriter creates a Rewriter backed by the given LLM.
func NewRewriter(b llm.Backend) *Rewriter {
	return &Rewriter{Backend: b}
}

// Rewrite returns the retrieval-optimized form of the query.
func (r *Rewriter) Rewrite(ctx context.Context, query string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	system := r.SystemPrompt
	if system == "" {
		system = defaultRewritePrompt
	}
	out, err := chatSystemUser(ctx, r.Backend, system, query, true)
	if err != nil {
		return "", err
	}
	// Take the first line: models occasionally append commentary.
	if i := strings.IndexByte(out, '\n'); i >= 0 {
		out = strings.TrimSpace(out[:i])
	}
	if out == "" {
		return strings.TrimSpace(query), nil
	}
	return out, nil
}
