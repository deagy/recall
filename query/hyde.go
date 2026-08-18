package query

import (
	"context"
	"strings"

	"github.com/deagy/recall/llm"
)

// HyDE implements Hypothetical Document Embeddings: instead of embedding
// the short (often ambiguous) query, an LLM drafts a plausible answer
// paragraph, and that paragraph is embedded for retrieval. Documents
// similar to the hypothetical answer tend to be the ones actually
// answering the question, which closes the query-document lexical gap.
type HyDE struct {
	// Backend is the LLM that drafts the hypothetical document. Required.
	Backend llm.Backend

	// SystemPrompt overrides the default HyDE instructions.
	SystemPrompt string
}

const defaultHyDEPrompt = `Write a short, factual paragraph (3-5 sentences) that could serve as
the ideal answer to the user's question, in the style of a reference
manual or encyclopedia entry. Use concrete domain terminology. Do not
mention that you are unsure; do not ask follow-up questions. Reply with
the paragraph only.`

// NewHyDE creates a HyDE generator backed by the given LLM.
func NewHyDE(b llm.Backend) *HyDE { return &HyDE{Backend: b} }

// Generate returns a hypothetical answer document for the query. The
// caller embeds the returned text and searches with that embedding.
func (h *HyDE) Generate(ctx context.Context, query string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	system := h.SystemPrompt
	if system == "" {
		system = defaultHyDEPrompt
	}
	out, err := chatSystemUser(ctx, h.Backend, system, query)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}
