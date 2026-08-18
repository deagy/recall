package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/deagy/recall/llm"
	"github.com/deagy/recall/store"
)

// MultiModalRAGResponse is the outcome of a multi-modal RAG answer.
type MultiModalRAGResponse struct {
	// Answer is the LLM answer, or the rendered prompt when no LLM
	// backend is configured.
	Answer string

	// Context is the assembled multi-modal context string.
	Context string

	// Sources are the retrieved items with their relevance scores.
	Sources []store.MultiModalResult
}

// MultiModalPipeline assembles context from a MultiModalStore (text
// and image artifacts in one shared embedding space) and produces an
// answer via an optional LLM backend. Image sources are represented in
// the context by their captions; callers rendering a UI can pull the
// raw bytes from Source.Item.Image.
type MultiModalPipeline struct {
	store    *store.MultiModalStore
	backend  llm.Backend
	template *Template
	topK     int
}

// NewMultiModalPipeline creates a pipeline over the given store.
// backend may be nil, in which case Answer holds the rendered prompt
// instead of an LLM completion.
func NewMultiModalPipeline(s *store.MultiModalStore, backend llm.Backend) *MultiModalPipeline {
	return &MultiModalPipeline{
		store:    s,
		backend:  backend,
		template: DefaultTemplate(),
		topK:     5,
	}
}

// WithTopK sets how many items are retrieved per query.
func (p *MultiModalPipeline) WithTopK(k int) *MultiModalPipeline {
	if k > 0 {
		p.topK = k
	}
	return p
}

// WithTemplate overrides the prompt template.
func (p *MultiModalPipeline) WithTemplate(t *Template) *MultiModalPipeline {
	if t != nil {
		p.template = t
	}
	return p
}

// Answer retrieves items for the question, assembles multi-modal
// context, and returns the answer (LLM-backed when available).
func (p *MultiModalPipeline) Answer(ctx context.Context, question string) (*MultiModalRAGResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if strings.TrimSpace(question) == "" {
		return nil, fmt.Errorf("multimodal pipeline: empty question")
	}
	results, err := p.store.SearchText(ctx, question, p.topK)
	if err != nil {
		return nil, err
	}

	var ctxLines []string
	for i, r := range results {
		it := r.Item
		if it.Modality == "image" {
			caption := it.Content
			if caption == "" {
				caption = "(uncaptioned image)"
			}
			ctxLines = append(ctxLines, fmt.Sprintf("%d. [image %s] %s", i+1, it.MimeType, caption))
		} else {
			ctxLines = append(ctxLines, fmt.Sprintf("%d. [text] %s", i+1, it.Content))
		}
	}
	contextText := strings.Join(ctxLines, "\n")
	if contextText == "" {
		contextText = "(no relevant content found)"
	}

	resp := &MultiModalRAGResponse{Context: contextText, Sources: results}

	rendered := p.template.RenderUser(map[string]interface{}{
		"Context":  contextText,
		"Question": question,
	})
	if p.backend == nil {
		resp.Answer = p.template.Render(map[string]interface{}{
			"Context":  contextText,
			"Question": question,
		})
		return resp, nil
	}

	chatResp, err := p.backend.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: "system", Content: p.template.RenderSystem(nil)},
			{Role: "user", Content: rendered},
		},
		Temperature: 0,
	})
	if err != nil {
		return nil, fmt.Errorf("multimodal answer: %w", err)
	}
	resp.Answer = strings.TrimSpace(chatResp.Message.Content)
	return resp, nil
}
