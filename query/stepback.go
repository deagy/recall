package query

import (
	"context"
	"strings"

	"github.com/deagy/recall/llm"
)

// StepBack implements step-back prompting: the LLM first answers
// "what is the more general concept or question behind this query?",
// producing a higher-level query that retrieves the foundational
// context. Retrieval can then use both the step-back query and the
// original, which improves recall for questions that depend on
// background knowledge.
type StepBack struct {
	// Backend is the LLM that derives the step-back question. Required.
	Backend llm.Backend

	// SystemPrompt overrides the default step-back instructions.
	SystemPrompt string
}

const defaultStepBackPrompt = `Given the user's question, state ONE more general or conceptual
question that must be understood in order to answer it (step back one
level of abstraction). The general question should be answerable from
background or foundational material. Reply with the question only.`

// NewStepBack creates a StepBack generator backed by the given LLM.
func NewStepBack(b llm.Backend) *StepBack { return &StepBack{Backend: b} }

// Generate returns the step-back (more abstract) question for the query.
func (s *StepBack) Generate(ctx context.Context, query string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	system := s.SystemPrompt
	if system == "" {
		system = defaultStepBackPrompt
	}
	out, err := chatSystemUser(ctx, s.Backend, system, query)
	if err != nil {
		return "", err
	}
	if i := strings.IndexByte(out, '\n'); i >= 0 {
		out = strings.TrimSpace(out[:i])
	}
	return out, nil
}
