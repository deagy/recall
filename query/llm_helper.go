package query

import (
	"context"
	"fmt"
	"strings"

	"github.com/deagy/recall/llm"
)

// chatSystemUser sends a system+user prompt to the backend and returns
// the trimmed assistant reply. When allowEmpty is true an empty reply
// is returned as "" instead of an error.
func chatSystemUser(ctx context.Context, b llm.Backend, system, user string, allowEmpty ...bool) (string, error) {
	if b == nil {
		return "", fmt.Errorf("query: llm backend is required")
	}
	resp, err := b.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: "system", Content: system},
			{Role: "user", Content: user},
		},
		Temperature: 0,
	})
	if err != nil {
		return "", err
	}
	out := strings.TrimSpace(resp.Message.Content)
	if out == "" {
		if len(allowEmpty) > 0 && allowEmpty[0] {
			return "", nil
		}
		return "", fmt.Errorf("query: empty llm response")
	}
	return out, nil
}
