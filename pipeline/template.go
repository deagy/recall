// Package pipeline provides RAG (Retrieval-Augmented Generation) pipeline
// functionality including context window management, prompt templating,
// and context assembly for LLM queries.
package pipeline

import (
	"fmt"
	"strings"
)

// Template represents a prompt template with system and user instructions.
// Variables are substituted using {{.VarName}} syntax.
type Template struct {
	System string
	User   string
}

// NewTemplate creates a new template with the given system and user strings.
func NewTemplate(system, user string) *Template {
	return &Template{
		System: system,
		User:   user,
	}
}

// DefaultTemplate returns a sensible default RAG template.
func DefaultTemplate() *Template {
	return NewTemplate(
		"You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.",
		"Context:\n{{.Context}}\n\nQuestion: {{.Question}}\n\nAnswer:",
	)
}

// Render replaces {{.VarName}} placeholders with values from the vars map.
func (t *Template) Render(vars map[string]interface{}) string {
	result := t.System
	for k, v := range vars {
		placeholder := "{{." + k + "}}"
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", v))
	}
	result += "\n\n" + t.User
	for k, v := range vars {
		placeholder := "{{." + k + "}}"
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", v))
	}
	return result
}

// RenderSystem renders only the system prompt with variable substitution.
func (t *Template) RenderSystem(vars map[string]interface{}) string {
	result := t.System
	for k, v := range vars {
		placeholder := "{{." + k + "}}"
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", v))
	}
	return result
}

// RenderUser renders only the user prompt with variable substitution.
func (t *Template) RenderUser(vars map[string]interface{}) string {
	result := t.User
	for k, v := range vars {
		placeholder := "{{." + k + "}}"
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", v))
	}
	return result
}