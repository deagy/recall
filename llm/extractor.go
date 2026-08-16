package llm

import (
	"context"
	"encoding/json"
	"fmt"
)

// LLMExtractor extracts entities and relations using an LLM.
type LLMExtractor struct {
	backend Backend
	model   string
}

// NewLLMExtractor creates a new LLMExtractor.
func NewLLMExtractor(backend Backend, model string) *LLMExtractor {
	return &LLMExtractor{
		backend: backend,
		model:   model,
	}
}

// ExtractEntities extracts entities from text using the LLM.
func (e *LLMExtractor) ExtractEntities(ctx context.Context, text string, sourceChunkID string) ([]Entity, error) {
	prompt := fmt.Sprintf("Extract all entities from the following text. Return a JSON array of objects with id, label, type, and confidence fields.\n\nText: %s", text)

	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a helpful assistant that extracts entities from text."},
			{Role: "user", Content: prompt},
		},
		Model:     e.model,
		MaxTokens: 1000,
		Stream:    false,
	}

	resp, err := e.backend.Chat(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM extraction failed: %w", err)
	}

	var entities []Entity
	if err := json.Unmarshal([]byte(resp.Message.Content), &entities); err != nil {
		return nil, fmt.Errorf("failed to parse entities: %w", err)
	}

	for i := range entities {
		if entities[i].Properties == nil {
			entities[i].Properties = make(map[string]string)
		}
		entities[i].Properties["source_chunk_id"] = sourceChunkID
	}

	return entities, nil
}

// ExtractRelations extracts relations from text using the LLM.
func (e *LLMExtractor) ExtractRelations(ctx context.Context, text string, sourceChunkID string) ([]Relation, error) {
	prompt := fmt.Sprintf("Extract all relations from the following text. Return a JSON array of objects with from, to, type, and confidence fields.\n\nText: %s", text)

	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a helpful assistant that extracts relations from text."},
			{Role: "user", Content: prompt},
		},
		Model:     e.model,
		MaxTokens: 1000,
		Stream:    false,
	}

	resp, err := e.backend.Chat(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM extraction failed: %w", err)
	}

	var relations []Relation
	if err := json.Unmarshal([]byte(resp.Message.Content), &relations); err != nil {
		return nil, fmt.Errorf("failed to parse relations: %w", err)
	}

	for i := range relations {
		if relations[i].Properties == nil {
			relations[i].Properties = make(map[string]string)
		}
		relations[i].Properties["source_chunk_id"] = sourceChunkID
	}

	return relations, nil
}
