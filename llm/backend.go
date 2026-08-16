// Package llm provides LLM integration for Recall, including pluggable backends,
// streaming support, and LLM-assisted extraction capabilities.
package llm

import (
	"context"
)

// Message represents a message in a conversation.
type Message struct {
	// Role is the role of the message sender (system, user, assistant).
	Role string

	// Content is the message content.
	Content string
}

// ChatRequest represents a request to the LLM.
type ChatRequest struct {
	// Messages is the conversation history.
	Messages []Message

	// Model is the model to use (e.g., "gpt-4", "ollama/llama2").
	Model string

	// Temperature controls randomness (0.0 to 2.0).
	Temperature float64

	// MaxTokens is the maximum number of tokens in the response.
	MaxTokens int

	// Stop is a list of strings to stop generation.
	Stop []string

	// Stream enables streaming mode.
	Stream bool

	// ResponseFormat specifies the expected response format.
	ResponseFormat *ResponseFormat
}

// ResponseFormat specifies the expected response format.
type ResponseFormat struct {
	// Type is the format type (text, json_object, json_schema).
	Type string

	// JSONSchema is the JSON schema for json_schema format.
	JSONSchema *JSONSchema
}

// JSONSchema defines a JSON schema for structured responses.
type JSONSchema struct {
	// Name is the schema name.
	Name string

	// Schema is the JSON schema definition.
	Schema map[string]interface{}
}

// ChatResponse represents a response from the LLM.
type ChatResponse struct {
	// Message is the assistant's response message.
	Message Message

	// Usage contains token usage statistics.
	Usage Usage

	// FinishReason indicates why generation stopped.
	FinishReason string
}

// Usage contains token usage statistics.
type Usage struct {
	// PromptTokens is the number of tokens in the prompt.
	PromptTokens int

	// CompletionTokens is the number of tokens in the completion.
	CompletionTokens int

	// TotalTokens is the total number of tokens.
	TotalTokens int
}

// StreamChunk represents a chunk of a streaming response.
type StreamChunk struct {
	// Delta is the content delta for this chunk.
	Delta Message

	// Usage contains token usage statistics (only in final chunk).
	Usage *Usage

	// FinishReason indicates why generation stopped (only in final chunk).
	FinishReason string
}

// Backend defines the interface for LLM backends.
type Backend interface {
	// Chat sends a chat request and returns a response.
	Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// ChatStream sends a chat request with streaming enabled.
	ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error
}

// Extractor defines the interface for LLM-assisted extraction.
type Extractor interface {
	// ExtractEntities extracts entities from text using the LLM.
	ExtractEntities(ctx context.Context, text string, sourceChunkID string) ([]Entity, error)

	// ExtractRelations extracts relations from text using the LLM.
	ExtractRelations(ctx context.Context, text string, sourceChunkID string) ([]Relation, error)
}

// Entity represents an entity extracted by the LLM.
type Entity struct {
	// ID is a unique identifier.
	ID string

	// Label is a human-readable name.
	Label string

	// Type is the entity type.
	Type string

	// Confidence is the confidence score (0.0 to 1.0).
	Confidence float64

	// Properties are additional metadata.
	Properties map[string]string
}

// Relation represents a relation extracted by the LLM.
type Relation struct {
	// From is the source entity ID.
	From string

	// To is the target entity ID.
	To string

	// Type is the relation type.
	Type string

	// Confidence is the confidence score (0.0 to 1.0).
	Confidence float64

	// Properties are additional metadata.
	Properties map[string]string
}
