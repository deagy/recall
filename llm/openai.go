package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OpenAIClient is an LLM backend for OpenAI's API.
type OpenAIClient struct {
	// APIKey is the OpenAI API key.
	APIKey string

	// BaseURL is the OpenAI API base URL.
	BaseURL string

	// HTTPClient is the HTTP client to use.
	HTTPClient *http.Client

	// Timeout is the request timeout.
	Timeout time.Duration
}

// NewOpenAIClient creates a new OpenAIClient.
func NewOpenAIClient(apiKey, baseURL string) *OpenAIClient {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	return &OpenAIClient{
		APIKey:  apiKey,
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Chat sends a chat request to OpenAI.
func (c *OpenAIClient) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	openaiReq := c.convertRequest(req)

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var openaiResp OpenAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return c.convertResponse(&openaiResp), nil
}

// ChatStream sends a chat request with streaming enabled.
func (c *OpenAIClient) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	openaiReq := c.convertRequest(req)
	openaiReq.Stream = true

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	decoder := json.NewDecoder(resp.Body)
	for {
		var openaiChunk OpenAIStreamChunk
		if err := decoder.Decode(&openaiChunk); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to decode stream chunk: %w", err)
		}

		chunk := c.convertStreamChunk(&openaiChunk)
		if err := fn(chunk); err != nil {
			return err
		}

		if openaiChunk.Choices[0].FinishReason != "" {
			break
		}
	}

	return nil
}

// convertRequest converts a ChatRequest to OpenAI format.
func (c *OpenAIClient) convertRequest(req *ChatRequest) OpenAIChatRequest {
	openaiReq := OpenAIChatRequest{
		Model: req.Model,
	}

	for _, msg := range req.Messages {
		openaiReq.Messages = append(openaiReq.Messages, OpenAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	if req.Temperature > 0 {
		openaiReq.Temperature = req.Temperature
	}
	if req.MaxTokens > 0 {
		openaiReq.MaxTokens = req.MaxTokens
	}
	if len(req.Stop) > 0 {
		openaiReq.Stop = req.Stop
	}

	return openaiReq
}

// convertResponse converts an OpenAI response to ChatResponse.
func (c *OpenAIClient) convertResponse(resp *OpenAIChatResponse) *ChatResponse {
	choice := resp.Choices[0]
	msg := choice.Message

	return &ChatResponse{
		Message: Message{
			Role:    msg.Role,
			Content: msg.Content,
		},
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		FinishReason: choice.FinishReason,
	}
}

// convertStreamChunk converts an OpenAI stream chunk to StreamChunk.
func (c *OpenAIClient) convertStreamChunk(chunk *OpenAIStreamChunk) *StreamChunk {
	choice := chunk.Choices[0]

	result := &StreamChunk{
		Delta: Message{
			Role:    choice.Delta.Role,
			Content: choice.Delta.Content,
		},
	}

	if chunk.Usage != nil {
		result.Usage = &Usage{
			PromptTokens:     chunk.Usage.PromptTokens,
			CompletionTokens: chunk.Usage.CompletionTokens,
			TotalTokens:      chunk.Usage.TotalTokens,
		}
	}

	if choice.FinishReason != "" {
		result.FinishReason = choice.FinishReason
	}

	return result
}

// OpenAIChatRequest represents an OpenAI chat completion request.
type OpenAIChatRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

// OpenAIMessage represents a message in an OpenAI request.
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIChatResponse represents an OpenAI chat completion response.
type OpenAIChatResponse struct {
	ID      string         `json:"id"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

// OpenAIChoice represents a choice in an OpenAI response.
type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// OpenAIUsage represents usage statistics.
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIStreamChunk represents a stream chunk from OpenAI.
type OpenAIStreamChunk struct {
	ID      string               `json:"id"`
	Choices []OpenAIStreamChoice `json:"choices"`
	Usage   *OpenAIUsage         `json:"usage,omitempty"`
}

// OpenAIStreamChoice represents a choice in a stream chunk.
type OpenAIStreamChoice struct {
	Index        int           `json:"index"`
	Delta        OpenAIMessage `json:"delta"`
	FinishReason string        `json:"finish_reason"`
}
