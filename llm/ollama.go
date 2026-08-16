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

// OllamaClient is an LLM backend for Ollama.
type OllamaClient struct {
	// BaseURL is the Ollama API base URL.
	BaseURL string

	// HTTPClient is the HTTP client to use.
	HTTPClient *http.Client

	// Timeout is the request timeout.
	Timeout time.Duration
}

// NewOllamaClient creates a new OllamaClient.
func NewOllamaClient(baseURL string) *OllamaClient {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	return &OllamaClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Chat sends a chat request to Ollama.
func (c *OllamaClient) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	ollamaReq := c.convertRequest(req)

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/chat", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var ollamaResp OllamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return c.convertResponse(&ollamaResp), nil
}

// ChatStream sends a chat request with streaming enabled.
func (c *OllamaClient) ChatStream(ctx context.Context, req *ChatRequest, fn func(chunk *StreamChunk) error) error {
	ollamaReq := c.convertRequest(req)
	ollamaReq.Stream = true

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/chat", bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

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
		var ollamaChunk OllamaStreamChunk
		if err := decoder.Decode(&ollamaChunk); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to decode stream chunk: %w", err)
		}

		chunk := c.convertStreamChunk(&ollamaChunk)
		if err := fn(chunk); err != nil {
			return err
		}

		if ollamaChunk.Done {
			break
		}
	}

	return nil
}

// convertRequest converts a ChatRequest to Ollama format.
func (c *OllamaClient) convertRequest(req *ChatRequest) OllamaChatRequest {
	ollamaReq := OllamaChatRequest{
		Model: req.Model,
	}

	for _, msg := range req.Messages {
		ollamaReq.Messages = append(ollamaReq.Messages, OllamaMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	if req.Temperature > 0 {
		ollamaReq.Options = OllamaOptions{
			Temperature: req.Temperature,
		}
	}

	return ollamaReq
}

// convertResponse converts an Ollama response to ChatResponse.
func (c *OllamaClient) convertResponse(resp *OllamaChatResponse) *ChatResponse {
	msg := resp.Message

	return &ChatResponse{
		Message: Message{
			Role:    msg.Role,
			Content: msg.Content,
		},
		Usage: Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
		FinishReason: "stop",
	}
}

// convertStreamChunk converts an Ollama stream chunk to StreamChunk.
func (c *OllamaClient) convertStreamChunk(chunk *OllamaStreamChunk) *StreamChunk {
	result := &StreamChunk{
		Delta: Message{
			Role:    chunk.Message.Role,
			Content: chunk.Message.Content,
		},
	}

	if chunk.Done {
		result.Usage = &Usage{
			PromptTokens:     chunk.PromptEvalCount,
			CompletionTokens: chunk.EvalCount,
			TotalTokens:      chunk.PromptEvalCount + chunk.EvalCount,
		}
		result.FinishReason = "stop"
	}

	return result
}

// OllamaChatRequest represents an Ollama chat request.
type OllamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream,omitempty"`
	Options  OllamaOptions   `json:"options,omitempty"`
}

// OllamaMessage represents a message in an Ollama request.
type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OllamaOptions represents Ollama model options.
type OllamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
}

// OllamaChatResponse represents an Ollama chat response.
type OllamaChatResponse struct {
	Model           string        `json:"model"`
	CreatedAt       string        `json:"created_at"`
	Message         OllamaMessage `json:"message"`
	Done            bool          `json:"done"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
}

// OllamaStreamChunk represents a stream chunk from Ollama.
type OllamaStreamChunk struct {
	Model           string        `json:"model"`
	CreatedAt       string        `json:"created_at"`
	Message         OllamaMessage `json:"message"`
	Done            bool          `json:"done"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
}
