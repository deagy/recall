package testutil_test

import (
	"context"
	"testing"

	"github.com/deagy/recall/llm"
	"github.com/deagy/recall/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMockLLM_ScriptedOrder(t *testing.T) {
	ctx := context.Background()
	m := testutil.NewMockLLM("first", "second")

	r1, err := m.Chat(ctx, &llm.ChatRequest{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	require.NoError(t, err)
	assert.Equal(t, "first", r1.Message.Content)
	assert.Equal(t, "hi", m.LastRequest().Messages[0].Content)
	assert.Equal(t, 1, m.Calls())

	r2, err := m.Chat(ctx, &llm.ChatRequest{})
	require.NoError(t, err)
	assert.Equal(t, "second", r2.Message.Content)

	r3, err := m.Chat(ctx, &llm.ChatRequest{})
	require.NoError(t, err)
	assert.Equal(t, "second", r3.Message.Content, "last scripted response repeats")

	assert.Equal(t, 3, m.Calls())
}

func TestMockLLM_DefaultAndTracking(t *testing.T) {
	ctx := context.Background()
	m := testutil.NewMockLLM()
	assert.Nil(t, m.LastRequest())
	r, err := m.Chat(ctx, &llm.ChatRequest{Model: "test-model"})
	require.NoError(t, err)
	assert.Equal(t, "mock answer", r.Message.Content)
	assert.Equal(t, 1, m.Calls())
	assert.Equal(t, "test-model", m.LastRequest().Model)
}

func TestMockLLM_Stream(t *testing.T) {
	ctx := context.Background()
	m := testutil.NewMockLLM("hello world")
	var chunks []llm.StreamChunk
	err := m.ChatStream(ctx, &llm.ChatRequest{}, func(c *llm.StreamChunk) error {
		chunks = append(chunks, *c)
		return nil
	})
	require.NoError(t, err)
	// two word chunks + one final usage chunk
	require.Len(t, chunks, 3)
	assert.Equal(t, "hello ", chunks[0].Delta.Content)
	assert.Equal(t, "world ", chunks[1].Delta.Content)
	assert.Nil(t, chunks[0].Usage)
	require.NotNil(t, chunks[2].Usage)
	assert.Equal(t, "stop", chunks[2].FinishReason)
}

func TestMockLLM_StreamEmptyResponse(t *testing.T) {
	ctx := context.Background()
	m := testutil.NewMockLLM("")
	var count int
	err := m.ChatStream(ctx, &llm.ChatRequest{}, func(c *llm.StreamChunk) error {
		count++
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 1, count, "empty response still emits the final chunk")
}
