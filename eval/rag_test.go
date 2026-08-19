package eval

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOverlapJudge_Faithful(t *testing.T) {
	j := NewOverlapJudge()
	q, err := j.Judge(context.Background(), JudgeInput{
		Question: "What is the capital of France?",
		Context:  "Paris is the capital of France.",
		Answer:   "The capital of France is Paris.",
	})
	require.NoError(t, err)
	assert.InDelta(t, 1.0, q.Faithfulness, 1e-9)
	assert.InDelta(t, 1.0, q.Relevance, 1e-9)
	assert.Equal(t, 0.0, q.Correctness, "no reference -> 0 correctness")
}

func TestOverlapJudge_Hallucination(t *testing.T) {
	j := NewOverlapJudge()
	q, err := j.Judge(context.Background(), JudgeInput{
		Question: "What is the capital of France?",
		Context:  "Paris is the capital of France.",
		Answer:   "The capital of France is London.",
	})
	require.NoError(t, err)
	// "london" is unsupported by context -> faithfulness < 1
	assert.Less(t, q.Faithfulness, 1.0)
	assert.InDelta(t, 2.0/3.0, q.Faithfulness, 1e-9)
	// question words (capital, france) still addressed
	assert.InDelta(t, 1.0, q.Relevance, 1e-9)
}

func TestOverlapJudge_Correctness(t *testing.T) {
	j := NewOverlapJudge()
	q, err := j.Judge(context.Background(), JudgeInput{
		Question:  "Capital of France?",
		Context:   "Paris is the capital of France.",
		Answer:    "Paris is the capital.",
		Reference: "Paris is the capital.",
	})
	require.NoError(t, err)
	assert.InDelta(t, 1.0, q.Correctness, 1e-9)
}

func TestOverlapJudge_EmptyAnswer(t *testing.T) {
	j := NewOverlapJudge()
	q, err := j.Judge(context.Background(), JudgeInput{
		Question: "What is the capital of France?",
		Context:  "Paris is the capital of France.",
		Answer:   "",
	})
	require.NoError(t, err)
	assert.Equal(t, 0.0, q.Faithfulness)
	assert.Equal(t, 0.0, q.Relevance)
}

func TestRAGEval_NilJudge(t *testing.T) {
	e := NewRAGEval(nil)
	_, err := e.Evaluate(context.Background(), JudgeInput{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requires a Judge")
}

func TestRAGEval_Delegates(t *testing.T) {
	e := NewRAGEval(NewOverlapJudge())
	q, err := e.Evaluate(context.Background(), JudgeInput{
		Question: "Capital of France?",
		Context:  "Paris is the capital of France.",
		Answer:   "Paris is the capital of France.",
	})
	require.NoError(t, err)
	assert.InDelta(t, 1.0, q.Faithfulness, 1e-9)
}

func TestTokenF1(t *testing.T) {
	assert.InDelta(t, 1.0, tokenF1([]string{"a", "b", "c"}, []string{"a", "b", "c"}), 1e-9)
	assert.InDelta(t, 0.5, tokenF1([]string{"a", "b"}, []string{"b", "c"}), 1e-9)
	assert.Equal(t, 0.0, tokenF1(nil, []string{"a"}))
	assert.Equal(t, 0.0, tokenF1([]string{"a"}, nil))
	assert.Equal(t, 0.0, tokenF1([]string{"a"}, []string{"b"}))
}

// stubJudge is a custom Judge to confirm the interface is pluggable.
type stubJudge struct {
	q   AnswerQuality
	err error
}

func (s *stubJudge) Judge(ctx context.Context, in JudgeInput) (AnswerQuality, error) {
	return s.q, s.err
}

func TestJudgeInterface_Pluggable(t *testing.T) {
	e := NewRAGEval(&stubJudge{q: AnswerQuality{Faithfulness: 0.42}})
	got, err := e.Evaluate(context.Background(), JudgeInput{})
	require.NoError(t, err)
	assert.InDelta(t, 0.42, got.Faithfulness, 1e-9)
}
