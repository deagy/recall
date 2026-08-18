package reranker

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/deagy/recall/index"
	"github.com/deagy/recall/llm"
)

// DefaultLLMRerankSystemPrompt is the system prompt used by LLMReranker when
// none is supplied.
const DefaultLLMRerankSystemPrompt = `You are a precise relevance judge for a retrieval system.
You will be given a query and a passage. Rate how well the passage answers the query on an integer scale from 0 to 10, where 0 is completely irrelevant and 10 is a perfect answer.
Reply with ONLY the integer score. No explanation, no extra text.`

// DefaultLLMRerankUserPrompt is the user prompt template. Placeholders are
// substituted with the query and passage.
const DefaultLLMRerankUserPrompt = "Query:\n%s\n\nPassage:\n%s\n\nRelevance score (0-10):"

// LLMRerankerConfig configures an LLMReranker.
type LLMRerankerConfig struct {
	// Backend is the LLM used to judge relevance (required).
	Backend llm.Backend

	// SystemPrompt overrides the judge's system message. Defaults to
	// DefaultLLMRerankSystemPrompt when empty.
	SystemPrompt string

	// UserPrompt overrides the per-passage prompt template. It must contain
	// two verbs for the query and passage placeholders, in that order.
	// Defaults to DefaultLLMRerankUserPrompt when empty.
	UserPrompt string

	// MaxCandidates limits how many top coarse results are judged. The rest
	// keep their coarse ordering at the tail. Zero means judge all.
	MaxCandidates int

	// Temperature controls sampling (0 recommended for deterministic judging).
	Temperature float64
}

// LLMReranker uses an LLM as a judge to score each candidate passage's
// relevance to the query. It is the highest-quality (and highest-cost)
// reranker in the package; the LLM backend is dependency-injected so no
// specific provider is required.
type LLMReranker struct {
	backend       llm.Backend
	systemPrompt  string
	userPrompt    string
	maxCandidates int
	temperature   float64
}

// NewLLMReranker creates an LLMReranker. It returns an error if no backend
// is provided.
func NewLLMReranker(cfg LLMRerankerConfig) (*LLMReranker, error) {
	if cfg.Backend == nil {
		return nil, fmt.Errorf("reranker: llm reranker requires a backend")
	}
	sys := cfg.SystemPrompt
	if sys == "" {
		sys = DefaultLLMRerankSystemPrompt
	}
	user := cfg.UserPrompt
	if user == "" {
		user = DefaultLLMRerankUserPrompt
	}
	return &LLMReranker{
		backend:       cfg.Backend,
		systemPrompt:  sys,
		userPrompt:    user,
		maxCandidates: cfg.MaxCandidates,
		temperature:   cfg.Temperature,
	}, nil
}

// Name implements Reranker.
func (r *LLMReranker) Name() string { return "llm-judge" }

// Rerank asks the LLM to score each candidate passage and returns the results
// ordered by the judge's score (0-1 normalized to 0-1). Candidates beyond
// MaxCandidates (when set) are appended at the tail in coarse order with the
// coarse score used as their RerankScore.
func (r *LLMReranker) Rerank(ctx context.Context, query string, results []index.SearchResult) ([]index.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	judgeN := len(results)
	if r.maxCandidates > 0 && r.maxCandidates < judgeN {
		judgeN = r.maxCandidates
	}

	scored := make([]float64, len(results))
	for i := 0; i < judgeN; i++ {
		res := results[i]
		if res.Chunk == nil {
			return nil, fmt.Errorf("reranker: llm: result %d has no chunk", i)
		}
		raw, err := r.judge(ctx, query, res.Chunk.Content)
		if err != nil {
			return nil, err
		}
		scored[i] = clampScore(raw, 10.0) / 10.0
	}

	out := make([]index.SearchResult, 0, len(results))
	for i := range results {
		r2 := results[i]
		if i < judgeN {
			r2.RerankScore = scored[i]
		} else {
			r2.RerankScore = r2.Score // fall back to coarse score
		}
		out = append(out, r2)
	}

	return finalize(r.Name(), out), nil
}

// judge sends one (query, passage) to the LLM and parses the integer score.
func (r *LLMReranker) judge(ctx context.Context, query, passage string) (float64, error) {
	resp, err := r.backend.Chat(ctx, &llm.ChatRequest{
		Messages: []llm.Message{
			{Role: "system", Content: r.systemPrompt},
			{Role: "user", Content: fmt.Sprintf(r.userPrompt, query, passage)},
		},
		Temperature: r.temperature,
	})
	if err != nil {
		return 0, err
	}
	return parseScore(resp.Message.Content), nil
}

// parseScore extracts the first numeric value from an LLM reply. It tolerates
// surrounding prose ("The score is 7.") and out-of-range values (clamped by
// the caller). A missing number yields 0.
func parseScore(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	// Fast path: the whole reply is a number (possibly decimal).
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	// Scan for the first run that parses as a float.
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '-' || c == '.' || (c >= '0' && c <= '9') {
			j := i
			for j < len(s) {
				d := s[j]
				if d == '-' || d == '.' || (d >= '0' && d <= '9') {
					j++
				} else {
					break
				}
			}
			if v, err := strconv.ParseFloat(s[i:j], 64); err == nil {
				return v
			}
			i = j
		}
	}
	return 0
}
