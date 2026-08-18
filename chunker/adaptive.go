package chunker

import (
	"strings"
	"unicode/utf8"

	"github.com/deagy/recall/core"
)

// AdaptiveChunker tunes its chunk size per document based on content
// structure. It measures the distribution of sentence lengths and picks
// a target chunk size so that chunks align with natural sentence
// boundaries: documents with long, dense sentences get larger chunks
// (to avoid mid-thought cuts), documents with short, choppy sentences
// get smaller chunks (to keep each chunk topically tight). The
// configured MaxTokens/MinChunkSize act as hard bounds.
type AdaptiveChunker struct {
	// MinTokens is the lower bound on the chosen chunk size.
	MinTokens int

	// MaxTokens is the upper bound on the chosen chunk size.
	MaxTokens int

	// SentencesPerChunk targets how many sentences a chunk holds
	// before it is cut. Default 6.
	SentencesPerChunk int

	// Separator between parts inside a chunk. Default "\n\n".
	Separator string
}

// NewAdaptive creates an AdaptiveChunker from a base Config:
// MaxTokens becomes the upper bound and MinChunkSize the lower bound
// (converted at 4 chars/token).
func NewAdaptive(cfg Config) *AdaptiveChunker {
	minTokens := cfg.MinChunkSize / 4
	if minTokens <= 0 {
		minTokens = 64
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	if minTokens > maxTokens {
		minTokens = maxTokens / 4
	}
	return &AdaptiveChunker{
		MinTokens:         minTokens,
		MaxTokens:         maxTokens,
		SentencesPerChunk: 6,
		Separator:         cfg.Separator,
	}
}

// EstimateTokens returns the chunk size (in tokens) the adaptive
// chunker would choose for this content.
func (a *AdaptiveChunker) EstimateTokens(content string) int {
	sentences := splitSentenceRuns(content)
	if len(sentences) == 0 {
		return a.MinTokens
	}
	lens := make([]int, 0, len(sentences))
	for _, s := range sentences {
		lens = append(lens, utf8.RuneCountInString(strings.TrimSpace(s)))
	}
	mean := 0
	for _, l := range lens {
		mean += l
	}
	mean /= len(lens)
	if mean <= 0 {
		return a.MinTokens
	}
	target := mean * a.SentencesPerChunk
	tokens := target / 4
	if tokens < a.MinTokens {
		tokens = a.MinTokens
	}
	if tokens > a.MaxTokens {
		tokens = a.MaxTokens
	}
	return tokens
}

// Chunk splits the document using a content-adaptive fixed chunker.
func (a *AdaptiveChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}
	tokens := a.EstimateTokens(content)
	cfg := Config{
		MaxTokens:     tokens,
		MinChunkSize:  tokens * 4 / 4, // MinChunkSize is a char count; use 1 token ≈ 4 chars
		OverlapTokens: tokens / 8,
		Separator:     a.Separator,
	}
	if cfg.Separator == "" {
		cfg.Separator = "\n\n"
	}
	return NewFixed(cfg).Chunk(doc, content)
}

// splitSentenceRuns splits content into sentences on terminal punctuation
// and newlines.
func splitSentenceRuns(content string) []string {
	var out []string
	var cur strings.Builder
	for _, r := range content {
		cur.WriteRune(r)
		if r == '.' || r == '!' || r == '?' || r == '\n' {
			if s := strings.TrimSpace(cur.String()); s != "" {
				out = append(out, s)
			}
			cur.Reset()
		}
	}
	if s := strings.TrimSpace(cur.String()); s != "" {
		out = append(out, s)
	}
	return out
}
