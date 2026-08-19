package pipeline

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
)

// estimateTokens approximates a token count as characters / 4, matching the
// heuristic used by ContextWindow.
func estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	return len(s) / 4
}

// ---------------------------------------------------------------------------
// SmartContextWindow
// ---------------------------------------------------------------------------

// ScoredChunk pairs a chunk with a priority score used for selection.
type ScoredChunk struct {
	// Chunk is the candidate chunk.
	Chunk core.Chunk

	// Score is the priority (higher is more important), e.g. a relevance score.
	Score float64
}

// SmartContextWindow selects which chunks fit a token budget using a
// priority-based policy: candidates are considered in descending score order
// and included while they fit, so the most relevant chunks win the limited
// context space.
type SmartContextWindow struct {
	MaxTokens int
}

// NewSmartContextWindow creates a SmartContextWindow with the given token
// budget. A non-positive budget defaults to 4096.
func NewSmartContextWindow(maxTokens int) *SmartContextWindow {
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	return &SmartContextWindow{MaxTokens: maxTokens}
}

// Select returns the subset of candidates that fits the token budget, chosen
// by priority. Candidates are considered in descending score order (ties broken
// by chunk ID for determinism); each is included if it fits the remaining
// budget, otherwise skipped. The result is ordered by descending score.
func (w *SmartContextWindow) Select(candidates []ScoredChunk) []core.Chunk {
	ordered := make([]ScoredChunk, len(candidates))
	copy(ordered, candidates)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Score != ordered[j].Score {
			return ordered[i].Score > ordered[j].Score
		}
		return ordered[i].Chunk.ID < ordered[j].Chunk.ID
	})

	used := 0
	selected := make([]core.Chunk, 0, len(ordered))
	for _, c := range ordered {
		tokens := estimateTokens(c.Chunk.Content)
		if used+tokens > w.MaxTokens {
			continue
		}
		used += tokens
		selected = append(selected, c.Chunk)
	}
	return selected
}

// ---------------------------------------------------------------------------
// Sentence / word helpers (deterministic, shared by compression & detection)
// ---------------------------------------------------------------------------

var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
	"in": true, "on": true, "at": true, "to": true, "of": true, "for": true,
	"with": true, "by": true, "is": true, "are": true, "was": true, "were": true,
	"be": true, "it": true, "its": true, "this": true, "that": true, "as": true,
	"from": true, "not": true, "no": true, "so": true, "if": true, "then": true,
	"than": true, "too": true, "very": true, "can": true, "will": true,
	"just": true, "about": true, "into": true, "over": true, "after": true,
	"before": true, "between": true, "out": true, "up": true, "down": true,
	"do": true, "does": true, "did": true, "have": true, "has": true, "had": true,
	"which": true, "who": true, "whom": true, "what": true, "when": true,
	"where": true, "why": true, "how": true, "all": true, "any": true, "each": true,
	"more": true, "most": true, "other": true, "some": true, "such": true, "only": true,
	"own": true, "same": true, "should": true, "would": true, "could": true,
}

// tokenize lowercases s and splits it into alphanumeric word tokens.
func tokenize(s string) []string {
	var (
		tokens []string
		cur    strings.Builder
	)
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

// isContentWord reports whether w is a meaningful (non-stopword) token.
func isContentWord(w string) bool {
	return len(w) >= 3 && !stopwords[w]
}

// contentWords returns the content words of s.
func contentWords(s string) []string {
	toks := tokenize(s)
	out := make([]string, 0, len(toks))
	for _, w := range toks {
		if isContentWord(w) {
			out = append(out, w)
		}
	}
	return out
}

// splitSentences splits text into trimmed, non-empty sentences.
func splitSentences(text string) []string {
	var (
		out []string
		cur strings.Builder
	)
	flush := func() {
		s := strings.TrimSpace(cur.String())
		if s != "" {
			out = append(out, s)
		}
		cur.Reset()
	}
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		cur.WriteRune(r)
		if r == '.' || r == '!' || r == '?' || r == '\n' {
			flush()
		}
	}
	flush()
	return out
}

// firstSentence returns the first sentence of s (or s itself if none).
func firstSentence(s string) string {
	if s = strings.TrimSpace(s); s == "" {
		return ""
	}
	if parts := splitSentences(s); len(parts) > 0 {
		return parts[0]
	}
	return s
}

// ---------------------------------------------------------------------------
// ContextCompression
// ---------------------------------------------------------------------------

// Summarizer reduces a piece of text to a shorter summary. It may be an LLM
// call or a deterministic heuristic.
type Summarizer func(ctx context.Context, text string) (string, error)

// ExtractiveSummarizer returns a deterministic, LLM-free summarizer that keeps
// the maxSentences most important sentences of a text. Importance is the sum
// of inverse document frequencies of its content words, plus a small bonus for
// earlier sentences.
func ExtractiveSummarizer(maxSentences int) Summarizer {
	if maxSentences <= 0 {
		maxSentences = 3
	}
	return func(ctx context.Context, text string) (string, error) {
		sentences := splitSentences(text)
		if len(sentences) <= maxSentences {
			return text, nil
		}

		// Document frequency of each content word.
		freq := make(map[string]int)
		for _, s := range sentences {
			seen := make(map[string]bool)
			for _, w := range contentWords(s) {
				if !seen[w] {
					seen[w] = true
					freq[w]++
				}
			}
		}

		type scored struct {
			idx   int
			score float64
		}
		scoredSentences := make([]scored, 0, len(sentences))
		n := float64(len(sentences))
		for i, s := range sentences {
			sc := 0.0
			for _, w := range contentWords(s) {
				sc += 1.0 / float64(freq[w])
			}
			// Small bonus for earlier sentences (they often carry the gist).
			sc += 0.1 * (n - float64(i)) / n
			scoredSentences = append(scoredSentences, scored{idx: i, score: sc})
		}
		sort.Slice(scoredSentences, func(i, j int) bool {
			if scoredSentences[i].score != scoredSentences[j].score {
				return scoredSentences[i].score > scoredSentences[j].score
			}
			return scoredSentences[i].idx < scoredSentences[j].idx
		})

		kept := make([]scored, 0, maxSentences)
		for i, sc := range scoredSentences {
			if i == maxSentences {
				break
			}
			kept = append(kept, sc)
		}
		// Restore original order for readability.
		sort.Slice(kept, func(i, j int) bool { return kept[i].idx < kept[j].idx })

		var b strings.Builder
		for i, sc := range kept {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString(sentences[sc.idx])
		}
		return b.String(), nil
	}
}

// ContextCompressor shortens oversized chunks by summarizing them, so long
// contexts fit a token budget without dropping whole sources.
type ContextCompressor struct {
	// Summarize is the summarization function. Defaults to ExtractiveSummarizer.
	Summarize Summarizer

	// MaxChunkTokens is the per-chunk token budget; chunks above it are
	// summarized. Zero uses 512.
	MaxChunkTokens int
}

// NewContextCompressor creates a ContextCompressor. A nil summarize uses the
// built-in extractive summarizer.
func NewContextCompressor(summarize Summarizer) *ContextCompressor {
	if summarize == nil {
		summarize = ExtractiveSummarizer(3)
	}
	return &ContextCompressor{Summarize: summarize, MaxChunkTokens: 512}
}

// SetMaxChunkTokens sets the per-chunk token budget.
func (c *ContextCompressor) SetMaxChunkTokens(n int) {
	if n > 0 {
		c.MaxChunkTokens = n
	}
}

// Compress returns the chunks with any that exceed MaxChunkTokens replaced by
// their summary. Chunks within the budget are returned unchanged.
func (c *ContextCompressor) Compress(ctx context.Context, chunks []core.Chunk) ([]core.Chunk, error) {
	out := make([]core.Chunk, 0, len(chunks))
	for _, ch := range chunks {
		if estimateTokens(ch.Content) > c.MaxChunkTokens {
			summary, err := c.Summarize(ctx, ch.Content)
			if err != nil {
				return nil, fmt.Errorf("summarizing chunk %s: %w", ch.ID, err)
			}
			cp := ch
			cp.Content = summary
			out = append(out, cp)
		} else {
			out = append(out, ch)
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// CitationTracking
// ---------------------------------------------------------------------------

// Citation is a single ranked reference to a source chunk.
type Citation struct {
	// Number is the 1-based citation index in rank order.
	Number int

	// ChunkID is the source chunk's ID.
	ChunkID string

	// DocumentRef is the source document reference.
	DocumentRef string

	// Score is the chunk's relevance score.
	Score float64

	// Snippet is a short excerpt (the chunk's first sentence).
	Snippet string
}

// String renders a citation as a single line.
func (c Citation) String() string {
	return fmt.Sprintf("[%d] %s (chunk %s, score %.3f): %s",
		c.Number, c.DocumentRef, c.ChunkID, c.Score, c.Snippet)
}

// TrackCitations builds ordered citations from search results. Each result
// (skipping nil chunks) becomes a citation numbered in rank order, carrying a
// short snippet of its first sentence.
func TrackCitations(results []index.SearchResult) []Citation {
	citations := make([]Citation, 0, len(results))
	for _, r := range results {
		if r.Chunk == nil {
			continue
		}
		citations = append(citations, Citation{
			Number:      len(citations) + 1,
			ChunkID:     r.Chunk.ID,
			DocumentRef: r.Chunk.DocumentRef,
			Score:       r.Score,
			Snippet:     firstSentence(r.Chunk.Content),
		})
	}
	return citations
}

// RenderCitations formats a citation list for appending to a prompt.
func RenderCitations(citations []Citation) string {
	if len(citations) == 0 {
		return ""
	}
	var b strings.Builder
	for _, c := range citations {
		b.WriteString(c.String())
		b.WriteString("\n")
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// HallucinationDetection
// ---------------------------------------------------------------------------

// ClaimCheck reports whether a single claim (sentence) is supported by the
// sources.
type ClaimCheck struct {
	// Claim is the checked sentence.
	Claim string

	// Supported is true when the claim's support ratio meets the threshold.
	Supported bool

	// Support is the fraction (0..1) of the claim's content words that appear
	// in the sources.
	Support float64
}

// HallucinationDetector flags claims in an answer that are not lexically
// grounded in the provided sources. It is a deterministic, model-free signal:
// a strong first pass for surfacing potentially unsupported statements.
type HallucinationDetector struct {
	// Threshold is the minimum support ratio for a claim to be considered
	// grounded. Zero uses 0.5.
	Threshold float64
}

// NewHallucinationDetector creates a detector with the given support threshold.
func NewHallucinationDetector(threshold float64) *HallucinationDetector {
	if threshold <= 0 {
		threshold = 0.5
	}
	return &HallucinationDetector{Threshold: threshold}
}

// Check evaluates each claim (sentence) in the answer against the sources,
// reporting the lexical support of each.
func (d *HallucinationDetector) Check(answer string, sources []core.Chunk) []ClaimCheck {
	corpus := make(map[string]bool)
	for _, ch := range sources {
		for _, w := range contentWords(ch.Content) {
			corpus[w] = true
		}
	}

	checks := make([]ClaimCheck, 0)
	for _, claim := range splitSentences(answer) {
		words := contentWords(claim)
		if len(words) == 0 {
			continue
		}
		supported := 0
		for _, w := range words {
			if corpus[w] {
				supported++
			}
		}
		ratio := float64(supported) / float64(len(words))
		checks = append(checks, ClaimCheck{
			Claim:     claim,
			Supported: ratio >= d.Threshold,
			Support:   ratio,
		})
	}
	return checks
}

// HallucinationRate returns the fraction of claims in the answer that are NOT
// supported by the sources (0..1). It returns 0 when the answer has no claims.
func (d *HallucinationDetector) HallucinationRate(answer string, sources []core.Chunk) float64 {
	checks := d.Check(answer, sources)
	if len(checks) == 0 {
		return 0
	}
	unsupported := 0
	for _, c := range checks {
		if !c.Supported {
			unsupported++
		}
	}
	return float64(unsupported) / float64(len(checks))
}
