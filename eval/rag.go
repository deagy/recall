package eval

import (
	"context"
	"fmt"
	"strings"
	"unicode"
)

// JudgeInput is the material a Judge scores for answer quality.
type JudgeInput struct {
	// Question is the user's question.
	Question string

	// Context is the retrieved context the answer was grounded on.
	Context string

	// Answer is the generated answer to evaluate.
	Answer string

	// Reference is an optional ground-truth answer for correctness scoring.
	Reference string
}

// AnswerQuality holds the 0-1 answer-quality scores.
type AnswerQuality struct {
	// Faithfulness measures how much of the answer is supported by the context.
	Faithfulness float64

	// Relevance measures how much of the question the answer addresses.
	Relevance float64

	// Correctness measures agreement with the reference answer (0 when no
	// reference is provided).
	Correctness float64

	// Comment is an optional explanation from the judge.
	Comment string
}

// Judge scores an answer against a question, context, and optional reference.
// A deterministic lexical judge (OverlapJudge) is provided; an LLM-based judge
// can be plugged in by implementing this interface.
type Judge interface {
	// Judge returns answer-quality scores for the input.
	Judge(ctx context.Context, in JudgeInput) (AnswerQuality, error)
}

// RAGEval evaluates the quality of RAG answers using a pluggable Judge.
type RAGEval struct {
	// Judge scores answers.
	Judge Judge
}

// NewRAGEval creates a RAGEval with the given judge.
func NewRAGEval(j Judge) *RAGEval { return &RAGEval{Judge: j} }

// errNilJudge is returned when a RAGEval is used without a Judge.
func errNilJudge() error { return fmt.Errorf("eval: RAGEval requires a Judge") }

// Evaluate scores a single (question, context, answer) triple.
func (e *RAGEval) Evaluate(ctx context.Context, in JudgeInput) (AnswerQuality, error) {
	if e.Judge == nil {
		return AnswerQuality{}, errNilJudge()
	}
	return e.Judge.Judge(ctx, in)
}

// OverlapJudge is a deterministic, dependency-free Judge that approximates
// answer quality using lexical overlap:
//
//   - Faithfulness: fraction of the answer's content words that appear in the
//     context (claims supported by the context).
//   - Relevance: fraction of the question's content words that appear in the
//     answer.
//   - Correctness: token F1 between the answer and the reference answer (0
//     when no reference is provided).
//
// It is intended as a fast, reproducible baseline and for tests; plug in an
// LLM Judge for semantic judgment.
type OverlapJudge struct{}

// NewOverlapJudge creates a lexical OverlapJudge.
func NewOverlapJudge() *OverlapJudge { return &OverlapJudge{} }

// Judge implements Judge using lexical overlap.
func (j *OverlapJudge) Judge(ctx context.Context, in JudgeInput) (AnswerQuality, error) {
	q := AnswerQuality{}

	ans := uniqueContentWords(in.Answer)
	ctxWords := uniqueContentWords(in.Context)
	ques := uniqueContentWords(in.Question)
	ref := contentWords(in.Reference)

	// Faithfulness: answer content words supported by the context.
	if len(ans) > 0 {
		hits := 0
		for w := range ans {
			if _, ok := ctxWords[w]; ok {
				hits++
			}
		}
		q.Faithfulness = float64(hits) / float64(len(ans))
	}

	// Relevance: question content words addressed by the answer.
	if len(ques) > 0 {
		ansSet := uniqueContentWords(in.Answer)
		hits := 0
		for w := range ques {
			if _, ok := ansSet[w]; ok {
				hits++
			}
		}
		q.Relevance = float64(hits) / float64(len(ques))
	}

	// Correctness: token F1 against the reference answer.
	if len(ref) > 0 {
		q.Correctness = tokenF1(contentWords(in.Answer), ref)
	}

	return q, nil
}

// contentWords returns lowercased, stopword-filtered tokens (with duplicates),
// dropping tokens shorter than two characters.
func contentWords(text string) []string {
	var out []string
	for _, tok := range contentTokens(text) {
		if len(tok) < 2 {
			continue
		}
		out = append(out, tok)
	}
	return out
}

// uniqueContentWords returns the set of unique content words for a text.
func uniqueContentWords(text string) map[string]struct{} {
	s := make(map[string]struct{})
	for _, w := range contentWords(text) {
		s[w] = struct{}{}
	}
	return s
}

// tokenF1 returns the harmonic mean of precision and recall of a against b,
// counting duplicates. Returns 0 when either side is empty.
func tokenF1(a, b []string) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	countA := make(map[string]int, len(a))
	for _, w := range a {
		countA[w]++
	}
	common := 0
	for _, w := range b {
		if countA[w] > 0 {
			countA[w]--
			common++
		}
	}
	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(a))
	recall := float64(common) / float64(len(b))
	return 2 * precision * recall / (precision + recall)
}

// contentTokens splits text into lowercased alphanumeric tokens, dropping
// stopwords.
func contentTokens(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			t := cur.String()
			cur.Reset()
			if _, stop := evalStopWords[t]; stop {
				return
			}
			out = append(out, t)
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(unicode.ToLower(r))
		} else {
			flush()
		}
	}
	flush()
	return out
}

// evalStopWords is a minimal English stopword set for lexical evaluation.
var evalStopWords = map[string]struct{}{
	"a": {}, "an": {}, "the": {}, "and": {}, "or": {}, "but": {}, "if": {},
	"then": {}, "else": {}, "when": {}, "at": {}, "by": {}, "for": {},
	"with": {}, "about": {}, "between": {}, "into": {}, "through": {},
	"during": {}, "before": {}, "after": {}, "above": {}, "below": {},
	"to": {}, "from": {}, "up": {}, "down": {}, "in": {}, "out": {},
	"on": {}, "off": {}, "over": {}, "under": {}, "again": {}, "once": {},
	"here": {}, "there": {}, "all": {}, "any": {}, "both": {}, "each": {},
	"few": {}, "more": {}, "most": {}, "other": {}, "some": {}, "such": {},
	"no": {}, "nor": {}, "not": {}, "only": {}, "own": {}, "same": {},
	"so": {}, "than": {}, "too": {}, "very": {}, "can": {}, "will": {},
	"just": {}, "should": {}, "now": {}, "is": {}, "are": {}, "was": {},
	"were": {}, "be": {}, "been": {}, "being": {}, "have": {}, "has": {},
	"had": {}, "having": {}, "do": {}, "does": {}, "did": {}, "doing": {},
	"would": {}, "could": {}, "of": {}, "as": {}, "it": {}, "its": {},
	"this": {}, "that": {}, "these": {}, "those": {}, "i": {}, "you": {},
	"he": {}, "she": {}, "we": {}, "they": {}, "what": {}, "which": {},
	"who": {}, "whom": {}, "why": {}, "how": {},
}
