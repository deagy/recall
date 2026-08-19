package feedback

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// RocchioParams configures the classic Rocchio query expansion algorithm.
//
//	Rocchio:  Q' = Alpha*Q + Beta*mean(relevant) - Gamma*mean(irrelevant)
//
// Defaults (via DefaultRocchioParams): Alpha=1, Beta=0.5, Gamma=0.3,
// Normalize=true.
type RocchioParams struct {
	// Alpha is the weight of the original query vector. 0 keeps the query from
	// contributing.
	Alpha float64

	// Beta is the weight of the mean relevant-document vector. 0 disables the
	// positive shift.
	Beta float64

	// Gamma is the weight of the mean not-relevant-document vector (subtracted).
	// 0 disables the negative shift.
	Gamma float64

	// Normalize L2-normalizes the resulting vector so it stays comparable with
	// stored embeddings under cosine similarity.
	Normalize bool
}

// DefaultRocchioParams returns the standard Rocchio weights.
func DefaultRocchioParams() RocchioParams {
	return RocchioParams{Alpha: 1, Beta: 0.5, Gamma: 0.3, Normalize: true}
}

// Rocchio applies the Rocchio algorithm in embedding space. It returns a new
// query vector shifted toward the centroid of relevant embeddings and away
// from the centroid of not-relevant embeddings.
//
// All vectors are expected to have the same dimension as query. If relevant or
// irrelevant is empty, that term is simply omitted. If the input query is
// empty, the dimension is inferred from the feedback vectors; if none are
// supplied, an empty slice is returned.
func Rocchio(query []float32, relevant, irrelevant [][]float32, p RocchioParams) []float32 {
	dim := len(query)
	if dim == 0 {
		if len(relevant) > 0 {
			dim = len(relevant[0])
		} else if len(irrelevant) > 0 {
			dim = len(irrelevant[0])
		}
	}
	if dim == 0 {
		return []float32{}
	}

	out := make([]float32, dim)
	if p.Alpha != 0 && len(query) == dim {
		for i, v := range query {
			out[i] += float32(p.Alpha) * v
		}
	}
	if p.Beta != 0 && len(relevant) > 0 {
		mean := MeanVectors(relevant)
		for i := 0; i < dim && i < len(mean); i++ {
			out[i] += float32(p.Beta) * mean[i]
		}
	}
	if p.Gamma != 0 && len(irrelevant) > 0 {
		mean := MeanVectors(irrelevant)
		for i := 0; i < dim && i < len(mean); i++ {
			out[i] -= float32(p.Gamma) * mean[i]
		}
	}
	if p.Normalize {
		out = L2Normalize(out)
	}
	return out
}

// MeanVectors returns the element-wise mean of a set of vectors. Vectors of
// differing lengths are handled using the length of the first vector; shorter
// vectors contribute zero for their missing dimensions. Empty input returns an
// empty slice.
func MeanVectors(vectors [][]float32) []float32 {
	if len(vectors) == 0 {
		return []float32{}
	}
	dim := len(vectors[0])
	sum := make([]float64, dim)
	for _, v := range vectors {
		for i := 0; i < dim && i < len(v); i++ {
			sum[i] += float64(v[i])
		}
	}
	out := make([]float32, dim)
	n := float64(len(vectors))
	for i := range sum {
		out[i] = float32(sum[i] / n)
	}
	return out
}

// L2Normalize returns a unit-length copy of v. A zero vector is returned as-is.
func L2Normalize(v []float32) []float32 {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return v
	}
	norm = math.Sqrt(norm)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}

// CosineSimilarity returns the cosine similarity between two vectors. The
// shorter vector is zero-padded. Returns 0 for any zero vector.
func CosineSimilarity(a, b []float32) float64 {
	dim := len(a)
	if len(b) < dim {
		dim = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < dim; i++ {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TermRocchioParams configures lexical (term-frequency) Rocchio expansion.
type TermRocchioParams struct {
	// Beta is the weight of terms from relevant documents.
	Beta float64

	// Gamma is the weight of terms from not-relevant documents (subtracted).
	Gamma float64

	// MaxTerms limits how many terms are kept in the expanded query. 0 keeps
	// every term with a positive weight.
	MaxTerms int
}

// DefaultTermRocchioParams returns standard lexical Rocchio weights.
func DefaultTermRocchioParams() TermRocchioParams {
	return TermRocchioParams{Beta: 0.5, Gamma: 0.3, MaxTerms: 12}
}

// RocchioTerms applies the Rocchio algorithm in term space. It returns an
// expanded query string (top terms by adjusted weight) plus the full
// term-weight map for inspection.
//
// The adjusted weight of a term t is:
//
//	w(t) = freq_query(t) + Beta*meanFreq_relevant(t) - Gamma*meanFreq_irrelevant(t)
//
// Negative weights are clamped to zero and zero-weight terms are dropped.
func RocchioTerms(query string, relevantTexts, irrelevantTexts []string, p TermRocchioParams) (string, map[string]float64) {
	qw := termWeights(query)
	relMean := meanTermWeights(relevantTexts)
	irrMean := meanTermWeights(irrelevantTexts)

	terms := make(map[string]struct{})
	for t := range qw {
		terms[t] = struct{}{}
	}
	for t := range relMean {
		terms[t] = struct{}{}
	}
	for t := range irrMean {
		terms[t] = struct{}{}
	}

	adjusted := make(map[string]float64, len(terms))
	for t := range terms {
		w := qw[t] + p.Beta*relMean[t] - p.Gamma*irrMean[t]
		if w < 0 {
			w = 0
		}
		if w > 0 {
			adjusted[t] = w
		}
	}

	top := topTerms(adjusted, p.MaxTerms)
	return strings.Join(top, " "), adjusted
}

// termWeights returns lowercased term frequencies for a text, excluding a small
// stopword set.
func termWeights(text string) map[string]float64 {
	w := make(map[string]float64)
	for _, tok := range tokenize(text) {
		if _, stop := stopWords[tok]; stop {
			continue
		}
		w[tok]++
	}
	return w
}

// meanTermWeights returns the average term frequency across a set of texts.
func meanTermWeights(texts []string) map[string]float64 {
	if len(texts) == 0 {
		return map[string]float64{}
	}
	sum := make(map[string]float64)
	for _, t := range texts {
		for term, f := range termWeights(t) {
			sum[term] += f
		}
	}
	n := float64(len(texts))
	out := make(map[string]float64, len(sum))
	for term, f := range sum {
		out[term] = f / n
	}
	return out
}

// topTerms returns term names sorted by descending weight (ties broken
// alphabetically), limited to max (0 = all).
func topTerms(weights map[string]float64, max int) []string {
	type kv struct {
		k string
		v float64
	}
	list := make([]kv, 0, len(weights))
	for k, v := range weights {
		list = append(list, kv{k, v})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].v != list[j].v {
			return list[i].v > list[j].v
		}
		return list[i].k < list[j].k
	})
	if max > 0 && len(list) > max {
		list = list[:max]
	}
	out := make([]string, len(list))
	for i, e := range list {
		out[i] = e.k
	}
	return out
}

// tokenize splits text into lowercased alphanumeric tokens.
func tokenize(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
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

// stopWords is a minimal English stopword set used by lexical Rocchio.
var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "the": {}, "and": {}, "or": {}, "but": {}, "if": {},
	"then": {}, "else": {}, "when": {}, "at": {}, "by": {}, "for": {},
	"with": {}, "about": {}, "against": {}, "between": {}, "into": {},
	"through": {}, "during": {}, "before": {}, "after": {}, "above": {},
	"below": {}, "to": {}, "from": {}, "up": {}, "down": {}, "in": {},
	"out": {}, "on": {}, "off": {}, "over": {}, "under": {}, "again": {},
	"further": {}, "once": {}, "here": {}, "there": {}, "all": {}, "any": {},
	"both": {}, "each": {}, "few": {}, "more": {}, "most": {}, "other": {},
	"some": {}, "such": {}, "no": {}, "nor": {}, "not": {}, "only": {},
	"own": {}, "same": {}, "so": {}, "than": {}, "too": {}, "very": {},
	"can": {}, "will": {}, "just": {}, "should": {}, "now": {}, "is": {},
	"are": {}, "was": {}, "were": {}, "be": {}, "been": {}, "being": {},
	"have": {}, "has": {}, "had": {}, "having": {}, "do": {}, "does": {},
	"did": {}, "doing": {}, "would": {}, "could": {}, "of": {}, "as": {},
	"it": {}, "its": {}, "this": {}, "that": {}, "these": {}, "those": {},
	"i": {}, "you": {}, "he": {}, "she": {}, "we": {}, "they": {}, "what": {},
	"which": {}, "who": {}, "whom": {}, "why": {}, "how": {},
}
