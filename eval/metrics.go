// Package eval provides evaluation metrics and a benchmark suite for measuring
// and regression-testing retrieval and RAG quality.
//
// The package is dependency-free. Retrieval metrics (Precision@K, Recall@K,
// MRR, NDCG@K) are pure functions over ranked ID lists and ground truth, and
// answer-quality metrics use a pluggable Judge (a deterministic lexical judge
// is included; an LLM judge can be plugged in via the Judge interface).
package eval

import (
	"context"
	"math"
)

// Retriever retrieves ranked chunk IDs for a query. It is intentionally small
// so it can be satisfied by a store, an index, or a thin adapter in tests.
type Retriever interface {
	// Retrieve returns up to k chunk IDs ranked by relevance for the query.
	Retrieve(ctx context.Context, query string, k int) ([]string, error)
}

// RetrievalMetrics holds the computed metrics for a single query.
type RetrievalMetrics struct {
	// Query is the evaluated query text.
	Query string

	// K is the cutoff used for the @K metrics.
	K int

	// Precision is Precision@K (fraction of top-K results that are relevant).
	Precision float64

	// Recall is Recall@K (fraction of relevant items retrieved in top-K).
	Recall float64

	// MRR is the reciprocal rank of the first relevant result (1/rank, 0 if none).
	MRR float64

	// NDCG is NDCG@K using graded (or binary) relevance.
	NDCG float64

	// NumRelevant is the total number of ground-truth relevant items.
	NumRelevant int

	// NumRetrieved is the number of results actually returned.
	NumRetrieved int
}

// PrecisionAtK returns the fraction of the top-K retrieved items that are in
// the relevant set. Returns 0 when k <= 0.
func PrecisionAtK(retrieved []string, relevant map[string]bool, k int) float64 {
	if k <= 0 {
		return 0
	}
	n := k
	if n > len(retrieved) {
		n = len(retrieved)
	}
	hits := 0
	for i := 0; i < n; i++ {
		if relevant[retrieved[i]] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

// RecallAtK returns the fraction of all relevant items that appear in the
// top-K retrieved items. Returns 0 when there are no relevant items.
func RecallAtK(retrieved []string, relevant map[string]bool, k int) float64 {
	total := 0
	for _, ok := range relevant {
		if ok {
			total++
		}
	}
	if total == 0 {
		return 0
	}
	n := k
	if n > len(retrieved) {
		n = len(retrieved)
	}
	hits := 0
	seen := make(map[string]struct{}, n)
	for i := 0; i < n; i++ {
		id := retrieved[i]
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		if relevant[id] {
			hits++
		}
	}
	return float64(hits) / float64(total)
}

// MRR returns the reciprocal rank of the first relevant result in the
// retrieved list (1.0 if the first result is relevant, 0.5 if the second,
// ...). Returns 0 when no result is relevant. The full list is considered.
func MRR(retrieved []string, relevant map[string]bool) float64 {
	for i, id := range retrieved {
		if relevant[id] {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// NDCGAtK returns the normalized discounted cumulative gain at cutoff K.
// Relevance grades come from the relevance map (grade 0 when absent). When
// there is no graded relevance (all grades <= 0), the result is 0.
func NDCGAtK(retrieved []string, relevance map[string]int, k int) float64 {
	if k <= 0 {
		return 0
	}
	n := k
	if n > len(retrieved) {
		n = len(retrieved)
	}

	var dcg float64
	for i := 0; i < n; i++ {
		rel := float64(relevance[retrieved[i]])
		if rel <= 0 {
			continue
		}
		dcg += (math.Pow(2, rel) - 1) / math.Log2(float64(i+2))
	}

	// Ideal DCG: sort all positive grades descending, take top-K.
	grades := make([]int, 0, len(relevance))
	for _, g := range relevance {
		if g > 0 {
			grades = append(grades, g)
		}
	}
	// sort grades descending
	for i := 1; i < len(grades); i++ {
		for j := i; j > 0 && grades[j] > grades[j-1]; j-- {
			grades[j], grades[j-1] = grades[j-1], grades[j]
		}
	}
	idealN := k
	if idealN > len(grades) {
		idealN = len(grades)
	}
	var idcg float64
	for i := 0; i < idealN; i++ {
		rel := float64(grades[i])
		idcg += (math.Pow(2, rel) - 1) / math.Log2(float64(i+2))
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

// ComputeRetrievalMetrics computes Precision@K, Recall@K, MRR, and NDCG@K for a
// single query given its ranked results and ground-truth relevance.
//
// If gradedRelevance is non-nil it is used for NDCG (and as the relevance set
// for the other metrics, treating grade > 0 as relevant). Otherwise relevantIDs
// define a binary relevance set (grade 1).
func ComputeRetrievalMetrics(query string, retrieved []string, relevantIDs []string, gradedRelevance map[string]int, k int) RetrievalMetrics {
	relevanceSet := make(map[string]bool)
	if gradedRelevance != nil {
		for id, g := range gradedRelevance {
			if g > 0 {
				relevanceSet[id] = true
			}
		}
	} else {
		for _, id := range relevantIDs {
			relevanceSet[id] = true
		}
	}
	gradeMap := gradedRelevance
	if gradeMap == nil {
		gradeMap = make(map[string]int, len(relevantIDs))
		for _, id := range relevantIDs {
			gradeMap[id] = 1
		}
	}

	numRelevant := 0
	for _, ok := range relevanceSet {
		if ok {
			numRelevant++
		}
	}

	return RetrievalMetrics{
		Query:        query,
		K:            k,
		Precision:    PrecisionAtK(retrieved, relevanceSet, k),
		Recall:       RecallAtK(retrieved, relevanceSet, k),
		MRR:          MRR(retrieved, relevanceSet),
		NDCG:         NDCGAtK(retrieved, gradeMap, k),
		NumRelevant:  numRelevant,
		NumRetrieved: len(retrieved),
	}
}
