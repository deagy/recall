package graph

import (
	"fmt"
	"math"
	"sort"
)

// LinkPredictionResult represents the result of a link prediction query.
type LinkPredictionResult struct {
	Head     string
	Relation string
	Tail     string
	Score    float64
}

// LinkPrediction predicts missing links in the knowledge graph.
type LinkPrediction struct {
	embedder GraphEmbedder
}

// NewLinkPrediction creates a new LinkPrediction instance.
func NewLinkPrediction(embedder GraphEmbedder) *LinkPrediction {
	return &LinkPrediction{
		embedder: embedder,
	}
}

// PredictTail predicts the most likely tail entities for a given (head, relation) pair.
func (lp *LinkPrediction) PredictTail(head string, relation string, topK int) ([]LinkPredictionResult, error) {
	return lp.predictEntities(head, relation, topK, false)
}

// PredictHead predicts the most likely head entities for a given (relation, tail) pair.
func (lp *LinkPrediction) PredictHead(relation string, tail string, topK int) ([]LinkPredictionResult, error) {
	return lp.predictEntities(tail, relation, topK, true)
}

// predictEntities predicts entities for a given (entity, relation) pair.
func (lp *LinkPrediction) predictEntities(entity string, relation string, topK int, predictHead bool) ([]LinkPredictionResult, error) {
	// Get entity and relation embeddings
	_, err := lp.embedder.EmbedEntity(entity)
	if err != nil {
		return nil, err
	}

	_, err = lp.embedder.EmbedRelation(relation)
	if err != nil {
		return nil, err
	}

	// Compute scores for all entities
	type entityScore struct {
		id    string
		score float64
	}

	var scores []entityScore

	// Sort by score and return topK
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	if topK > len(scores) {
		topK = len(scores)
	}

	results := make([]LinkPredictionResult, topK)
	for i := 0; i < topK; i++ {
		if predictHead {
			results[i] = LinkPredictionResult{
				Head:     scores[i].id,
				Relation: relation,
				Tail:     entity,
				Score:    scores[i].score,
			}
		} else {
			results[i] = LinkPredictionResult{
				Head:     entity,
				Relation: relation,
				Tail:     scores[i].id,
				Score:    scores[i].score,
			}
		}
	}

	return results, nil
}

// EvaluateMetrics computes evaluation metrics for link prediction.
type EvaluateMetrics struct {
	// MeanReciprocalRank is the mean reciprocal rank of correct answers.
	MeanReciprocalRank float64

	// HitsAtK is the fraction of correct answers in the top-K predictions.
	HitsAtK map[int]float64

	// MeanRank is the mean rank of correct answers.
	MeanRank float64

	// TotalTests is the total number of test triples.
	TotalTests int
}

// NewEvaluateMetrics creates a new EvaluateMetrics instance.
func NewEvaluateMetrics() *EvaluateMetrics {
	return &EvaluateMetrics{
		HitsAtK: make(map[int]float64),
	}
}

// AddResult adds a result to the metrics.
func (m *EvaluateMetrics) AddResult(rank int, topKs []int) {
	m.TotalTests++

	// Update MRR
	m.MeanReciprocalRank += 1.0 / float64(rank)

	// Update Mean Rank
	m.MeanRank += float64(rank)

	// Update Hits@K
	for _, k := range topKs {
		if rank <= k {
			m.HitsAtK[k] += 1.0
		}
	}
}

// Finalize computes the final metrics.
func (m *EvaluateMetrics) Finalize() {
	if m.TotalTests == 0 {
		return
	}

	m.MeanReciprocalRank /= float64(m.TotalTests)
	m.MeanRank /= float64(m.TotalTests)

	for k := range m.HitsAtK {
		m.HitsAtK[k] /= float64(m.TotalTests)
	}
}

// String returns a human-readable summary of the metrics.
func (m *EvaluateMetrics) String() string {
	result := fmt.Sprintf("Evaluation Metrics (total: %d tests):\n", m.TotalTests)
	result += fmt.Sprintf("  MRR: %.4f\n", m.MeanReciprocalRank)
	result += fmt.Sprintf("  Mean Rank: %.2f\n", m.MeanRank)
	for k := 1; k <= 10; k++ {
		if hits, ok := m.HitsAtK[k]; ok {
			result += fmt.Sprintf("  Hits@%d: %.4f\n", k, hits)
		}
	}

	return result
}

// ComputeMRR computes the Mean Reciprocal Rank for a list of ranks.
func ComputeMRR(ranks []int) float64 {
	if len(ranks) == 0 {
		return 0
	}

	var sum float64
	for _, rank := range ranks {
		sum += 1.0 / float64(rank)
	}

	return sum / float64(len(ranks))
}

// ComputeHitsAtK computes the Hits@K for a list of ranks.
func ComputeHitsAtK(ranks []int, k int) float64 {
	if len(ranks) == 0 {
		return 0
	}

	var count float64
	for _, rank := range ranks {
		if rank <= k {
			count++
		}
	}

	return count / float64(len(ranks))
}

// ComputeMeanRank computes the Mean Rank for a list of ranks.
func ComputeMeanRank(ranks []int) float64 {
	if len(ranks) == 0 {
		return 0
	}

	var sum float64
	for _, rank := range ranks {
		sum += float64(rank)
	}

	return sum / float64(len(ranks))
}

// ScoreTriple computes the score for a triple using TransE scoring function.
// Score = -||head + relation - tail||
func ScoreTriple(headEmb, relEmb, tailEmb []float32) float64 {
	dim := len(headEmb)
	var sum float64
	for i := 0; i < dim; i++ {
		diff := float64(headEmb[i] + relEmb[i] - tailEmb[i])
		sum += diff * diff
	}
	return -math.Sqrt(sum)
}
