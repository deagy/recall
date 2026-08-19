package hitl

import "sort"

// Candidate is a chunk under consideration for review, with an uncertainty
// score in [0,1] (higher = more uncertain = more worth a human's time).
type Candidate struct {
	ChunkID     string
	Uncertainty float64
}

// ActiveLearning selects which chunks to prioritize for human review, enqueuing
// the most uncertain candidates first.
type ActiveLearning struct {
	// Queue is where selected chunks are enqueued for review.
	Queue *ReviewQueue

	// BatchSize is the maximum number of chunks to enqueue per selection.
	BatchSize int
}

// NewActiveLearning creates an ActiveLearning policy with the given queue and
// batch size (batch size <= 0 means no cap).
func NewActiveLearning(queue *ReviewQueue, batchSize int) *ActiveLearning {
	return &ActiveLearning{Queue: queue, BatchSize: batchSize}
}

// Select enqueues up to BatchSize of the most uncertain candidates that are not
// already pending, and returns the chunk IDs it enqueued (most uncertain first).
func (a *ActiveLearning) Select(candidates []Candidate) []string {
	if a.Queue == nil || len(candidates) == 0 {
		return nil
	}
	sorted := make([]Candidate, len(candidates))
	copy(sorted, candidates)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].Uncertainty > sorted[j].Uncertainty
	})

	var out []string
	for _, c := range sorted {
		if a.BatchSize > 0 && len(out) >= a.BatchSize {
			break
		}
		if c.ChunkID == "" {
			continue
		}
		if status, ok := a.Queue.Status(c.ChunkID); ok && status != StatusPending {
			continue // already reviewed
		}
		a.Queue.Enqueue(c.ChunkID, "active learning: high uncertainty", c.Uncertainty)
		out = append(out, c.ChunkID)
	}
	return out
}

// UncertaintyFromScores computes a per-chunk uncertainty score in [0,1] from a
// query's retrieval scores using the least-confidence criterion: a chunk is more
// uncertain the lower its (min-max normalized) score, i.e. the less the model is
// confident it is relevant. The highest-scoring chunk gets 0; the lowest gets 1.
// If all scores are equal (or there is a single score), every chunk is
// maximally uninformative and receives 0.5.
func UncertaintyFromScores(scores []float64) []float64 {
	n := len(scores)
	if n == 0 {
		return nil
	}
	out := make([]float64, n)
	if n == 1 {
		out[0] = 0.5
		return out
	}
	minS, maxS := scores[0], scores[0]
	for _, s := range scores[1:] {
		if s < minS {
			minS = s
		}
		if s > maxS {
			maxS = s
		}
	}
	if maxS == minS {
		for i := range out {
			out[i] = 0.5
		}
		return out
	}
	span := maxS - minS
	for i, s := range scores {
		normalized := (s - minS) / span
		out[i] = 1 - normalized
	}
	return out
}

// Margin returns the confidence margin between the top and second-highest
// score (0 when there are fewer than two scores). A small margin indicates the
// retrieval is uncertain about the ranking and the top result deserves a human
// check. The input need not be sorted.
func Margin(scores []float64) float64 {
	if len(scores) < 2 {
		return 0
	}
	var top, second float64
	if scores[0] >= scores[1] {
		top, second = scores[0], scores[1]
	} else {
		top, second = scores[1], scores[0]
	}
	for _, s := range scores[2:] {
		if s > top {
			second = top
			top = s
		} else if s > second {
			second = s
		}
	}
	if m := top - second; m < 0 {
		return 0
	} else {
		return m
	}
}
