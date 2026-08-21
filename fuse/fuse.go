// Package fuse implements score fusion methods to combine rankings
// from multiple retrieval methods (e.g., vector similarity + BM25).
package fuse

import (
	"sort"
)

// Fusion defines the interface for combining scores from multiple ranking methods.
type Fusion interface {
	Fuse(scores ...map[string]float64) map[string]float64
}

// WeightedFusion combines scores using a weighted sum.
// Alpha is the weight for the first score set; (1-alpha) is the combined
// weight of the remaining score sets, split evenly across them.
type WeightedFusion struct {
	Alpha float64
}

// NewWeightedFusion creates a new WeightedFusion. Alpha should be in [0, 1].
func NewWeightedFusion(alpha float64) *WeightedFusion {
	return &WeightedFusion{Alpha: alpha}
}

// Fuse combines scores using a weighted sum. With two score maps the result
// is alpha*s1 + (1-alpha)*s2. With N > 2 maps, alpha is the weight of the
// first map and (1-alpha) is split evenly across the remaining N-1 maps, so
// the per-map weights always sum to 1.
func (f *WeightedFusion) Fuse(scores ...map[string]float64) map[string]float64 {
	result := make(map[string]float64)
	if len(scores) == 0 {
		return result
	}
	if len(scores) == 1 {
		for k, v := range scores[0] {
			result[k] = v
		}
		return result
	}

	// Collect all docIDs
	allIDs := make(map[string]bool)
	for _, s := range scores {
		for id := range s {
			allIDs[id] = true
		}
	}

	restWeight := (1.0 - f.Alpha) / float64(len(scores)-1)
	for id := range allIDs {
		var sum float64
		for i, s := range scores {
			var weight float64
			if i == 0 {
				weight = f.Alpha
			} else {
				weight = restWeight
			}
			if s != nil {
				sum += weight * s[id]
			}
		}
		if sum > 0 {
			result[id] = sum
		}
	}
	return result
}

// RRFFusion implements Reciprocal Rank Fusion.
// Score = Σ 1 / (k + rank_i) where rank starts at 1.
type RRFFusion struct {
	K int
}

// NewRRFFusion creates a new RRFFusion with the given constant k (default 60).
func NewRRFFusion(k int) *RRFFusion {
	if k <= 0 {
		k = 60
	}
	return &RRFFusion{K: k}
}

// Fuse combines scores using Reciprocal Rank Fusion.
func (f *RRFFusion) Fuse(scores ...map[string]float64) map[string]float64 {
	result := make(map[string]float64)
	if len(scores) == 0 {
		return result
	}

	for _, scoreMap := range scores {
		if len(scoreMap) == 0 {
			continue
		}

		// Sort by score descending to determine ranks
		type entry struct {
			id    string
			score float64
		}
		entries := make([]entry, 0, len(scoreMap))
		for id, score := range scoreMap {
			entries = append(entries, entry{id, score})
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].score > entries[j].score
		})

		// Assign ranks and accumulate RRF scores
		for rank, e := range entries {
			result[e.id] += 1.0 / float64(f.K+rank+1)
		}
	}

	return result
}
