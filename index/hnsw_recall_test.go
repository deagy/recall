package index

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// clusteredEmbeddings builds n well-separated clusters in dim dimensions:
// cluster c dominates dimension c with a large value and the rest is small
// Gaussian noise, so the true top-10 of any query lies inside its cluster.
func clusteredEmbeddings(clusters, perCluster, dim int, seed int64) (embeds [][]float32, clusterOf []int) {
	rng := rand.New(rand.NewSource(seed))
	total := clusters * perCluster
	embeds = make([][]float32, total)
	clusterOf = make([]int, total)
	for i := 0; i < total; i++ {
		c := i % clusters
		clusterOf[i] = c
		emb := make([]float32, dim)
		for d := 0; d < dim; d++ {
			if d == c {
				emb[d] = 5.0 + float32(rng.NormFloat64())*0.3
			} else {
				emb[d] = float32(rng.NormFloat64()) * 0.3
			}
		}
		embeds[i] = emb
	}
	return embeds, clusterOf
}

// bruteForceTopK returns the indices of the k most similar embeddings to query.
func bruteForceTopK(query []float32, embeds [][]float32, k int) map[int]bool {
	type sc struct {
		i int
		s float64
	}
	scores := make([]sc, len(embeds))
	for i, e := range embeds {
		scores[i] = sc{i, cosineSim(query, e)}
	}
	sort.Slice(scores, func(a, b int) bool { return scores[a].s > scores[b].s })
	top := make(map[int]bool, k)
	for i := 0; i < k && i < len(scores); i++ {
		top[scores[i].i] = true
	}
	return top
}

// averageLayer0Degree returns the mean number of layer-0 connections per node
// and the count of isolated non-entry nodes.
func averageLayer0Degree(h *HNSW) (avg float64, isolated int) {
	total := 0
	for i, n := range h.nodes {
		total += len(n.connections[0])
		if i > 0 && len(n.connections[0]) == 0 {
			isolated++
		}
	}
	return float64(total) / float64(len(h.nodes)), isolated
}

// TestHNSW_Recall_Clustered verifies recall@10 stays high against brute force
// on a well-separated clustered dataset. The legacy single-neighbor graph
// collapsed searches to a handful of hops and recall dropped to near zero.
func TestHNSW_Recall_Clustered(t *testing.T) {
	const (
		clusters   = 8
		perCluster = 60
		dim        = 16
		topK       = 10
	)

	embeds, clusterOf := clusteredEmbeddings(clusters, perCluster, dim, 7)
	h := NewHNSW(dim, DefaultHNSWConfig())
	for i, emb := range embeds {
		h.Add(fmt.Sprintf("p-%d", i), emb)
	}

	// Graph density: with M0=32 every node should have multiple layer-0 links;
	// the legacy implementation left most nodes with at most one.
	avgDeg, isolated := averageLayer0Degree(h)
	assert.GreaterOrEqual(t, avgDeg, 4.0, "average layer-0 out-degree should far exceed the legacy 1-neighbor graph")
	assert.Less(t, isolated, len(embeds)/10, "at most a few nodes may remain isolated")

	// Recall@10 for a sample query from every cluster.
	hits, returns := 0, 0
	for q := 0; q < perCluster; q += 10 {
		for c := 0; c < clusters; c++ {
			i := c + q*clusters
			truth := bruteForceTopK(embeds[i], embeds, topK)

			results := h.Search(embeds[i], topK)
			require.Len(t, results, topK, "expected a full top-%d result set", topK)
			for _, id := range results {
				assert.Equalf(t, clusterOf[i], clusterOf[truthIndex(id)],
					"query point %d (cluster %d) returned off-cluster point %s", i, clusterOf[i], id)
				if truth[truthIndex(id)] {
					hits++
				}
			}
			returns += len(results)
		}
	}

	recall := float64(hits) / float64(returns)
	assert.GreaterOrEqual(t, recall, 0.8, "recall@%d should be high on clustered data, got %.2f", topK, recall)
}

// truthIndex parses the "p-<i>" ID format used by the clustered recall tests.
func truthIndex(id string) int {
	var i int
	fmt.Sscanf(id, "p-%d", &i)
	return i
}

// TestHNSW_Recall_IncrementalInserts builds half the graph, then adds the rest
// one at a time (the post-activation path from fix 1) and checks recall stays
// high for points inserted both before and after activation.
func TestHNSW_Recall_IncrementalInserts(t *testing.T) {
	const (
		clusters   = 8
		perCluster = 40
		dim        = 16
		topK       = 10
	)

	embeds, _ := clusteredEmbeddings(clusters, perCluster, dim, 21)
	half := len(embeds) / 2

	h := NewHNSW(dim, DefaultHNSWConfig())
	for i := 0; i < half; i++ {
		h.Add(fmt.Sprintf("p-%d", i), embeds[i])
	}
	for i := half; i < len(embeds); i++ {
		h.Add(fmt.Sprintf("p-%d", i), embeds[i])
	}

	hits, returns := 0, 0
	for i := 0; i < len(embeds); i += 7 {
		truth := bruteForceTopK(embeds[i], embeds, topK)
		results := h.Search(embeds[i], topK)
		require.Len(t, results, topK)
		for _, id := range results {
			if truth[truthIndex(id)] {
				hits++
			}
		}
		returns += len(results)
	}

	recall := float64(hits) / float64(returns)
	assert.GreaterOrEqual(t, recall, 0.8, "recall@%d after incremental inserts should be high, got %.2f", topK, recall)
}
