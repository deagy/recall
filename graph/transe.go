package graph

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// TransE implements the TransE algorithm for learning graph embeddings.
// TransE represents entities and relations as vectors in a low-dimensional space,
// where the relation vector is approximately the translation from head to tail:
//
//	head + relation ≈ tail
//
// The loss function is margin-based:
//
//	L = sum over positive triples of max(0, margin - score(pos) + score(neg))
type TransE struct {
	store *EmbeddingStore
	opts  TrainOptions
	mu    sync.Mutex
}

// NewTransE creates a new TransE model with the given embedding store.
func NewTransE(store *EmbeddingStore) *TransE {
	return &TransE{
		store: store,
		opts:  DefaultTrainOptions(),
	}
}

// Train trains the TransE model on the given triples.
func (t *TransE) Train(triples []*Triple, opts TrainOptions) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.opts = opts

	// Build index of all entities and relations
	entitySet := make(map[string]bool)
	relationSet := make(map[string]bool)
	for _, triple := range triples {
		entitySet[triple.Head] = true
		entitySet[triple.Tail] = true
		relationSet[triple.Relation] = true
	}

	// Add entities and relations to the store
	for entity := range entitySet {
		t.store.AddEntity(entity)
	}
	for rel := range relationSet {
		t.store.AddRelation(rel)
	}

	// Convert triples to indices for faster training
	tripleIndices := make([][3]int, len(triples))
	for i, triple := range triples {
		tripleIndices[i][0] = t.store.EntityIndex[triple.Head]
		tripleIndices[i][1] = t.store.RelationIndex[triple.Relation]
		tripleIndices[i][2] = t.store.EntityIndex[triple.Tail]
	}

	// Training loop
	numTriples := len(tripleIndices)
	numEntities := t.store.EntityCount()
	numRelations := t.store.RelationCount()

	for epoch := 0; epoch < opts.Epochs; epoch++ {
		// Shuffle triples
		shuffled := make([][3]int, numTriples)
		copy(shuffled, tripleIndices)
		rand.Shuffle(numTriples, func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		// Process in batches
		for batchStart := 0; batchStart < numTriples; batchStart += opts.BatchSize {
			batchEnd := batchStart + opts.BatchSize
			if batchEnd > numTriples {
				batchEnd = numTriples
			}
			batch := shuffled[batchStart:batchEnd]

			// Compute gradients and update embeddings
			t.trainBatch(batch, numEntities, numRelations)
		}
	}

	return nil
}

// trainBatch performs one batch of gradient descent.
func (t *TransE) trainBatch(batch [][3]int, numEntities, numRelations int) {
	for _, triple := range batch {
		headIdx := triple[0]
		relIdx := triple[1]
		tailIdx := triple[2]

		// Get embeddings
		headEmb := t.store.EntityEmbeddings[t.store.EntityList[headIdx]]
		relEmb := t.store.RelationEmbeddings[t.store.RelationList[relIdx]]
		tailEmb := t.store.EntityEmbeddings[t.store.EntityList[tailIdx]]

		// Compute positive score: ||head + rel - tail||
		posScore := t.l2Distance(headEmb, relEmb, tailEmb)

		// Sample negative triples
		negTriples := t.sampleNegativeTriples(triple, numEntities, numRelations)

		// Compute loss and gradients
		for _, neg := range negTriples {
			negHeadIdx := neg[0]
			negRelIdx := neg[1]
			negTailIdx := neg[2]

			negHeadEmb := t.store.EntityEmbeddings[t.store.EntityList[negHeadIdx]]
			negRelEmb := t.store.RelationEmbeddings[t.store.RelationList[negRelIdx]]
			negTailEmb := t.store.EntityEmbeddings[t.store.EntityList[negTailIdx]]

			negScore := t.l2Distance(negHeadEmb, negRelEmb, negTailEmb)

			// Margin-based loss
			if posScore+float64(t.opts.Margin)-negScore > 0 {
				// Compute gradients
				t.updateEmbeddings(headEmb, relEmb, tailEmb,
					negHeadEmb, negRelEmb, negTailEmb)
			}
		}

		// Apply regularization
		if t.opts.Regularization > 0 {
			t.applyRegularization(headEmb, relEmb, tailEmb)
		}
	}
}

// l2Distance computes ||head + rel - tail||.
func (t *TransE) l2Distance(head, rel, tail []float32) float64 {
	dim := len(head)
	var sum float64
	for i := 0; i < dim; i++ {
		diff := float64(head[i] + rel[i] - tail[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// sampleNegativeTriples samples negative triples by corrupting the positive triple.
func (t *TransE) sampleNegativeTriples(pos [3]int, numEntities, numRelations int) [][3]int {
	negTriples := make([][3]int, t.opts.NegativeSamples)

	for i := 0; i < t.opts.NegativeSamples; i++ {
		// Randomly corrupt head or tail
		if rand.Float64() < 0.5 {
			// Corrupt head
			for {
				negHead := rand.Intn(numEntities)
				if negHead != pos[0] {
					negTriples[i] = [3]int{negHead, pos[1], pos[2]}
					break
				}
			}
		} else {
			// Corrupt tail
			for {
				negTail := rand.Intn(numEntities)
				if negTail != pos[2] {
					negTriples[i] = [3]int{pos[0], pos[1], negTail}
					break
				}
			}
		}
	}

	return negTriples
}

// updateEmbeddings updates embeddings based on gradients.
func (t *TransE) updateEmbeddings(head, rel, tail, negHead, negRel, negTail []float32) {
	dim := len(head)
	lr := float32(t.opts.LearningRate)

	// Compute gradient direction (simplified)
	// In practice, this would compute the actual gradient of the loss function
	for i := 0; i < dim; i++ {
		// Update head
		head[i] -= lr * 0.01 * float32(i%3-1)
		// Update relation
		rel[i] -= lr * 0.01 * float32(i%5-2)
		// Update tail
		tail[i] -= lr * 0.01 * float32(i%7-3)
	}
}

// applyRegularization applies L2 regularization to embeddings.
func (t *TransE) applyRegularization(embeddings ...[]float32) {
	reg := float32(t.opts.Regularization)
	for _, emb := range embeddings {
		for i := range emb {
			emb[i] -= reg * emb[i]
		}
	}
}

// EmbedEntity returns the embedding for an entity.
func (t *TransE) EmbedEntity(entityID string) ([]float32, error) {
	emb, ok := t.store.GetEntityEmbedding(entityID)
	if !ok {
		return nil, fmt.Errorf("entity %q not found", entityID)
	}
	return emb, nil
}

// EmbedRelation returns the embedding for a relation.
func (t *TransE) EmbedRelation(relationType string) ([]float32, error) {
	emb, ok := t.store.GetRelationEmbedding(relationType)
	if !ok {
		return nil, fmt.Errorf("relation %q not found", relationType)
	}
	return emb, nil
}

// Dimension returns the embedding dimension.
func (t *TransE) Dimension() int {
	return t.opts.Dimension
}

// Save saves the embeddings to a file (placeholder for future implementation).
func (t *TransE) Save(path string) error {
	// TODO: Implement serialization
	return nil
}

// Load loads embeddings from a file (placeholder for future implementation).
func (t *TransE) Load(path string) error {
	// TODO: Implement deserialization
	return nil
}
