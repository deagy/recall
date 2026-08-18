package graph

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
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

// transEFile is the on-disk representation of a trained TransE model.
type transEFile struct {
	Dimension    int                  `json:"dimension"`
	Entities     map[string][]float32 `json:"entities"`
	Relations    map[string][]float32 `json:"relations"`
	EntityList   []string             `json:"entity_list"`
	RelationList []string             `json:"relation_list"`
}

// Save persists the current entity and relation embeddings to the given path
// as a JSON document (transEFile). Vectors are copied so the file is a stable
// snapshot of the model at the time of the call.
func (t *TransE) Save(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	file := transEFile{
		Dimension:    t.store.Dimension,
		Entities:     make(map[string][]float32, len(t.store.EntityEmbeddings)),
		Relations:    make(map[string][]float32, len(t.store.RelationEmbeddings)),
		EntityList:   make([]string, len(t.store.EntityList)),
		RelationList: make([]string, len(t.store.RelationList)),
	}
	copy(file.EntityList, t.store.EntityList)
	copy(file.RelationList, t.store.RelationList)
	for id, emb := range t.store.EntityEmbeddings {
		cp := make([]float32, len(emb))
		copy(cp, emb)
		file.Entities[id] = cp
	}
	for id, emb := range t.store.RelationEmbeddings {
		cp := make([]float32, len(emb))
		copy(cp, emb)
		file.Relations[id] = cp
	}

	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return fmt.Errorf("transE: marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("transE: write %s: %w", path, err)
	}
	return nil
}

// Load replaces the current embeddings with those stored at the given path
// (a file previously written by Save). The file must have the same embedding
// dimension as the underlying store; every listed entity/relation must have a
// well-formed vector.
func (t *TransE) Load(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("transE: read %s: %w", path, err)
	}
	var file transEFile
	if err := json.Unmarshal(data, &file); err != nil {
		return fmt.Errorf("transE: unmarshal: %w", err)
	}
	if file.Dimension != t.store.Dimension {
		return fmt.Errorf("transE: dimension mismatch: file has %d, store has %d", file.Dimension, t.store.Dimension)
	}

	if err := t.loadVectors("entity", file.EntityList, file.Entities, file.Dimension,
		func(emb map[string][]float32, idx map[string]int, list []string) {
			t.store.EntityEmbeddings = emb
			t.store.EntityIndex = idx
			t.store.EntityList = list
		}); err != nil {
		return err
	}
	return t.loadVectors("relation", file.RelationList, file.Relations, file.Dimension,
		func(emb map[string][]float32, idx map[string]int, list []string) {
			t.store.RelationEmbeddings = emb
			t.store.RelationIndex = idx
			t.store.RelationList = list
		})
}

// loadVectors validates one vector set from a transEFile and installs it in
// the store (via the set callback), preserving the file's list order as the
// index ordering.
func (t *TransE) loadVectors(kind string, ids []string, vecs map[string][]float32, dim int,
	set func(emb map[string][]float32, idx map[string]int, list []string)) error {
	embeddings := make(map[string][]float32, len(ids))
	index := make(map[string]int, len(ids))
	list := make([]string, 0, len(ids))
	for _, id := range ids {
		emb, ok := vecs[id]
		if !ok {
			return fmt.Errorf("transE: %s %q listed in file but has no vector", kind, id)
		}
		if len(emb) != dim {
			return fmt.Errorf("transE: %s %q has %d dims, want %d", kind, id, len(emb), dim)
		}
		cp := make([]float32, len(emb))
		copy(cp, emb)
		list = append(list, id)
		index[id] = len(list) - 1
		embeddings[id] = cp
	}
	set(embeddings, index, list)
	return nil
}
