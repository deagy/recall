package graph

import (
	"math"
	"sync"
)

// Triple represents a fact in the knowledge graph: (head, relation, tail).
type Triple struct {
	Head     string
	Relation string
	Tail     string
}

// TrainOptions configures the training process for graph embeddings.
type TrainOptions struct {
	// Dimension is the embedding dimension for entities and relations.
	Dimension int

	// LearningRate is the step size for gradient descent.
	LearningRate float64

	// Margin is the margin for the loss function (TransE uses this for ranking).
	Margin float64

	// Regularization is the L2 regularization strength.
	Regularization float64

	// NegativeSamples is the number of negative samples per positive triple.
	NegativeSamples int

	// Epochs is the number of training epochs.
	Epochs int

	// BatchSize is the number of triples per gradient update.
	BatchSize int
}

// DefaultTrainOptions returns TrainOptions with sensible defaults.
func DefaultTrainOptions() TrainOptions {
	return TrainOptions{
		Dimension:       64,
		LearningRate:    0.01,
		Margin:          1.0,
		Regularization:  0.0001,
		NegativeSamples: 5,
		Epochs:          100,
		BatchSize:       32,
	}
}

// GraphEmbedder defines the interface for learning and querying graph embeddings.
type GraphEmbedder interface {
	// EmbedEntity returns the embedding vector for an entity.
	EmbedEntity(entityID string) ([]float32, error)

	// EmbedRelation returns the embedding vector for a relation type.
	EmbedRelation(relationType string) ([]float32, error)

	// Dimension returns the embedding dimension.
	Dimension() int

	// Train trains the embeddings on the given triples.
	Train(triples []*Triple, opts TrainOptions) error

	// Save saves the embeddings to a file.
	Save(path string) error

	// Load loads embeddings from a file.
	Load(path string) error
}

// EmbeddingStore stores entity and relation embeddings in memory.
type EmbeddingStore struct {
	mu sync.RWMutex

	// EntityEmbeddings maps entity IDs to their embedding vectors.
	EntityEmbeddings map[string][]float32

	// RelationEmbeddings maps relation types to their embedding vectors.
	RelationEmbeddings map[string][]float32

	// Dimension is the embedding dimension.
	Dimension int

	// EntityIndex maps entity IDs to indices for fast lookup.
	EntityIndex map[string]int

	// RelationIndex maps relation types to indices for fast lookup.
	RelationIndex map[string]int

	// EntityList is the list of entity IDs in index order.
	EntityList []string

	// RelationList is the list of relation types in index order.
	RelationList []string
}

// NewEmbeddingStore creates a new empty EmbeddingStore.
func NewEmbeddingStore(dimension int) *EmbeddingStore {
	return &EmbeddingStore{
		EntityEmbeddings:   make(map[string][]float32),
		RelationEmbeddings: make(map[string][]float32),
		Dimension:          dimension,
		EntityIndex:        make(map[string]int),
		RelationIndex:      make(map[string]int),
	}
}

// AddEntity adds an entity to the store with a random embedding.
func (s *EmbeddingStore) AddEntity(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.EntityIndex[id]; exists {
		return
	}

	index := len(s.EntityList)
	s.EntityList = append(s.EntityList, id)
	s.EntityIndex[id] = index
	s.EntityEmbeddings[id] = randomVector(s.Dimension)
}

// AddRelation adds a relation to the store with a random embedding.
func (s *EmbeddingStore) AddRelation(relType string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.RelationIndex[relType]; exists {
		return
	}

	index := len(s.RelationList)
	s.RelationList = append(s.RelationList, relType)
	s.RelationIndex[relType] = index
	s.RelationEmbeddings[relType] = randomVector(s.Dimension)
}

// GetEntityEmbedding returns the embedding for an entity.
func (s *EmbeddingStore) GetEntityEmbedding(entityID string) ([]float32, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	emb, ok := s.EntityEmbeddings[entityID]
	return emb, ok
}

// GetRelationEmbedding returns the embedding for a relation.
func (s *EmbeddingStore) GetRelationEmbedding(relType string) ([]float32, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	emb, ok := s.RelationEmbeddings[relType]
	return emb, ok
}

// EntityCount returns the number of entities.
func (s *EmbeddingStore) EntityCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.EntityList)
}

// RelationCount returns the number of relations.
func (s *EmbeddingStore) RelationCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.RelationList)
}

// randomVector generates a random vector with values in [-1, 1].
func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(mathrand()-0.5) * 2.0 / float32(dim)
	}
	return vec
}

// mathrand is a wrapper around math/rand for testing purposes.
var mathrand = func() func() float64 {
	// Use a simple hash-based random number generator for reproducibility
	seed := uint64(42)
	return func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed>>33) / float64(1<<31)
	}
}()

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
