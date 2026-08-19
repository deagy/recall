// Package hitl provides human-in-the-loop primitives for RAG quality: a queue
// of chunks awaiting human review, a store of human annotations, and active
// learning that prioritizes uncertain chunks for review.
//
// The package is dependency-free and thread-safe. A web annotation UI is out of
// scope here (see the roadmap) and can be built on top of these types.
package hitl

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sort"
	"sync"
	"time"
)

// AnnotationType categorizes a human annotation.
type AnnotationType int

const (
	// AnnotationRelevance records a relevance judgment (Score in [0,1]).
	AnnotationRelevance AnnotationType = iota
	// AnnotationCorrection records corrected/replaced chunk content (Value).
	AnnotationCorrection
	// AnnotationFeedback records free-form reviewer feedback (Note/Value).
	AnnotationFeedback
)

// String returns a human-readable annotation type name.
func (t AnnotationType) String() string {
	switch t {
	case AnnotationRelevance:
		return "relevance"
	case AnnotationCorrection:
		return "correction"
	case AnnotationFeedback:
		return "feedback"
	default:
		return "unknown"
	}
}

// Annotation is a human annotation attached to a chunk.
type Annotation struct {
	// ID is a unique identifier for this annotation.
	ID string

	// ChunkID is the chunk this annotation refers to.
	ChunkID string

	// Type categorizes the annotation.
	Type AnnotationType

	// Value holds the annotation payload (corrected text, label, etc.).
	Value string

	// Score is an optional numeric value, e.g. relevance in [0,1].
	Score float64

	// Author identifies who created the annotation.
	Author string

	// Note is an optional free-form comment.
	Note string

	// CreatedAt is when the annotation was created.
	CreatedAt time.Time
}

// NewAnnotation creates an annotation with a generated ID and timestamp.
func NewAnnotation(chunkID string, t AnnotationType, value string) *Annotation {
	return &Annotation{
		ID:        newHitlID(),
		ChunkID:   chunkID,
		Type:      t,
		Value:     value,
		CreatedAt: time.Now().UTC(),
	}
}

// AnnotationStore is a thread-safe store of human annotations indexed by chunk
// and by annotation ID. Annotations are stored by value (defensive copy), so
// mutating the caller's struct after Add has no effect on the store.
type AnnotationStore struct {
	mu      sync.RWMutex
	byID    map[string]Annotation
	byChunk map[string][]Annotation
}

// NewAnnotationStore creates an empty AnnotationStore.
func NewAnnotationStore() *AnnotationStore {
	return &AnnotationStore{
		byID:    make(map[string]Annotation),
		byChunk: make(map[string][]Annotation),
	}
}

// Add stores a copy of the annotation. If the ID already exists the existing
// entry (and its chunk index entry, if the chunk differs) is replaced.
func (s *AnnotationStore) Add(a *Annotation) {
	if a == nil {
		return
	}
	if a.ID == "" {
		a.ID = newHitlID()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if old, ok := s.byID[a.ID]; ok && old.ChunkID != a.ChunkID {
		s.byChunk[old.ChunkID] = removeAnnotation(s.byChunk[old.ChunkID], a.ID)
	}
	s.byID[a.ID] = *a
	s.byChunk[a.ChunkID] = append(s.byChunk[a.ChunkID], *a)
}

// Get returns a copy of an annotation by ID.
func (s *AnnotationStore) Get(id string) (Annotation, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	a, ok := s.byID[id]
	return a, ok
}

// ForChunk returns copies of all annotations for a chunk, in insertion order.
func (s *AnnotationStore) ForChunk(chunkID string) []Annotation {
	s.mu.RLock()
	defer s.mu.RUnlock()
	list := s.byChunk[chunkID]
	out := make([]Annotation, len(list))
	copy(out, list)
	return out
}

// RelevanceFor returns the most recent relevance annotation's score for a
// chunk, if any.
func (s *AnnotationStore) RelevanceFor(chunkID string) (float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var best *Annotation
	for i := range s.byChunk[chunkID] {
		a := s.byChunk[chunkID][i]
		if a.Type == AnnotationRelevance && (best == nil || !a.CreatedAt.Before(best.CreatedAt)) {
			cp := a
			best = &cp
		}
	}
	if best == nil {
		return 0, false
	}
	return best.Score, true
}

// Count returns the total number of annotations.
func (s *AnnotationStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.byID)
}

// Chunks returns the sorted set of chunk IDs that have at least one annotation.
func (s *AnnotationStore) Chunks() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.byChunk))
	for id := range s.byChunk {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func removeAnnotation(list []Annotation, id string) []Annotation {
	out := make([]Annotation, 0, len(list))
	for _, a := range list {
		if a.ID != id {
			out = append(out, a)
		}
	}
	return out
}

// newHitlID returns a short unique identifier.
func newHitlID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("a-%d", time.Now().UnixNano())
	}
	return "a-" + hex.EncodeToString(b)
}
