package store

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// MultiModalItem is a single stored artifact in a MultiModalStore:
// either text or an image, embedded in the shared multi-modal vector
// space.
type MultiModalItem struct {
	// ID uniquely identifies the item.
	ID string

	// Modality is "text" or "image".
	Modality embedder.Modality

	// Content holds the text for text items, or the caption/alt-text
	// for image items.
	Content string

	// Image holds raw image bytes (image items only).
	Image []byte

	// MimeType is the image MIME type (image items only).
	MimeType string

	// Embedding is the shared-space vector for this item.
	Embedding []float32

	// CreatedAt is when the item was stored.
	CreatedAt time.Time
}

// MultiModalResult pairs an item with its relevance score.
type MultiModalResult struct {
	// Item is the matched artifact.
	Item *MultiModalItem

	// Score is the cosine similarity (higher = more similar).
	Score float64
}

// MultiModalStore stores and retrieves text and image artifacts in a
// single shared embedding space, enabling cross-modal search (text
// queries retrieve images, image queries retrieve text, and same-modal
// search). It is a self-contained, thread-safe in-memory store.
type MultiModalStore struct {
	embedder embedder.MultiModalEmbedder

	mu    sync.RWMutex
	items map[string]*MultiModalItem

	counter int64
}

// NewMultiModalStore creates a store backed by the given
// multi-modal embedder.
func NewMultiModalStore(e embedder.MultiModalEmbedder) (*MultiModalStore, error) {
	if e == nil {
		return nil, fmt.Errorf("multimodal store: embedder is required")
	}
	return &MultiModalStore{
		embedder: e,
		items:    make(map[string]*MultiModalItem),
	}, nil
}

// Dimension returns the shared embedding dimension.
func (s *MultiModalStore) Dimension() int { return s.embedder.Dimension() }

// AddText stores a text item. An empty ID is auto-generated.
func (s *MultiModalStore) AddText(ctx context.Context, id, text string) error {
	if text == "" {
		return core.ErrInvalidDocument
	}
	emb, err := s.embedder.EmbedText(ctx, text)
	if err != nil {
		return fmt.Errorf("embedding text: %w", err)
	}
	return s.store(id, &MultiModalItem{
		Modality:  embedder.ModalityText,
		Content:   text,
		Embedding: emb,
		CreatedAt: time.Now(),
	})
}

// AddImage stores an image item. An empty ID is auto-generated;
// caption may be empty.
func (s *MultiModalStore) AddImage(ctx context.Context, id string, data []byte, mimeType, caption string) error {
	if len(data) == 0 {
		return core.ErrInvalidDocument
	}
	emb, err := s.embedder.EmbedImage(ctx, data, mimeType)
	if err != nil {
		return fmt.Errorf("embedding image: %w", err)
	}
	img := make([]byte, len(data))
	copy(img, data)
	return s.store(id, &MultiModalItem{
		Modality:  embedder.ModalityImage,
		Content:   caption,
		Image:     img,
		MimeType:  mimeType,
		Embedding: emb,
		CreatedAt: time.Now(),
	})
}

// store assigns an ID when needed and inserts the item.
func (s *MultiModalStore) store(id string, item *MultiModalItem) error {
	if id == "" {
		n := atomic.AddInt64(&s.counter, 1)
		id = fmt.Sprintf("mm-%d", n)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.items[id]; exists {
		return core.ErrDuplicate
	}
	item.ID = id
	s.items[id] = item
	return nil
}

// Get returns an item by ID.
func (s *MultiModalStore) Get(id string) (*MultiModalItem, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	it, ok := s.items[id]
	return it, ok
}

// Delete removes an item.
func (s *MultiModalStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.items[id]; !ok {
		return core.ErrNotFound
	}
	delete(s.items, id)
	return nil
}

// Count returns the number of stored items.
func (s *MultiModalStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.items)
}

// Close releases resources (in-memory store: no-op, kept for
// interface symmetry).
func (s *MultiModalStore) Close() error { return nil }

// SearchText performs a cross-modal search with a text query: both
// text and image items are scored against the query embedding and the
// top K are returned.
func (s *MultiModalStore) SearchText(ctx context.Context, query string, topK int) ([]MultiModalResult, error) {
	if query == "" {
		return nil, core.ErrEmptyQuery
	}
	qemb, err := s.embedder.EmbedText(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	return s.search(qemb, topK)
}

// SearchImage performs a cross-modal search with an image query.
func (s *MultiModalStore) SearchImage(ctx context.Context, data []byte, mimeType string, topK int) ([]MultiModalResult, error) {
	if len(data) == 0 {
		return nil, core.ErrEmptyQuery
	}
	qemb, err := s.embedder.EmbedImage(ctx, data, mimeType)
	if err != nil {
		return nil, fmt.Errorf("embedding image query: %w", err)
	}
	return s.search(qemb, topK)
}

// search scores all items against a query embedding and returns the
// top K, sorted by score descending (ID as tie-break).
func (s *MultiModalStore) search(queryEmb []float32, topK int) ([]MultiModalResult, error) {
	if topK <= 0 {
		topK = 10
	}
	type scored struct {
		item  *MultiModalItem
		score float64
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	var results []scored
	for _, it := range s.items {
		results = append(results, scored{it, embedder.CosineSimilarity(queryEmb, it.Embedding)})
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].score != results[j].score {
			return results[i].score > results[j].score
		}
		return results[i].item.ID < results[j].item.ID
	})
	if len(results) > topK {
		results = results[:topK]
	}
	out := make([]MultiModalResult, len(results))
	for i, r := range results {
		out[i] = MultiModalResult{Item: r.item, Score: r.score}
	}
	return out, nil
}
