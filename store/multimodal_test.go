package store

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

func newTestMultiModalStore(t *testing.T) *MultiModalStore {
	t.Helper()
	s, err := NewMultiModalStore(embedder.NewMockMultiModal(32))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestMultiModalStore_AddAndGet(t *testing.T) {
	s := newTestMultiModalStore(t)
	ctx := context.Background()

	if _, err := NewMultiModalStore(nil); err == nil {
		t.Fatal("nil embedder should fail")
	}

	// Text items with explicit and auto IDs.
	if err := s.AddText(ctx, "t1", "solar system has eight planets"); err != nil {
		t.Fatal(err)
	}
	if err := s.AddText(ctx, "", "go channels synchronize goroutines"); err != nil {
		t.Fatal(err)
	}
	// Duplicate ID.
	if err := s.AddText(ctx, "t1", "dup"); err != core.ErrDuplicate {
		t.Fatalf("duplicate: %v", err)
	}
	// Empty text.
	if err := s.AddText(ctx, "t2", ""); err != core.ErrInvalidDocument {
		t.Fatalf("empty text: %v", err)
	}

	// Image items.
	orig := []byte("car-bytes")
	if err := s.AddImage(ctx, "i1", orig, "image/png", "a red car on the road"); err != nil {
		t.Fatal(err)
	}
	if err := s.AddImage(ctx, "i2", []byte("chart-bytes"), "image/png", ""); err != nil {
		t.Fatal(err)
	}
	if err := s.AddImage(ctx, "i3", nil, "image/png", "nope"); err != core.ErrInvalidDocument {
		t.Fatalf("empty image: %v", err)
	}

	if s.Count() != 4 {
		t.Fatalf("count = %d", s.Count())
	}
	if s.Dimension() != 32 {
		t.Fatalf("dim = %d", s.Dimension())
	}
	it, ok := s.Get("i1")
	if !ok || it.Modality != embedder.ModalityImage || len(it.Image) != len("car-bytes") || it.MimeType != "image/png" {
		t.Fatalf("Get(i1) = %+v", it)
	}
	// Image bytes are defensively copied at store time (mutating the
	// caller's original slice must not affect the stored item).
	orig[0] = 0
	if fresh, _ := s.Get("i1"); fresh.Image[0] == 0 {
		t.Fatal("image bytes were not copied")
	}

	if err := s.Delete(ctx, "t1"); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("t1"); ok {
		t.Fatal("deleted item still present")
	}
	if err := s.Delete(ctx, "t1"); err != core.ErrNotFound {
		t.Fatalf("double delete: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMultiModalStore_CrossModalSearch(t *testing.T) {
	s := newTestMultiModalStore(t)
	ctx := context.Background()
	_ = s.AddText(ctx, "t1", "solar system has eight planets")
	_ = s.AddText(ctx, "t2", "go channels synchronize goroutines")
	_ = s.AddImage(ctx, "i1", []byte("planet-img"), "image/png", "planets orbiting the sun")
	_ = s.AddImage(ctx, "i2", []byte("code-img"), "image/png", "goroutine diagram")

	// Same-modal: text query should find t1 first.
	res, err := s.SearchText(ctx, "solar system has eight planets", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].Item.ID != "t1" {
		t.Fatalf("text search = %+v", res)
	}
	if res[0].Score < 0.99 {
		t.Fatalf("self match score = %f", res[0].Score)
	}

	// Empty query errors.
	if _, err := s.SearchText(ctx, "", 1); err != core.ErrEmptyQuery {
		t.Fatalf("empty query: %v", err)
	}
	if _, err := s.SearchImage(ctx, nil, "image/png", 1); err != core.ErrEmptyQuery {
		t.Fatalf("empty image query: %v", err)
	}

	// Cross-modal by image: same bytes find the exact item, and the
	// image list includes the text item (shared space).
	res, err = s.SearchImage(ctx, []byte("planet-img"), "image/png", 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].Item.ID != "i1" {
		t.Fatalf("image search = %+v", res)
	}
	sawText := false
	for _, r := range res {
		if r.Item.Modality == embedder.ModalityText {
			sawText = true
		}
	}
	if !sawText {
		t.Fatal("image query should surface text items (shared space)")
	}

	// topK=0 defaults to 10.
	if res, _ = s.SearchText(ctx, "planets", 0); len(res) != 4 {
		t.Fatalf("default topK: %d results", len(res))
	}
}
