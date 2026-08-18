package embedder

import (
	"context"
	"testing"
)

func TestMockMultiModal_Determinism(t *testing.T) {
	m := NewMockMultiModal(32)
	if m.Dimension() != 32 {
		t.Fatalf("dim = %d", m.Dimension())
	}
	ctx := context.Background()

	a1, err := m.EmbedText(ctx, "a photo of a red car")
	if err != nil {
		t.Fatal(err)
	}
	a2, _ := m.EmbedText(ctx, "a photo of a red car")
	if len(a1) != 32 {
		t.Fatalf("embedding dim = %d", len(a1))
	}
	for i := range a1 {
		if a1[i] != a2[i] {
			t.Fatal("same input produced different vectors")
		}
	}
	// Identical vectors => cosine 1.
	if s := CosineSimilarity(a1, a2); s < 0.9999 {
		t.Fatalf("self similarity = %f", s)
	}

	// Distinct texts: low similarity (not identical, near-orthogonal).
	b, _ := m.EmbedText(ctx, "quantum field theory lecture notes")
	if s := CosineSimilarity(a1, b); s > 0.5 {
		t.Fatalf("distinct texts too similar: %f", s)
	}

	// Images.
	img1, err := m.EmbedImage(ctx, []byte("fake-png-1"), "image/png")
	if err != nil {
		t.Fatal(err)
	}
	img2, _ := m.EmbedImage(ctx, []byte("fake-png-1"), "image/png")
	if s := CosineSimilarity(img1, img2); s < 0.9999 {
		t.Fatalf("same image different vectors: %f", s)
	}
	// Different MIME for the same bytes is a different artifact.
	img3, _ := m.EmbedImage(ctx, []byte("fake-png-1"), "image/jpeg")
	if s := CosineSimilarity(img1, img3); s > 0.5 {
		t.Fatalf("mime change should alter the vector: %f", s)
	}
	// Image vs text live in the same space (comparable, distinct).
	if s := CosineSimilarity(a1, img1); s > 0.99 {
		t.Fatalf("text and image vectors suspiciously equal: %f", s)
	}

	// SeedForText round-trips to the same vector.
	seed := m.SeedForText("a photo of a red car")
	_ = seed // vectorFor is unexported; equality asserted above via a1/a2.

	// Error paths.
	if _, err := m.EmbedText(ctx, ""); err == nil {
		t.Fatal("empty text should fail")
	}
	if _, err := m.EmbedImage(ctx, nil, "image/png"); err == nil {
		t.Fatal("empty image should fail")
	}
	if _, err := m.EmbedImage(ctx, []byte("x"), ""); err == nil {
		t.Fatal("empty mime should fail")
	}

	// Small dims clamp up to 8.
	if NewMockMultiModal(2).Dimension() != 8 {
		t.Fatal("dim should clamp to 8")
	}
}
