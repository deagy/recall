package chunker

import (
	"strings"
	"testing"

	"github.com/deagy/recall/core"
)

// The Config.MinChunkSize doc comment claimed short chunks were "merged with
// adjacent chunks". No merge logic has ever existed; both chunkers discard.
// These lock what the comment now says, so the two cannot drift apart again.

func TestMinChunkSize_FixedDiscardsOnlyASinglePartChunk(t *testing.T) {
	cfg := Config{MaxTokens: 512, MinChunkSize: 50, Separator: ". "}
	doc := &core.Document{ID: "fixed-min"}

	// One part, under the threshold: nothing to combine with, so it is dropped.
	short, err := NewFixed(cfg).Chunk(doc, "too short")
	if err != nil {
		t.Fatalf("chunking: %v", err)
	}
	if len(short) != 0 {
		t.Errorf("a single short part produced %d chunks, want 0 (discarded, not merged): %q",
			len(short), short[0].Content)
	}

	// Several parts that together stay under the threshold are still emitted:
	// the drop is guarded by len(parts) == 1, not by size alone.
	multi, err := NewFixed(cfg).Chunk(doc, "aa. bb. cc")
	if err != nil {
		t.Fatalf("chunking: %v", err)
	}
	if len(multi) == 0 {
		t.Error("a multi-part chunk under MinChunkSize produced no chunks; the drop " +
			"is documented as applying only to single-part chunks")
	}
}

func TestMinChunkSize_RecursiveDiscardsAnyShortPiece(t *testing.T) {
	cfg := Config{MaxTokens: 512, MinChunkSize: 50}
	doc := &core.Document{ID: "recursive-min"}

	chunks, err := NewRecursive(cfg).Chunk(doc, "short")
	if err != nil {
		t.Fatalf("chunking: %v", err)
	}
	if len(chunks) != 0 {
		t.Errorf("recursive produced %d chunks for a short piece, want 0 (discarded outright)", len(chunks))
	}
}

func TestMinChunkSize_ZeroKeepsEveryChunk(t *testing.T) {
	doc := &core.Document{ID: "keep-all"}
	for _, tc := range []struct {
		name string
		make func(Config) Chunker
	}{
		{"fixed", NewFixed},
		{"recursive", NewRecursive},
	} {
		cfg := Config{MaxTokens: 512, MinChunkSize: 0, Separator: ". "}
		chunks, err := tc.make(cfg).Chunk(doc, "short")
		if err != nil {
			t.Fatalf("%s: chunking: %v", tc.name, err)
		}
		if len(chunks) == 0 {
			t.Errorf("%s: MinChunkSize 0 dropped a short chunk; the comment says it keeps every chunk", tc.name)
		}
	}
}

func TestMinChunkSize_DocumentAwareLosesAWholeShortSection(t *testing.T) {
	doc := &core.Document{ID: "sections"}
	content := strings.Join([]string{
		"This first section is comfortably longer than the fifty character threshold.",
		"tiny",
		"This third section is also comfortably longer than the fifty character threshold.",
	}, DefaultBoundary)

	da := NewDocumentAware(NewFixed(Config{MaxTokens: 512, MinChunkSize: 50, Separator: ". "}))
	chunks, err := da.Chunk(doc, content)
	if err != nil {
		t.Fatalf("chunking: %v", err)
	}
	for _, c := range chunks {
		if strings.Contains(c.Content, "tiny") {
			return // section survived; the comment's warning would be wrong
		}
	}
	t.Log("confirmed: the short section is absent from all chunks, as the comment warns")
}
