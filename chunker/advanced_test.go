package chunker

import (
	"strings"
	"testing"

	"github.com/deagy/recall/core"
)

func longDoc(id string) (*core.Document, string) {
	var b strings.Builder
	for i := 0; i < 120; i++ {
		b.WriteString("This is sentence number " + itoa(i) + " about topic " + itoa(i%7) + " in document " + id + ". ")
	}
	return &core.Document{ID: id}, b.String()
}

func TestParentChild_ChunkAndExpand(t *testing.T) {
	doc, content := longDoc("pc-doc")
	pc := NewParentChild(
		NewFixed(Config{MaxTokens: 512, MinChunkSize: 50, Separator: ". "}),
		NewFixed(Config{MaxTokens: 64, MinChunkSize: 20, Separator: ". "}),
	)
	children, err := pc.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(children) < 2 {
		t.Fatal("expected multiple child chunks, got", len(children))
	}
	parents := pc.Parents("pc-doc")
	if len(parents) == 0 || len(parents) >= len(children) {
		t.Fatalf("parent/child granularity wrong: parents=%d children=%d", len(parents), len(children))
	}

	// Every child should carry a parent ID that resolves, and its text
	// should live inside the parent.
	unlinked := 0
	for _, c := range children {
		pid := c.GetMetadataString(MetaParentID)
		if pid == "" {
			unlinked++
			continue
		}
		p, ok := pc.ParentFor("pc-doc", c)
		if !ok {
			t.Fatalf("parent %q not found for child %s", pid, c.ID)
		}
		anchor := c.Content
		if len(anchor) > 40 {
			anchor = anchor[:40]
		}
		if !strings.Contains(p.Content, strings.TrimSpace(anchor)) {
			t.Fatalf("child %s anchor not contained in parent %s", c.ID, p.ID)
		}
	}
	if unlinked > len(children)/2 {
		t.Fatalf("too many unlinked children: %d/%d", unlinked, len(children))
	}

	// ExpandChunks de-duplicates to parents, preserving order.
	expanded := pc.ExpandChunks("pc-doc", children)
	if len(expanded) == 0 {
		t.Fatal("expand produced nothing")
	}
	seen := map[string]bool{}
	for _, e := range expanded {
		if seen[e.ID] {
			t.Fatalf("duplicate parent %s in expansion", e.ID)
		}
		seen[e.ID] = true
	}

	// Parents ordered by ChunkIndex.
	for i := 1; i < len(parents); i++ {
		if parents[i].ChunkIndex < parents[i-1].ChunkIndex {
			t.Fatal("parents not ordered by ChunkIndex")
		}
	}

	// ParentFor with missing/nil inputs.
	if _, ok := pc.ParentFor("pc-doc", nil); ok {
		t.Fatal("nil child should not resolve")
	}
	if _, ok := pc.ParentFor("other-doc", children[0]); ok {
		t.Fatal("unknown doc should not resolve")
	}
	// Child without parent metadata passes through ExpandChunks.
	orphan := &core.Chunk{ID: "orphan", Content: "x", DocumentRef: "pc-doc"}
	got := pc.ExpandChunks("pc-doc", []*core.Chunk{orphan})
	if len(got) != 1 || got[0].ID != "orphan" {
		t.Fatalf("orphan should pass through, got %+v", got)
	}

	// Empty content.
	if c, err := pc.Chunk(doc, ""); err != nil || c != nil {
		t.Fatalf("empty content: %v %v", c, err)
	}
}

func TestParentChild_RechunkReplacesCache(t *testing.T) {
	doc, content := longDoc("re")
	pc := NewParentChild(
		NewFixed(Config{MaxTokens: 512, MinChunkSize: 50, Separator: ". "}),
		NewFixed(Config{MaxTokens: 64, MinChunkSize: 20, Separator: ". "}),
	)
	if _, err := pc.Chunk(doc, content); err != nil {
		t.Fatal(err)
	}
	n1 := len(pc.Parents("re"))
	// Re-chunking the same doc must replace (not grow) the cache.
	if _, err := pc.Chunk(doc, content); err != nil {
		t.Fatal(err)
	}
	if n2 := len(pc.Parents("re")); n2 != n1 {
		t.Fatalf("cache grew after re-chunk: %d -> %d", n1, n2)
	}
}

func TestDocumentAware_NoCrossBoundaryChunks(t *testing.T) {
	doc := &core.Document{ID: "daw"}
	content := "First document body alpha bravo charlie.\n---\nSecond document body delta echo foxtrot.\n---\nThird document body golf hotel india."
	da := NewDocumentAware(NewFixed(Config{MaxTokens: 32, MinChunkSize: 10, Separator: ". "}))
	chunks, err := da.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks, got %d", len(chunks))
	}
	// No chunk may contain text from two different sections.
	multi := 0
	for _, c := range chunks {
		hits := 0
		for _, marker := range []string{"alpha", "delta", "golf"} {
			if strings.Contains(c.Content, marker) {
				hits++
			}
		}
		if hits > 1 {
			multi++
		}
	}
	if multi > 0 {
		t.Fatalf("%d chunks cross section boundaries", multi)
	}
	// Section metadata present.
	for _, c := range chunks {
		if v := c.GetMetadata(MetaSectionIndex); v == nil {
			t.Fatalf("chunk %s missing section index", c.ID)
		}
		if v := c.GetMetadata(MetaSectionCount); v == nil || v.String() != "3" {
			t.Fatalf("chunk %s section count = %v", c.ID, v)
		}
	}

	// Custom boundary + empty segments skipped.
	da2 := NewDocumentAware(NewFixed(Config{MaxTokens: 32, MinChunkSize: 5, Separator: ". "}))
	da2.Boundary = "###"
	if _, err := da2.Chunk(doc, "one two ### ### three four"); err != nil {
		t.Fatal(err)
	}
	// Empty content.
	if c, err := da2.Chunk(doc, ""); err != nil || c != nil {
		t.Fatalf("empty: %v %v", c, err)
	}
}

func TestAdaptive_Sizes(t *testing.T) {
	ad := &AdaptiveChunker{MinTokens: 64, MaxTokens: 1024, SentencesPerChunk: 6, Separator: ". "}

	shortDoc := strings.Repeat("Short. ", 120)
	longContent := strings.Repeat(makeLongSentence(80)+". ", 20)

	shortTokens := ad.EstimateTokens(shortDoc)
	longTokens := ad.EstimateTokens(longContent)
	if shortTokens >= longTokens {
		t.Fatalf("short %d should be < long %d", shortTokens, longTokens)
	}
	if shortTokens < 64 || shortTokens > 1024 || longTokens < 64 || longTokens > 1024 {
		t.Fatalf("tokens out of bounds: short=%d long=%d", shortTokens, longTokens)
	}
	// Empty content -> MinTokens.
	if got := ad.EstimateTokens(""); got != 64 {
		t.Fatalf("empty estimate = %d", got)
	}
}

func makeLongSentence(words int) string {
	parts := make([]string, words)
	for i := range parts {
		parts[i] = "word" + itoa(i)
	}
	return strings.Join(parts, " ")
}

func TestAdaptive_Chunk(t *testing.T) {
	doc, content := longDoc("ad")
	ad := NewAdaptive(Config{MaxTokens: 128, MinChunkSize: 100, Separator: ". "})
	if ad.SentencesPerChunk != 6 {
		t.Fatalf("default sentences per chunk = %d", ad.SentencesPerChunk)
	}
	chunks, err := ad.Chunk(doc, content)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	// Chunks should respect the bound: max chars <= MaxTokens*4 (+margin).
	maxChars := 0
	for _, c := range chunks {
		if len(c.Content) > maxChars {
			maxChars = len(c.Content)
		}
	}
	if maxChars > 128*4+50 {
		t.Fatalf("chunk too large: %d chars", maxChars)
	}
	if c, err := ad.Chunk(doc, ""); err != nil || c != nil {
		t.Fatalf("empty: %v %v", c, err)
	}
	// NewAdaptive clamps min>max.
	odd := NewAdaptive(Config{MaxTokens: 10, MinChunkSize: 400})
	if odd.MinTokens >= odd.MaxTokens {
		t.Fatalf("clamp failed: min=%d max=%d", odd.MinTokens, odd.MaxTokens)
	}
}
