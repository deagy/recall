package chunker

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/deagy/recall/core"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MaxTokens != 512 {
		t.Errorf("expected MaxTokens 512, got %d", cfg.MaxTokens)
	}
	if cfg.MinChunkSize != 50 {
		t.Errorf("expected MinChunkSize 50, got %d", cfg.MinChunkSize)
	}
	if cfg.OverlapTokens != 50 {
		t.Errorf("expected OverlapTokens 50, got %d", cfg.OverlapTokens)
	}
}

func TestFixedChunker_EmptyContent(t *testing.T) {
	c := NewFixed(DefaultConfig())
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if chunks != nil {
		t.Errorf("expected nil chunks, got %d", len(chunks))
	}
}

func TestFixedChunker_SmallContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 512
	cfg.MinChunkSize = 10
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "Hello world, this is a small piece of text."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Content != content {
		t.Errorf("expected content to match, got %q", chunks[0].Content)
	}
	if chunks[0].DocumentRef != "doc-1" {
		t.Errorf("expected DocumentRef 'doc-1', got %q", chunks[0].DocumentRef)
	}
}

func TestFixedChunker_LargeContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 50 // small for testing
	cfg.MinChunkSize = 10
	cfg.OverlapTokens = 10
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	// Create content that will need multiple chunks
	var sb strings.Builder
	for i := 0; i < 20; i++ {
		sb.WriteString(strings.Repeat("word ", 30))
		sb.WriteString("\n\n")
	}
	content := sb.String()

	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	// Verify all chunks reference the same document
	for _, chunk := range chunks {
		if chunk.DocumentRef != "doc-1" {
			t.Errorf("expected DocumentRef 'doc-1', got %q", chunk.DocumentRef)
		}
		if chunk.ChunkIndex < 0 {
			t.Errorf("expected non-negative ChunkIndex, got %d", chunk.ChunkIndex)
		}
	}
}

func TestFixedChunker_MetadataPropagation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	doc.Metadata = map[string]core.Value{
		"author": core.String{Value: "Alice"},
		"date":   core.Number{Value: 2024},
	}
	content := "Short text."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Metadata["author"].String() != "Alice" {
		t.Errorf("expected author 'Alice', got %q", chunks[0].Metadata["author"].String())
	}
}

func TestRecursiveChunker_EmptyContent(t *testing.T) {
	c := NewRecursive(DefaultConfig())
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if chunks != nil {
		t.Errorf("expected nil chunks, got %d", len(chunks))
	}
}

func TestRecursiveChunker_SmallContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	content := "Hello world."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestRecursiveChunker_ParagraphSplitting(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 5 // extremely small to force splitting
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "First paragraph with enough words to exceed the tiny limit here.\n\nSecond paragraph also with enough words to exceed the limit here.\n\nThird paragraph with enough words to exceed the limit here too."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks from paragraph splitting, got %d", len(chunks))
	}
}

func TestRecursiveChunker_VeryLongContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 10 // tiny for testing
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	// Create a very long string that must be split many times
	content := strings.Repeat("a very long sentence that exceeds the limit. ", 200)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for very long content, got %d", len(chunks))
	}

	// Verify all chunks are within size limit
	maxChars := cfg.MaxTokens * 4
	for i, chunk := range chunks {
		runes := utf8.RuneCountInString(chunk.Content)
		if runes > maxChars {
			t.Errorf("chunk %d has %d runes, exceeds max %d", i, runes, maxChars)
		}
	}
}

func TestFixedChunker_ChunkIDs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := strings.Repeat("word ", 100)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify unique chunk IDs
	seen := make(map[string]bool)
	for _, chunk := range chunks {
		if seen[chunk.ID] {
			t.Errorf("duplicate chunk ID: %s", chunk.ID)
		}
		seen[chunk.ID] = true
		if !strings.HasPrefix(chunk.ID, "doc-1::chunk-") {
			t.Errorf("unexpected chunk ID format: %s", chunk.ID)
		}
	}
}

func TestFixedChunker_CustomSeparator(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Separator = ". "
	cfg.MinChunkSize = 1
	cfg.MaxTokens = 50
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestFixedChunker_ZeroOverlap(t *testing.T) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 0
	cfg.MinChunkSize = 1
	cfg.MaxTokens = 50
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := strings.Repeat("word ", 100)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestFixedChunker_LargeOverlap(t *testing.T) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 1000
	cfg.MinChunkSize = 1
	cfg.MaxTokens = 100
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := strings.Repeat("word ", 50)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestFixedChunker_NewlineSeparator(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Separator = "\n"
	cfg.MinChunkSize = 1
	cfg.MaxTokens = 100
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestRecursiveChunker_EmptyMetadata(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	doc.Metadata = nil

	content := "Hello world. This is a test."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestRecursiveChunker_VeryShortContent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "Hi"
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestRecursiveChunker_NoSeparators(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 10
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "ThisIsAVeryLongStringWithoutAnySpacesOrPunctuationThatWillNeedToBeSplitByCharacters"
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestRecursiveChunker_SingleParagraph(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 1000
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "This is a single paragraph with enough text to be considered a valid chunk by the recursive chunker."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestRecursiveChunker_MetadataPropagation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")
	doc.Metadata = map[string]core.Value{
		"author": core.String{Value: "TestAuthor"},
	}

	content := "This is a test document with metadata that should be propagated to chunks."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	if chunks[0].Metadata["author"].String() != "TestAuthor" {
		t.Errorf("expected author 'TestAuthor', got %q", chunks[0].Metadata["author"].String())
	}
}

func TestRecursiveChunker_ChunkIDs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 10
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := strings.Repeat("word ", 100)
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	seen := make(map[string]bool)
	for _, chunk := range chunks {
		if seen[chunk.ID] {
			t.Errorf("duplicate chunk ID: %s", chunk.ID)
		}
		seen[chunk.ID] = true
	}
}

func TestNewRecursive_DefaultConfig(t *testing.T) {
	cfg := Config{}
	c := NewRecursive(cfg)
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	// Verify it works with default config
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "Hello world. This is a test.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestNewRecursive_CustomConfig(t *testing.T) {
	cfg := Config{
		MaxTokens: 100,
	}
	c := NewRecursive(cfg)
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "Hello world. This is a test.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestNewFixed_DefaultConfig(t *testing.T) {
	cfg := Config{}
	c := NewFixed(cfg)
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "Hello world. This is a test.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestNewFixed_NegativeOverlap(t *testing.T) {
	cfg := Config{
		MaxTokens:     100,
		OverlapTokens: -10,
	}
	c := NewFixed(cfg)
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, strings.Repeat("word ", 50))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestNewFixed_OverlapExceedsMax(t *testing.T) {
	cfg := Config{
		MaxTokens:     100,
		OverlapTokens: 200,
	}
	c := NewFixed(cfg)
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, strings.Repeat("word ", 50))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestNewFixed_ZeroMaxTokens(t *testing.T) {
	cfg := Config{
		MaxTokens: 0,
	}
	c := NewFixed(cfg)
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	doc := core.NewDocument("doc-1", "Test", "source")
	chunks, err := c.Chunk(doc, "Hello world. This is a test.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestChunker_Interface(t *testing.T) {
	c := NewFixed(DefaultConfig())
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}

	r := NewRecursive(DefaultConfig())
	if r == nil {
		t.Fatal("expected non-nil chunker")
	}
}

func TestFixedChunker_OnlyNewlines(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "\n\n\n\n\n"
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestFixedChunker_SingleWord(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "Hello"
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestFixedChunker_MultipleSeparators(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Separator = ". "
	cfg.MinChunkSize = 1
	cfg.MaxTokens = 100
	c := NewFixed(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestRecursiveChunker_EmptyLines(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "\n\n\n\n\n"
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_ = chunks
}

func TestRecursiveChunker_SingleSentence(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 1000
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "This is a single sentence with enough text to be considered a valid chunk."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
}

func TestRecursiveChunker_MultipleSentences(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 1000
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "First sentence. Second sentence. Third sentence. Fourth sentence."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestRecursiveChunker_Paragraphs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxTokens = 1000
	cfg.MinChunkSize = 1
	c := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	content := "First paragraph with enough text.\n\nSecond paragraph with enough text.\n\nThird paragraph with enough text."
	chunks, err := c.Chunk(doc, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestFixedChunker_GetOverlap(t *testing.T) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 10
	cfg.MinChunkSize = 1
	c := NewFixed(cfg)

	fc := c.(*FixedChunker)
	parts := []string{"part1", "part2", "part3"}
	overlap := fc.getOverlap(parts, 50)
	if overlap == "" {
		t.Error("expected non-empty overlap")
	}
}

func TestFixedChunker_GetOverlap_EmptyParts(t *testing.T) {
	cfg := DefaultConfig()
	c := NewFixed(cfg)

	fc := c.(*FixedChunker)
	overlap := fc.getOverlap([]string{}, 50)
	if overlap != "" {
		t.Errorf("expected empty overlap, got %q", overlap)
	}
}

func TestFixedChunker_GetOverlap_SmallMaxChars(t *testing.T) {
	cfg := DefaultConfig()
	cfg.OverlapTokens = 10
	c := NewFixed(cfg)

	fc := c.(*FixedChunker)
	parts := []string{"part1", "part2"}
	overlap := fc.getOverlap(parts, 0)
	if overlap != "" {
		t.Errorf("expected empty overlap, got %q", overlap)
	}
}

func TestRecursiveChunker_splitBySeparator(t *testing.T) {
	s := "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
	sep := "\n\n"
	parts := splitBySeparator(s, sep)
	if len(parts) < 2 {
		t.Fatalf("expected at least 2 parts, got %d", len(parts))
	}
}

func TestRecursiveChunker_splitBySeparator_NoMatch(t *testing.T) {
	s := "No separators here"
	sep := "\n\n"
	parts := splitBySeparator(s, sep)
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
}

func TestRecursiveChunker_splitBySeparator_EmptyString(t *testing.T) {
	s := ""
	sep := "\n\n"
	parts := splitBySeparator(s, sep)
	_ = parts
}

func TestFixedChunker_splitBySize_SmallString(t *testing.T) {
	s := "Small"
	maxChars := 100
	sep := "\n\n"
	parts := splitBySize(s, maxChars, sep)
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
}

func TestFixedChunker_splitBySize_LargeString(t *testing.T) {
	s := strings.Repeat("word ", 100)
	maxChars := 50
	sep := "\n\n"
	parts := splitBySize(s, maxChars, sep)
	if len(parts) < 2 {
		t.Fatalf("expected at least 2 parts, got %d", len(parts))
	}
}

func TestFixedChunker_splitBySize_NoNewlines(t *testing.T) {
	s := strings.Repeat("word ", 100)
	maxChars := 50
	sep := "\n\n"
	parts := splitBySize(s, maxChars, sep)
	if len(parts) < 2 {
		t.Fatalf("expected at least 2 parts, got %d", len(parts))
	}
}

func TestFixedChunker_generateChunkID(t *testing.T) {
	id := generateChunkID("doc-1", 0)
	if id != "doc-1::chunk-0" {
		t.Errorf("expected 'doc-1::chunk-0', got %q", id)
	}

	id = generateChunkID("doc-1", 5)
	if id != "doc-1::chunk-5" {
		t.Errorf("expected 'doc-1::chunk-5', got %q", id)
	}
}

func TestFixedChunker_itoa_Zero(t *testing.T) {
	if itoa(0) != "0" {
		t.Errorf("expected '0', got %q", itoa(0))
	}
}

func TestFixedChunker_itoa_Positive(t *testing.T) {
	if itoa(42) != "42" {
		t.Errorf("expected '42', got %q", itoa(42))
	}
}

func TestFixedChunker_itoa_Negative(t *testing.T) {
	if itoa(-42) != "-42" {
		t.Errorf("expected '-42', got %q", itoa(-42))
	}
}

func TestFixedChunker_copyMetadata_Nil(t *testing.T) {
	result := copyMetadata(nil)
	if result != nil {
		t.Error("expected nil for nil input")
	}
}

func TestFixedChunker_copyMetadata_Empty(t *testing.T) {
	result := copyMetadata(map[string]core.Value{})
	if result == nil {
		t.Error("expected non-nil for empty map")
	}
	if len(result) != 0 {
		t.Errorf("expected empty map, got %d items", len(result))
	}
}

func TestFixedChunker_copyMetadata_WithItems(t *testing.T) {
	input := map[string]core.Value{
		"key": core.String{Value: "value"},
	}
	result := copyMetadata(input)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result) != 1 {
		t.Errorf("expected 1 item, got %d", len(result))
	}
	if result["key"].String() != "value" {
		t.Errorf("expected 'value', got %q", result["key"].String())
	}
}

func TestRecursiveChunker_buildChunk_Valid(t *testing.T) {
	cfg := Config{
		MaxTokens:    512,
		MinChunkSize: 1,
	}
	rc := NewRecursive(cfg)
	doc := core.NewDocument("doc-1", "Test", "source")

	// RecursiveChunker doesn't expose buildChunk, but we can test through Chunk
	chunks, err := rc.Chunk(doc, "Hello world. This is a test.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestSeparators_Variable(t *testing.T) {
	if len(separators) != 6 {
		t.Errorf("expected 6 separators, got %d", len(separators))
	}
	if separators[0] != "\n\n\n" {
		t.Errorf("expected '\n\n\n', got %q", separators[0])
	}
	if separators[1] != "\n\n" {
		t.Errorf("expected '\n\n', got %q", separators[1])
	}
	if separators[2] != "\n" {
		t.Errorf("expected '\n', got %q", separators[2])
	}
	if separators[3] != ". " {
		t.Errorf("expected '. ', got %q", separators[3])
	}
	if separators[4] != ", " {
		t.Errorf("expected ', ', got %q", separators[4])
	}
	if separators[5] != "" {
		t.Errorf("expected '', got %q", separators[5])
	}
}
