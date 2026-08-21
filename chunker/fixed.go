package chunker

import (
	"strings"
	"unicode/utf8"

	"github.com/deagy/recall/core"
)

// FixedChunker splits text into chunks of approximately fixed size.
// It uses character-based splitting with overlap.
type FixedChunker struct {
	config Config
}

// NewFixed creates a new FixedChunker with the given config.
func NewFixed(cfg Config) Chunker {
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 512
	}
	if cfg.Separator == "" {
		cfg.Separator = "\n\n"
	}
	if cfg.OverlapTokens < 0 {
		cfg.OverlapTokens = 0
	}
	if cfg.OverlapTokens >= cfg.MaxTokens {
		cfg.OverlapTokens = cfg.MaxTokens / 2
	}
	return &FixedChunker{config: cfg}
}

// Chunk splits the document content into fixed-size chunks. Each chunk is at
// most MaxTokens*4 runes (rough estimate: 1 token ≈ 4 chars); a single part
// longer than that is pre-split so it can never be flushed whole into a
// chunk. The config's Separator splits the input and rejoins parts within a
// chunk.
func (f *FixedChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}

	sep := f.config.Separator
	sepLen := utf8.RuneCountInString(sep)

	maxChars := f.config.MaxTokens * 4 // rough estimate: 1 token ≈ 4 chars
	overlapChars := f.config.OverlapTokens * 4

	// Split by separator first, then reassemble into fixed-size chunks.
	// Pre-split any part that exceeds maxChars so no single part can
	// produce an oversized chunk, regardless of accumulated state.
	parts := strings.Split(content, sep)
	if len(parts) == 0 {
		parts = []string{content}
	}
	normalized := make([]string, 0, len(parts))
	for _, part := range parts {
		if utf8.RuneCountInString(strings.TrimSpace(part)) > maxChars {
			normalized = append(normalized, splitBySize(part, maxChars, sep)...)
			continue
		}
		normalized = append(normalized, part)
	}

	var chunks []*core.Chunk
	var currentParts []string
	currentSize := 0 // rune count of the joined content, separators included
	chunkIndex := 0

	for _, part := range normalized {
		partLen := utf8.RuneCountInString(strings.TrimSpace(part))

		// If adding this part would exceed max size, emit current chunk
		if len(currentParts) > 0 && currentSize+sepLen+partLen > maxChars {
			chunk := f.buildChunk(doc, currentParts, chunkIndex)
			if chunk != nil {
				chunks = append(chunks, chunk)
				chunkIndex++
			}

			// Start new chunk with overlap from previous
			if overlapChars > 0 && len(currentParts) > 0 {
				overlapText := f.getOverlap(currentParts, overlapChars)
				if overlapText != "" {
					currentParts = []string{overlapText}
					currentSize = utf8.RuneCountInString(overlapText)
				} else {
					currentParts = nil
					currentSize = 0
				}
			} else {
				currentParts = nil
				currentSize = 0
			}
		}

		if len(currentParts) > 0 {
			currentSize += sepLen
		}
		currentParts = append(currentParts, part)
		currentSize += partLen
	}

	// Flush remaining parts
	if len(currentParts) > 0 {
		chunk := f.buildChunk(doc, currentParts, chunkIndex)
		if chunk != nil {
			chunks = append(chunks, chunk)
		}
	}

	return chunks, nil
}

func (f *FixedChunker) buildChunk(doc *core.Document, parts []string, index int) *core.Chunk {
	content := strings.Join(parts, f.config.Separator)
	content = strings.TrimSpace(content)

	if utf8.RuneCountInString(content) < f.config.MinChunkSize && len(parts) == 1 {
		return nil // Too small, skip
	}

	return &core.Chunk{
		ID:          generateChunkID(doc.ID, index),
		Content:     content,
		DocumentRef: doc.ID,
		ChunkIndex:  index,
		Metadata:    copyMetadata(doc.Metadata),
	}
}

func (f *FixedChunker) getOverlap(parts []string, maxChars int) string {
	overlap := ""
	for i := len(parts) - 1; i >= 0; i-- {
		candidate := parts[i] + f.config.Separator + overlap
		if utf8.RuneCountInString(candidate) > maxChars {
			break
		}
		overlap = candidate
	}
	return strings.TrimRight(overlap, f.config.Separator)
}

// splitBySize splits a string into parts of approximately maxChars size,
// trying to split at newline boundaries.
func splitBySize(s string, maxChars int, sep string) []string {
	if utf8.RuneCountInString(s) <= maxChars {
		return []string{s}
	}

	var parts []string
	remaining := s
	for len(remaining) > 0 {
		if utf8.RuneCountInString(remaining) <= maxChars {
			parts = append(parts, remaining)
			break
		}

		// Try to find a good split point near maxChars
		splitAt := maxChars
		remainingRunes := []rune(remaining)
		if splitAt >= len(remainingRunes) {
			parts = append(parts, remaining)
			break
		}

		// Look for a newline near the split point
		bestSplit := splitAt
		searchStart := splitAt - 20
		if searchStart < 0 {
			searchStart = 0
		}
		for i := searchStart; i < splitAt; i++ {
			if remainingRunes[i] == '\n' {
				bestSplit = i + 1
				break
			}
		}

		part := string(remainingRunes[:bestSplit])
		parts = append(parts, strings.TrimSpace(part))
		remaining = strings.TrimSpace(string(remainingRunes[bestSplit:]))
	}

	return parts
}

func generateChunkID(docID string, index int) string {
	return docID + "::chunk-" + itoa(index)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := false
	if i < 0 {
		neg = true
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

func copyMetadata(m map[string]core.Value) map[string]core.Value {
	if m == nil {
		return nil
	}
	c := make(map[string]core.Value, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}
