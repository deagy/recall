package chunker

import (
	"strings"
	"unicode/utf8"

	"github.com/deagy/recall/core"
)

// RecursiveChunker splits text recursively by progressively smaller separators.
// It first tries to split by paragraphs, then sentences, then characters.
type RecursiveChunker struct {
	config Config
}

// NewRecursive creates a new RecursiveChunker with the given config.
func NewRecursive(cfg Config) Chunker {
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 512
	}
	return &RecursiveChunker{config: cfg}
}

// separator hierarchy for recursive splitting
var separators = []string{
	"\n\n\n", // triple newline
	"\n\n",   // paragraph
	"\n",     // line
	". ",     // sentence
	", ",     // clause
	"",       // character (fallback)
}

// Chunk splits the document content using recursive splitting.
func (r *RecursiveChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}

	maxChars := r.config.MaxTokens * 4
	minChars := r.config.MinChunkSize

	// Start with the full content as a single piece
	pieces := []string{content}

	// Recursively split until all pieces are small enough
	for {
		// Check if all pieces are within size limit
		allSmall := true
		for _, p := range pieces {
			if utf8.RuneCountInString(p) > maxChars {
				allSmall = false
				break
			}
		}
		if allSmall {
			break
		}

		// Find the largest piece that needs splitting
		largestIdx := 0
		largestLen := 0
		for i, p := range pieces {
			if utf8.RuneCountInString(p) > largestLen {
				largestLen = utf8.RuneCountInString(p)
				largestIdx = i
			}
		}

		// Try each separator level
		split := false
		for _, sep := range separators {
			if sep == "" {
				continue
			}
			newPieces := splitBySeparator(pieces[largestIdx], sep)
			if len(newPieces) > 1 {
				// Replace the large piece with its splits
				result := make([]string, 0, len(pieces)-1+len(newPieces))
				result = append(result, pieces[:largestIdx]...)
				result = append(result, newPieces...)
				result = append(result, pieces[largestIdx+1:]...)
				pieces = result
				split = true
				break
			}
		}

		if !split {
			// Last resort: split at the midpoint
			mid := largestLen / 2
			runes := []rune(pieces[largestIdx])
			firstHalf := string(runes[:mid])
			secondHalf := string(runes[mid:])
			result := make([]string, 0, len(pieces)+1)
			result = append(result, pieces[:largestIdx]...)
			result = append(result, firstHalf)
			result = append(result, secondHalf)
			result = append(result, pieces[largestIdx+1:]...)
			pieces = result
		}
	}

	// Build chunks from pieces
	var chunks []*core.Chunk
	for i, piece := range pieces {
		piece = strings.TrimSpace(piece)
		if utf8.RuneCountInString(piece) < minChars {
			continue
		}
		chunks = append(chunks, &core.Chunk{
			ID:          generateChunkID(doc.ID, i),
			Content:     piece,
			DocumentRef: doc.ID,
			ChunkIndex:  i,
			Metadata:    copyMetadata(doc.Metadata),
		})
	}

	return chunks, nil
}

func splitBySeparator(s, sep string) []string {
	parts := strings.Split(s, sep)
	var result []string
	current := ""
	for _, p := range parts {
		if current != "" {
			current += sep
		}
		current += p
		trimmed := strings.TrimSpace(current)
		if utf8.RuneCountInString(trimmed) > 0 {
			result = append(result, trimmed)
			current = ""
		}
	}
	if current != "" {
		result = append(result, strings.TrimSpace(current))
	}
	return result
}
