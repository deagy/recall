package chunker

import (
	"context"
	"strings"

	"github.com/deagy/recall/core"
)

// SemanticChunker splits text into chunks based on semantic similarity between
// sentences. Sentences with low similarity are used as split points, allowing
// chunks to maintain thematic coherence.
type SemanticChunker struct {
	embedder Embedder
	config   SemanticConfig
}

// SemanticConfig holds configuration for semantic chunking.
type SemanticConfig struct {
	// Threshold is the minimum cosine similarity between adjacent sentences
	// to keep them in the same chunk. Lower values create larger chunks.
	// Default: 0.7
	Threshold float64

	// MinChunkSize is the minimum number of characters for a chunk.
	// Default: 100
	MinChunkSize int

	// MaxChunkSize is the maximum number of characters for a chunk.
	// Default: 2000
	MaxChunkSize int

	// PreserveOverlap is whether to include overlapping sentences between chunks.
	// Default: false
	PreserveOverlap bool

	// OverlapSize is the number of sentences to overlap between chunks.
	// Default: 1
	OverlapSize int
}

// DefaultSemanticConfig returns a SemanticConfig with sensible defaults.
func DefaultSemanticConfig() SemanticConfig {
	return SemanticConfig{
		Threshold:     0.7,
		MinChunkSize:  100,
		MaxChunkSize:  2000,
		OverlapSize:   1,
	}
}

// Embedder is a minimal interface for text embedding, used by SemanticChunker.
// This avoids a circular dependency with the embedder package.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// NewSemantic creates a new SemanticChunker with the given embedder and config.
func NewSemantic(embedder Embedder, cfg SemanticConfig) *SemanticChunker {
	if cfg.Threshold <= 0 || cfg.Threshold > 1 {
		cfg.Threshold = 0.7
	}
	if cfg.MinChunkSize <= 0 {
		cfg.MinChunkSize = 100
	}
	if cfg.MaxChunkSize <= 0 {
		cfg.MaxChunkSize = 2000
	}
	if cfg.MaxChunkSize < cfg.MinChunkSize {
		cfg.MaxChunkSize = cfg.MinChunkSize * 2
	}
	return &SemanticChunker{
		embedder: embedder,
		config:   cfg,
	}
}

// Chunk splits the document content into semantically coherent chunks.
func (s *SemanticChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}

	// Split content into sentences
	sentences := splitSentences(content)
	if len(sentences) == 0 {
		return nil, nil
	}

	// Embed all sentences
	embeddings, err := s.embedBatch(sentences)
	if err != nil {
		return nil, err
	}

	// Compute similarity between adjacent sentences
	similarity, err := s.computeSimilarity(embeddings)
	if err != nil {
		return nil, err
	}

	// Find split points based on similarity threshold
	splitPoints := s.findSplitPoints(sentences, similarity)

	// Build chunks from split points
	chunks := s.buildChunks(doc, sentences, splitPoints)

	return chunks, nil
}

// splitSentences splits text into sentences using common delimiters.
func splitSentences(text string) []string {
	var sentences []string
	current := ""

	for i, char := range text {
		current += string(char)

		if char == '.' || char == '!' || char == '?' {
			// Check if this is the end of the string or followed by whitespace
			if i == len(text)-1 {
				s := strings.TrimSpace(current)
				if s != "" {
					sentences = append(sentences, s)
				}
				current = ""
			} else {
				nextChar := text[i+1]
				if nextChar == ' ' || nextChar == '\n' || nextChar == '\r' || nextChar == '\t' {
					s := strings.TrimSpace(current)
					if s != "" {
						sentences = append(sentences, s)
					}
					current = ""
				}
			}
		}
	}

	// Add any remaining text
	if current != "" {
		s := strings.TrimSpace(current)
		if s != "" {
			sentences = append(sentences, s)
		}
	}

	return sentences
}

// embedBatch embeds a batch of sentences using the configured embedder.
func (s *SemanticChunker) embedBatch(sentences []string) ([][]float32, error) {
	type batchEmbedder interface {
		EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	}

	if be, ok := s.embedder.(batchEmbedder); ok {
		return be.EmbedBatch(context.Background(), sentences)
	}

	embeddings := make([][]float32, len(sentences))
	for i, sentence := range sentences {
		emb, err := s.embedder.Embed(context.Background(), sentence)
		if err != nil {
			return nil, err
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// computeSimilarity computes cosine similarity between adjacent sentence embeddings.
func (s *SemanticChunker) computeSimilarity(embeddings [][]float32) ([]float64, error) {
	if len(embeddings) < 2 {
		return nil, nil
	}

	similarities := make([]float64, len(embeddings)-1)
	for i := 0; i < len(embeddings)-1; i++ {
		similarities[i] = cosineSimilarity(embeddings[i], embeddings[i+1])
	}
	return similarities, nil
}

// findSplitPoints identifies positions where similarity drops below the threshold.
func (s *SemanticChunker) findSplitPoints(sentences []string, similarities []float64) []int {
	if len(similarities) == 0 {
		return nil
	}

	var splitPoints []int
	for i, sim := range similarities {
		if sim < s.config.Threshold {
			splitPoints = append(splitPoints, i+1)
		}
	}
	return splitPoints
}

// buildChunks creates chunks from sentences and split points.
func (s *SemanticChunker) buildChunks(doc *core.Document, sentences []string, splitPoints []int) []*core.Chunk {
	if len(sentences) == 0 {
		return nil
	}

	var chunks []*core.Chunk
	start := 0
	chunkIndex := 0

	for _, splitPoint := range splitPoints {
		chunkSentences := sentences[start:splitPoint]
		chunkText := strings.Join(chunkSentences, ". ")

		if len(chunkText) >= s.config.MinChunkSize && len(chunkText) <= s.config.MaxChunkSize {
			chunk := &core.Chunk{
				ID:          generateChunkID(doc.ID, chunkIndex),
				Content:     chunkText,
				DocumentRef: doc.ID,
				ChunkIndex:  chunkIndex,
				Metadata:    copyMetadata(doc.Metadata),
			}
			chunks = append(chunks, chunk)
			chunkIndex++
		} else if len(chunkText) < s.config.MinChunkSize {
			continue
		} else {
			splitLargeChunk(doc, chunkSentences, &chunks, &chunkIndex)
		}

		if s.config.PreserveOverlap && s.config.OverlapSize > 0 {
			overlapEnd := splitPoint - s.config.OverlapSize
			if overlapEnd > start {
				start = overlapEnd
			} else {
				start = splitPoint
			}
		} else {
			start = splitPoint
		}
	}

	if start < len(sentences) {
		chunkSentences := sentences[start:]
		chunkText := strings.Join(chunkSentences, ". ")

		if len(chunkText) >= s.config.MinChunkSize {
			chunk := &core.Chunk{
				ID:          generateChunkID(doc.ID, chunkIndex),
				Content:     chunkText,
				DocumentRef: doc.ID,
				ChunkIndex:  chunkIndex,
				Metadata:    copyMetadata(doc.Metadata),
			}
			chunks = append(chunks, chunk)
		}
	}

	return chunks
}

// splitLargeChunk splits a large chunk into smaller pieces.
func splitLargeChunk(doc *core.Document, sentences []string, chunks *[]*core.Chunk, chunkIndex *int) {
	mid := len(sentences) / 2
	firstHalf := sentences[:mid]
	secondHalf := sentences[mid:]

	if len(firstHalf) > 0 {
		chunk := &core.Chunk{
			ID:          generateChunkID(doc.ID, *chunkIndex),
			Content:     strings.Join(firstHalf, ". "),
			DocumentRef: doc.ID,
			ChunkIndex:  *chunkIndex,
			Metadata:    copyMetadata(doc.Metadata),
		}
		*chunks = append(*chunks, chunk)
		*chunkIndex++
	}

	if len(secondHalf) > 0 {
		chunk := &core.Chunk{
			ID:          generateChunkID(doc.ID, *chunkIndex),
			Content:     strings.Join(secondHalf, ". "),
			DocumentRef: doc.ID,
			ChunkIndex:  *chunkIndex,
			Metadata:    copyMetadata(doc.Metadata),
		}
		*chunks = append(*chunks, chunk)
		*chunkIndex++
	}
}

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

	return dot / (sqrt(normA) * sqrt(normB))
}

// sqrt computes the square root of a float64.
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}

	z := x / 2.0
	for i := 0; i < 100; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}
