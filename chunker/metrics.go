package chunker

import (
	"math"
	"sync"
	"time"
)

// ChunkMetrics tracks quality and performance metrics for chunking operations.
type ChunkMetrics struct {
	mu sync.Mutex

	// TotalChunks is the total number of chunks created.
	TotalChunks int

	// TotalTokens is the total number of tokens processed.
	TotalTokens int

	// ChunkSizes tracks the size (in characters) of each chunk.
	ChunkSizes []int

	// ProcessingTimes tracks the time taken to process each chunk.
	ProcessingTimes []time.Duration

	// CoherenceScores tracks the semantic coherence of each chunk.
	CoherenceScores []float64

	// StartTime is when the first chunk was created.
	StartTime time.Time

	// EndTime is when the last chunk was created.
	EndTime time.Time

	// Errors is the number of errors encountered during chunking.
	Errors int
}

// NewChunkMetrics creates a new ChunkMetrics instance.
func NewChunkMetrics() *ChunkMetrics {
	return &ChunkMetrics{
		ChunkSizes:      make([]int, 0),
		ProcessingTimes: make([]time.Duration, 0),
		CoherenceScores: make([]float64, 0),
	}
}

// RecordChunk records metrics for a chunk.
func (m *ChunkMetrics) RecordChunk(chunkSize int, processingTime time.Duration, coherence float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalChunks++
	m.TotalTokens += chunkSize / 4 // rough estimate: 1 token ≈ 4 chars
	m.ChunkSizes = append(m.ChunkSizes, chunkSize)
	m.ProcessingTimes = append(m.ProcessingTimes, processingTime)
	m.CoherenceScores = append(m.CoherenceScores, coherence)

	if m.StartTime.IsZero() {
		m.StartTime = time.Now()
	}
	m.EndTime = time.Now()
}

// RecordError records an error encountered during chunking.
func (m *ChunkMetrics) RecordError() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Errors++
}

// Summary returns a summary of the chunking metrics.
func (m *ChunkMetrics) Summary() ChunkSummary {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.TotalChunks == 0 {
		return ChunkSummary{
			TotalChunks:    0,
			TotalTokens:    0,
			AvgChunkSize:   0,
			MinChunkSize:   0,
			MaxChunkSize:   0,
			AvgCoherence:   0,
			ProcessingTime: 0,
			Errors:         0,
		}
	}

	var totalSize, minSize, maxSize int
	var totalCoherence float64
	var totalTime time.Duration

	for i, size := range m.ChunkSizes {
		totalSize += size
		if i == 0 || size < minSize {
			minSize = size
		}
		if size > maxSize {
			maxSize = size
		}
	}

	for _, coherence := range m.CoherenceScores {
		totalCoherence += coherence
	}

	for _, t := range m.ProcessingTimes {
		totalTime += t
	}

	return ChunkSummary{
		TotalChunks:    m.TotalChunks,
		TotalTokens:    m.TotalTokens,
		AvgChunkSize:   float64(totalSize) / float64(m.TotalChunks),
		MinChunkSize:   minSize,
		MaxChunkSize:   maxSize,
		AvgCoherence:   totalCoherence / float64(len(m.CoherenceScores)),
		ProcessingTime: totalTime,
		Errors:         m.Errors,
	}
}

// ChunkSummary provides a summary of chunking metrics.
type ChunkSummary struct {
	TotalChunks    int
	TotalTokens    int
	AvgChunkSize   float64
	MinChunkSize   int
	MaxChunkSize   int
	AvgCoherence   float64
	ProcessingTime time.Duration
	Errors         int
}

// ChunkCoherence computes the semantic coherence of a chunk based on
// the similarity of its sentences.
func ChunkCoherence(sentences [][]float32) float64 {
	if len(sentences) < 2 {
		return 1.0 // Single sentence is perfectly coherent
	}

	var totalSimilarity float64
	for i := 0; i < len(sentences)-1; i++ {
		sim := cosineSimilarity(sentences[i], sentences[i+1])
		totalSimilarity += sim
	}

	return totalSimilarity / float64(len(sentences)-1)
}

// ChunkInfo provides information about a chunk's quality.
type ChunkInfo struct {
	Size       int
	Tokens     int
	Coherence  float64
	IsCoherent bool
}

// AnalyzeChunk analyzes a chunk and returns quality information.
func AnalyzeChunk(content string, coherence float64) ChunkInfo {
	size := len(content)
	tokens := size / 4 // rough estimate

	return ChunkInfo{
		Size:       size,
		Tokens:     tokens,
		Coherence:  coherence,
		IsCoherent: coherence >= 0.7,
	}
}

// NormalizeCoherence normalizes a coherence score to the range [0, 1].
func NormalizeCoherence(score float64) float64 {
	if score < -1 {
		score = -1
	}
	if score > 1 {
		score = 1
	}
	return (score + 1) / 2
}

// Variance computes the variance of a slice of float64 values.
func Variance(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	mean := Mean(data)
	var sum float64
	for _, v := range data {
		diff := v - mean
		sum += diff * diff
	}

	return sum / float64(len(data))
}

// Mean computes the mean of a slice of float64 values.
func Mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	var sum float64
	for _, v := range data {
		sum += v
	}

	return sum / float64(len(data))
}

// StdDev computes the standard deviation of a slice of float64 values.
func StdDev(data []float64) float64 {
	return math.Sqrt(Variance(data))
}
