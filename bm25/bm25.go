// Package bm25 implements the BM25 ranking function for keyword search.
package bm25

import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "shall": true,
	"should": true, "can": true, "could": true, "may": true,
	"might": true, "must": true, "i": true, "me": true, "my": true,
	"we": true, "our": true, "you": true, "your": true,
	"he": true, "him": true, "his": true, "she": true, "her": true,
	"it": true, "its": true, "they": true, "them": true, "their": true,
	"what": true, "which": true, "who": true, "this": true, "that": true,
	"these": true, "those": true, "in": true, "on": true, "at": true,
	"to": true, "for": true, "of": true, "with": true, "by": true,
	"from": true, "as": true, "into": true, "and": true, "but": true,
	"or": true, "not": true, "so": true, "if": true, "then": true,
	"than": true, "too": true, "very": true, "just": true,
}

// Config holds BM25 parameters.
type Config struct {
	K1 float64 // Term saturation (default: 1.2)
	B  float64 // Length normalization (default: 0.75)
}

// DefaultConfig returns standard BM25 parameters.
func DefaultConfig() Config { return Config{K1: 1.2, B: 0.75} }

// BM25 implements the BM25 ranking function.
type BM25 struct {
	mu        sync.RWMutex
	k1Param   float64
	bParam    float64
	docLens   map[string]int
	avgDocLen float64
	docFreq   map[string]int
	docCount  int
	postings  map[string]map[string]int
}

// New creates a new BM25 index.
func New(cfg Config) *BM25 {
	if cfg.K1 <= 0 { cfg.K1 = 1.2 }
	if cfg.B < 0 { cfg.B = 0.75 }
	return &BM25{
		k1Param: cfg.K1, bParam: cfg.B,
		docLens: make(map[string]int),
		docFreq: make(map[string]int),
		postings: make(map[string]map[string]int),
	}
}

func tokenize(content string) []string {
	s := strings.ToLower(content)
	var tokens []string
	var cur strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else {
			if cur.Len() > 0 {
				w := cur.String()
				if !stopWords[w] { tokens = append(tokens, w) }
				cur.Reset()
			}
		}
	}
	if cur.Len() > 0 {
		w := cur.String()
		if !stopWords[w] { tokens = append(tokens, w) }
	}
	return tokens
}

// AddDocument adds a document to the BM25 index. Returns tokenized tokens.
func (b *BM25) AddDocument(docID string, content string) []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	tokens := tokenize(content)
	b.docLens[docID] = len(tokens)
	b.docCount++
	totalLen := 0
	for _, l := range b.docLens {
		totalLen += l
	}
	b.avgDocLen = float64(totalLen) / float64(b.docCount)
	tf := make(map[string]int)
	for _, t := range tokens {
		tf[t]++
	}
	for term, count := range tf {
		if b.postings[term] == nil {
			b.postings[term] = make(map[string]int)
		}
		b.postings[term][docID] = count
		b.docFreq[term]++
	}
	return tokens
}

// SearchResult represents a single BM25 result.
type SearchResult struct {
	DocID string
	Score float64
}

// Search returns BM25 scores sorted descending.
func (b *BM25) Search(query string) []SearchResult {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.docCount == 0 {
		return nil
	}
	queryTokens := tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}
	scores := make(map[string]float64)
	for docID := range b.docLens {
		var score float64
		for _, term := range queryTokens {
			tf := 0
			if tp, ok := b.postings[term]; ok {
				tf = tp[docID]
			}
			if tf == 0 {
				continue
			}
			df := b.docFreq[term]
			idf := math.Log(float64(b.docCount+1)/float64(df+1) + 1)
			dl := float64(b.docLens[docID])
			num := float64(tf) * (b.k1Param + 1)
			den := float64(tf) + b.k1Param*(1-b.bParam+b.bParam*dl/b.avgDocLen)
			score += idf * num / den
		}
		if score > 0 {
			scores[docID] = score
		}
	}
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{DocID: docID, Score: score})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results
}

// Count returns the number of indexed documents.
func (b *BM25) Count() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.docCount
}

// RemoveDocument removes a document from the index.
func (b *BM25) RemoveDocument(docID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.docLens, docID)
	for term, dm := range b.postings {
		delete(dm, docID)
		if len(dm) == 0 {
			delete(b.postings, term)
			delete(b.docFreq, term)
		} else {
			b.docFreq[term]--
		}
	}
	b.docCount--
	if b.docCount > 0 {
		totalLen := 0
		for _, l := range b.docLens {
			totalLen += l
		}
		b.avgDocLen = float64(totalLen) / float64(b.docCount)
	} else {
		b.avgDocLen = 0
	}
}
