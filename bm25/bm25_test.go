package bm25

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBM25_AddAndSearch(t *testing.T) {
	b := New(DefaultConfig())

	b.AddDocument("doc1", "Go is a statically typed compiled programming language")
	b.AddDocument("doc2", "Python is a high-level general-purpose programming language")
	b.AddDocument("doc3", "Rust is a systems programming language focused on safety")

	require.Equal(t, 3, b.Count(), "expected 3 documents")

	results := b.Search("programming language")
	require.NotEmpty(t, results, "expected search results")

	// "Go" and "Rust" documents should rank higher for "programming language"
	// since "Python" doc also has those terms but "Go" and "Rust" docs are shorter
	// (fewer irrelevant terms), so they should have higher BM25 scores
	for _, r := range results {
		if r.DocID == "doc1" || r.DocID == "doc3" {
			assert.Greater(t, r.Score, float64(0), "expected positive score for %s", r.DocID)
		}
	}
}

func TestBM25_StopWordsFiltered(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "the quick brown fox jumps over the lazy dog")
	b.AddDocument("doc2", "a cat sits on the mat")

	results := b.Search("quick fox")
	require.NotEmpty(t, results, "expected results for 'quick fox'")
	// doc1 should rank first since it contains both "quick" and "fox"
	assert.Equal(t, "doc1", results[0].DocID, "expected doc1 first")
}

func TestBM25_EmptyQuery(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "some content here")
	results := b.Search("")
	assert.Nil(t, results, "expected nil results for empty query")
}

func TestBM25_EmptyIndex(t *testing.T) {
	b := New(DefaultConfig())
	results := b.Search("test")
	assert.Nil(t, results, "expected nil results for empty index")
}

func TestBM25_RemoveDocument(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go programming language")
	b.AddDocument("doc2", "python programming language")

	require.Equal(t, 2, b.Count(), "expected 2 documents")

	b.RemoveDocument("doc1")
	require.Equal(t, 1, b.Count(), "expected 1 after remove")

	results := b.Search("go")
	for _, r := range results {
		assert.NotEqual(t, "doc1", r.DocID, "doc1 should have been removed")
	}
}

func TestBM25_TermFrequency(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go go go go go programming language")
	b.AddDocument("doc2", "go programming language")

	results := b.Search("go")
	require.GreaterOrEqual(t, len(results), 2, "expected at least 2 results")
	// doc1 has higher term frequency for "go"
	assert.Equal(t, "doc1", results[0].DocID, "expected doc1 (higher tf) first")
}

func TestBM25_IDF(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go programming language")
	b.AddDocument("doc2", "python programming language")
	b.AddDocument("doc3", "rust programming language")
	b.AddDocument("doc4", "java programming language")
	b.AddDocument("doc5", "go is great")

	// "go" appears in 2 docs, "python" in 1 doc
	// "python" should have higher IDF
	results := b.Search("python")
	require.NotEmpty(t, results, "expected results for 'python'")
	// doc2 should be the only result with a score for "python"
	found := false
	for _, r := range results {
		if r.DocID == "doc2" && r.Score > 0 {
			found = true
		}
	}
	assert.True(t, found, "expected doc2 to have positive score for 'python'")
}

func TestBM25_LengthNormalization(t *testing.T) {
	b := New(DefaultConfig())
	// Short doc with the term
	b.AddDocument("short", "go")
	// Long doc with the term repeated many times
	longContent := "go "
	for i := 0; i < 100; i++ {
		longContent += "irrelevant word "
	}
	longContent += "go"
	b.AddDocument("long", longContent)

	results := b.Search("go")
	require.GreaterOrEqual(t, len(results), 2, "expected 2 results")
	// Short doc should rank higher due to length normalization
	assert.Equal(t, "short", results[0].DocID, "expected 'short' first (length norm)")
}

func BenchmarkBM25_AddDocument(b *testing.B) {
	bm := New(DefaultConfig())
	doc := "the quick brown fox jumps over the lazy dog and runs through the forest"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.AddDocument("doc", doc)
	}
}

func BenchmarkBM25_Search(b *testing.B) {
	bm := New(DefaultConfig())
	docs := make([]string, 1000)
	for i := range docs {
		docs[i] = "the quick brown fox jumps over the lazy dog"
		bm.AddDocument(string(rune('a'+i%26)), docs[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("quick fox")
	}
}

func BenchmarkBM25_SearchLargeCorpus(b *testing.B) {
	bm := New(DefaultConfig())
	docs := make([]string, 10000)
	for i := range docs {
		docs[i] = "the quick brown fox jumps over the lazy dog"
		bm.AddDocument(string(rune('a'+i%26)), docs[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("quick fox lazy")
	}
}
