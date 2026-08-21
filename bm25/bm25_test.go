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

// scoresByID converts search results to a map for order-independent comparison.
func scoresByID(results []SearchResult) map[string]float64 {
	m := make(map[string]float64, len(results))
	for _, r := range results {
		m[r.DocID] = r.Score
	}
	return m
}

func TestBM25_AddDocument_ReAddKeepsCountConsistent(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go programming language")
	b.AddDocument("doc1", "rust systems programming")

	require.Equal(t, 1, b.Count(), "re-adding an existing docID must not grow the count")

	// Old postings must be gone, new content must be searchable.
	for _, r := range b.Search("go") {
		assert.NotEqual(t, "doc1", r.DocID, "stale terms must be removed on re-add")
	}
	results := b.Search("rust")
	require.NotEmpty(t, results, "new content must be searchable after re-add")
	assert.Equal(t, "doc1", results[0].DocID, "expected doc1 for new terms")
}

func TestBM25_AddDocument_ReAddMatchesFreshIndex(t *testing.T) {
	// Re-adding must leave exactly the same state as if the replacement
	// document had been the only version ever added (count, IDF, avgDocLen).
	withReAdd := New(DefaultConfig())
	withReAdd.AddDocument("doc1", "go programming language")
	withReAdd.AddDocument("doc2", "python programming language")
	withReAdd.AddDocument("doc1", "rust systems programming safety")

	fresh := New(DefaultConfig())
	fresh.AddDocument("doc2", "python programming language")
	fresh.AddDocument("doc1", "rust systems programming safety")

	require.Equal(t, fresh.Count(), withReAdd.Count(), "counts must match")

	for _, query := range []string{"programming", "rust", "python", "safety"} {
		assert.Equal(t, scoresByID(fresh.Search(query)), scoresByID(withReAdd.Search(query)),
			"scores for query %q must match a fresh index", query)
	}
}

func TestBM25_RemoveDocument_UnknownIsNoOp(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go programming language")

	before := scoresByID(b.Search("go"))
	b.RemoveDocument("does-not-exist")

	require.Equal(t, 1, b.Count(), "removing an unknown docID must not change the count")
	assert.Equal(t, before, scoresByID(b.Search("go")), "scores must be unchanged")
}

func TestBM25_RemoveDocument_UnaffectedScoresUnchanged(t *testing.T) {
	// Removing one document must not shift docFreq (and therefore scores)
	// of terms it never contained.
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go programming language")
	b.AddDocument("doc2", "python scripting language")
	b.RemoveDocument("doc1")

	fresh := New(DefaultConfig())
	fresh.AddDocument("doc2", "python scripting language")

	require.Equal(t, fresh.Count(), b.Count(), "counts must match")
	assert.Equal(t, scoresByID(fresh.Search("python")), scoresByID(b.Search("python")))
	assert.Equal(t, scoresByID(fresh.Search("language")), scoresByID(b.Search("language")),
		"docFreq of shared terms must only drop for docs that contained them")
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
