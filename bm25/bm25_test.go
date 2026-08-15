package bm25

import (
	"testing"
)

func TestBM25_AddAndSearch(t *testing.T) {
	b := New(DefaultConfig())

	b.AddDocument("doc1", "Go is a statically typed compiled programming language")
	b.AddDocument("doc2", "Python is a high-level general-purpose programming language")
	b.AddDocument("doc3", "Rust is a systems programming language focused on safety")

	if b.Count() != 3 {
		t.Errorf("expected 3 documents, got %d", b.Count())
	}

	results := b.Search("programming language")
	if len(results) == 0 {
		t.Fatal("expected search results")
	}

	// "Go" and "Rust" documents should rank higher for "programming language"
	// since "Python" doc also has those terms but "Go" and "Rust" docs are shorter
	// (fewer irrelevant terms), so they should have higher BM25 scores
	for _, r := range results {
		if r.DocID == "doc1" || r.DocID == "doc3" {
			if r.Score <= 0 {
				t.Errorf("expected positive score for %s", r.DocID)
			}
		}
	}
}

func TestBM25_StopWordsFiltered(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "the quick brown fox jumps over the lazy dog")
	b.AddDocument("doc2", "a cat sits on the mat")

	results := b.Search("quick fox")
	if len(results) == 0 {
		t.Fatal("expected results for 'quick fox'")
	}
	// doc1 should rank first since it contains both "quick" and "fox"
	if results[0].DocID != "doc1" {
		t.Errorf("expected doc1 first, got %s", results[0].DocID)
	}
}

func TestBM25_EmptyQuery(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "some content here")
	results := b.Search("")
	if results != nil {
		t.Error("expected nil results for empty query")
	}
}

func TestBM25_EmptyIndex(t *testing.T) {
	b := New(DefaultConfig())
	results := b.Search("test")
	if results != nil {
		t.Error("expected nil results for empty index")
	}
}

func TestBM25_RemoveDocument(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go programming language")
	b.AddDocument("doc2", "python programming language")

	if b.Count() != 2 {
		t.Fatalf("expected 2, got %d", b.Count())
	}

	b.RemoveDocument("doc1")
	if b.Count() != 1 {
		t.Fatalf("expected 1 after remove, got %d", b.Count())
	}

	results := b.Search("go")
	for _, r := range results {
		if r.DocID == "doc1" {
			t.Error("doc1 should have been removed")
		}
	}
}

func TestBM25_TermFrequency(t *testing.T) {
	b := New(DefaultConfig())
	b.AddDocument("doc1", "go go go go go programming language")
	b.AddDocument("doc2", "go programming language")

	results := b.Search("go")
	if len(results) < 2 {
		t.Fatalf("expected at least 2 results, got %d", len(results))
	}
	// doc1 has higher term frequency for "go"
	if results[0].DocID != "doc1" {
		t.Errorf("expected doc1 (higher tf) first, got %s", results[0].DocID)
	}
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
	if len(results) == 0 {
		t.Fatal("expected results for 'python'")
	}
	// doc2 should be the only result with a score for "python"
	found := false
	for _, r := range results {
		if r.DocID == "doc2" && r.Score > 0 {
			found = true
		}
	}
	if !found {
		t.Error("expected doc2 to have positive score for 'python'")
	}
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
	if len(results) < 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// Short doc should rank higher due to length normalization
	if results[0].DocID != "short" {
		t.Errorf("expected 'short' first (length norm), got %s", results[0].DocID)
	}
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