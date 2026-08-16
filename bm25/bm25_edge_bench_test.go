package bm25

import (
	"strings"
	"testing"
)

func BenchmarkBM25_Search_SingleTerm(b *testing.B) {
	bm := New(DefaultConfig())
	for i := 0; i < 1000; i++ {
		bm.AddDocument(string(rune('a'+i%26)), "the quick brown fox jumps over the lazy dog")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("fox")
	}
}

func BenchmarkBM25_Search_StopWordsOnly(b *testing.B) {
	bm := New(DefaultConfig())
	for i := 0; i < 1000; i++ {
		bm.AddDocument(string(rune('a'+i%26)), "the quick brown fox jumps over the lazy dog")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("the is are was")
	}
}

func BenchmarkBM25_Search_VeryLongDocument(b *testing.B) {
	bm := New(DefaultConfig())
	longDoc := strings.Repeat("the quick brown fox jumps over the lazy dog and runs through the forest ", 100)
	for i := 0; i < 100; i++ {
		bm.AddDocument(string(rune('a'+i%26)), longDoc)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("quick fox lazy")
	}
}

func BenchmarkBM25_AddDocument_VeryLong(b *testing.B) {
	bm := New(DefaultConfig())
	longDoc := strings.Repeat("the quick brown fox jumps over the lazy dog and runs through the forest ", 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.AddDocument("doc", longDoc)
	}
}

func BenchmarkBM25_RemoveDocument(b *testing.B) {
	bm := New(DefaultConfig())
	for i := 0; i < 100; i++ {
		bm.AddDocument(string(rune('a'+i%26)), "the quick brown fox jumps over the lazy dog")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.RemoveDocument(string(rune('a' + i%26)))
	}
}

func BenchmarkBM25_Search_NoMatches(b *testing.B) {
	bm := New(DefaultConfig())
	for i := 0; i < 1000; i++ {
		bm.AddDocument(string(rune('a'+i%26)), "the quick brown fox jumps over the lazy dog")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("xyznonexistent")
	}
}

func BenchmarkBM25_Search_MultiTerm(b *testing.B) {
	bm := New(DefaultConfig())
	for i := 0; i < 1000; i++ {
		bm.AddDocument(string(rune('a'+i%26)), "the quick brown fox jumps over the lazy dog")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bm.Search("quick brown fox lazy dog jumps")
	}
}
