package index

import (
	"testing"
	"time"

	"github.com/deagy/recall/core"
)

func TestTermFilter_Match(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"source": core.String{Value: "test.txt"},
		},
	}

	f := &TermFilter{Key: "source", Value: "test.txt"}
	if !f.Match(chunk) {
		t.Error("expected match")
	}

	f2 := &TermFilter{Key: "source", Value: "other.txt"}
	if f2.Match(chunk) {
		t.Error("expected no match")
	}
}

func TestTermInFilter_Match(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"tag": core.String{Value: "go"},
		},
	}

	f := &TermInFilter{Key: "tag", Values: []string{"go", "rust"}}
	if !f.Match(chunk) {
		t.Error("expected match")
	}

	f2 := &TermInFilter{Key: "tag", Values: []string{"python", "java"}}
	if f2.Match(chunk) {
		t.Error("expected no match")
	}
}

func TestRangeFilter_Match(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 42},
		},
	}

	min := 10.0
	max := 50.0

	f := &RangeFilter{Key: "score", Min: &min, Max: &max, MinIncl: true, MaxIncl: true}
	if !f.Match(chunk) {
		t.Error("expected match (42 in [10, 50])")
	}

	f2 := &RangeFilter{Key: "score", Min: &min, Max: &max, MinIncl: false, MaxIncl: false}
	if !f2.Match(chunk) {
		t.Error("expected match (42 in (10, 50))")
	}

	// Value at boundary with exclusive min - 42 > 10, so it matches
	f3 := &RangeFilter{Key: "score", Min: &min, MinIncl: false}
	if !f3.Match(chunk) {
		t.Error("expected match (42 > 10 with exclusive min)")
	}

	// Value at min boundary with exclusive min - 10 is not > 10
	valAtMin := 10.0
	chunkAtMin := &core.Chunk{
		Metadata: map[string]core.Value{"score": core.Number{Value: valAtMin}},
	}
	f4 := &RangeFilter{Key: "score", Min: &min, MinIncl: false}
	if f4.Match(chunkAtMin) {
		t.Error("expected no match (10 not > 10 with exclusive min)")
	}

	// Missing key
	f5 := &RangeFilter{Key: "missing", Min: &min}
	if f5.Match(chunk) {
		t.Error("expected no match for missing key")
	}

	// Non-numeric value
	chunk2 := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.String{Value: "not-a-number"},
		},
	}
	if f.Match(chunk2) {
		t.Error("expected no match for non-numeric value")
	}
}

func TestDateRangeFilter_Match(t *testing.T) {
	now := time.Now()
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"date": core.String{Value: now.Format(time.RFC3339)},
		},
		CreatedAt: now,
	}

	min := now.Add(-time.Hour)
	max := now.Add(time.Hour)

	f := &DateRangeFilter{Key: "date", Min: &min, Max: &max, MinIncl: true, MaxIncl: true}
	if !f.Match(chunk) {
		t.Error("expected match")
	}

	minFar := now.Add(-24 * time.Hour)
	maxFar := now.Add(-12 * time.Hour)
	f2 := &DateRangeFilter{Key: "date", Min: &minFar, Max: &maxFar}
	if f2.Match(chunk) {
		t.Error("expected no match")
	}
}
