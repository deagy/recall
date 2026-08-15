package index

import (
	"testing"
	"time"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
)

func TestTermFilter_Match(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"source": core.String{Value: "test.txt"},
		},
	}

	f := &TermFilter{Key: "source", Value: "test.txt"}
	assert.True(t, f.Match(chunk), "expected match")

	f2 := &TermFilter{Key: "source", Value: "other.txt"}
	assert.False(t, f2.Match(chunk), "expected no match")
}

func TestTermInFilter_Match(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"tag": core.String{Value: "go"},
		},
	}

	f := &TermInFilter{Key: "tag", Values: []string{"go", "rust"}}
	assert.True(t, f.Match(chunk), "expected match")

	f2 := &TermInFilter{Key: "tag", Values: []string{"python", "java"}}
	assert.False(t, f2.Match(chunk), "expected no match")
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
	assert.True(t, f.Match(chunk), "expected match (42 in [10, 50])")

	f2 := &RangeFilter{Key: "score", Min: &min, Max: &max, MinIncl: false, MaxIncl: false}
	assert.True(t, f2.Match(chunk), "expected match (42 in (10, 50))")

	// Value at boundary with exclusive min - 42 > 10, so it matches
	f3 := &RangeFilter{Key: "score", Min: &min, MinIncl: false}
	assert.True(t, f3.Match(chunk), "expected match (42 > 10 with exclusive min)")

	// Value at min boundary with exclusive min - 10 is not > 10
	valAtMin := 10.0
	chunkAtMin := &core.Chunk{
		Metadata: map[string]core.Value{"score": core.Number{Value: valAtMin}},
	}
	f4 := &RangeFilter{Key: "score", Min: &min, MinIncl: false}
	assert.False(t, f4.Match(chunkAtMin), "expected no match (10 not > 10 with exclusive min)")

	// Missing key
	f5 := &RangeFilter{Key: "missing", Min: &min}
	assert.False(t, f5.Match(chunk), "expected no match for missing key")

	// Non-numeric value
	chunk2 := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.String{Value: "not-a-number"},
		},
	}
	assert.False(t, f.Match(chunk2), "expected no match for non-numeric value")
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
	assert.True(t, f.Match(chunk), "expected match")

	minFar := now.Add(-24 * time.Hour)
	maxFar := now.Add(-12 * time.Hour)
	f2 := &DateRangeFilter{Key: "date", Min: &minFar, Max: &maxFar}
	assert.False(t, f2.Match(chunk), "expected no match")
}

func TestTermFilter_Match_NilMetadata(t *testing.T) {
	chunk := &core.Chunk{}
	f := &TermFilter{Key: "source", Value: "test.txt"}
	assert.False(t, f.Match(chunk), "expected no match for nil metadata")
}

func TestTermInFilter_Match_NilMetadata(t *testing.T) {
	chunk := &core.Chunk{}
	f := &TermInFilter{Key: "tag", Values: []string{"go"}}
	assert.False(t, f.Match(chunk), "expected no match for nil metadata")
}

func TestTermInFilter_Match_EmptyValues(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"tag": core.String{Value: "go"},
		},
	}
	f := &TermInFilter{Key: "tag", Values: []string{}}
	assert.False(t, f.Match(chunk), "expected no match for empty values")
}

func TestRangeFilter_Match_NilMetadata(t *testing.T) {
	chunk := &core.Chunk{}
	min := 10.0
	f := &RangeFilter{Key: "score", Min: &min}
	assert.False(t, f.Match(chunk), "expected no match for nil metadata")
}

func TestRangeFilter_Match_ZeroValue(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 0},
		},
	}
	min := 0.0
	max := 100.0
	f := &RangeFilter{Key: "score", Min: &min, Max: &max, MinIncl: true, MaxIncl: true}
	assert.True(t, f.Match(chunk), "expected match for zero value")
}

func TestRangeFilter_Match_ExclusiveMin(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 10},
		},
	}
	min := 10.0
	f := &RangeFilter{Key: "score", Min: &min, MinIncl: false}
	assert.False(t, f.Match(chunk), "expected no match for exclusive min at boundary")
}

func TestRangeFilter_Match_ExclusiveMax(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 50},
		},
	}
	max := 50.0
	f := &RangeFilter{Key: "score", Max: &max, MaxIncl: false}
	assert.False(t, f.Match(chunk), "expected no match for exclusive max at boundary")
}

func TestDateRangeFilter_Match_NilMetadata(t *testing.T) {
	chunk := &core.Chunk{}
	now := time.Now()
	min := now.Add(-time.Hour)
	max := now.Add(time.Hour)
	f := &DateRangeFilter{Key: "date", Min: &min, Max: &max}
	assert.False(t, f.Match(chunk), "expected no match for nil metadata")
}

func TestDateRangeFilter_Match_InvalidFormat(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"date": core.String{Value: "not-a-date"},
		},
	}
	now := time.Now()
	min := now.Add(-time.Hour)
	max := now.Add(time.Hour)
	f := &DateRangeFilter{Key: "date", Min: &min, Max: &max}
	assert.False(t, f.Match(chunk), "expected no match for invalid date format")
}

func TestDateRangeFilter_Match_DateFormat(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"date": core.String{Value: "2024-01-15"},
		},
	}
	min := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	max := time.Date(2024, 12, 31, 0, 0, 0, 0, time.UTC)
	f := &DateRangeFilter{Key: "date", Min: &min, Max: &max}
	assert.True(t, f.Match(chunk), "expected match for date in range")
}

func TestDateRangeFilter_Match_ExclusiveMin(t *testing.T) {
	now := time.Now()
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"date": core.String{Value: now.Add(-30 * time.Minute).Format(time.RFC3339)},
		},
	}
	min := now
	f := &DateRangeFilter{Key: "date", Min: &min, MinIncl: false}
	assert.False(t, f.Match(chunk), "expected no match for exclusive min")
}

func TestDateRangeFilter_Match_ExclusiveMax(t *testing.T) {
	now := time.Now()
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"date": core.String{Value: now.Add(30 * time.Minute).Format(time.RFC3339)},
		},
	}
	max := now
	f := &DateRangeFilter{Key: "date", Max: &max, MaxIncl: false}
	assert.False(t, f.Match(chunk), "expected no match for exclusive max")
}

func TestTermFilter_Match_EmptyValue(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"source": core.String{Value: ""},
		},
	}
	f := &TermFilter{Key: "source", Value: ""}
	assert.True(t, f.Match(chunk), "expected match for empty value")
}

func TestTermInFilter_Match_EmptyKey(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"": core.String{Value: "go"},
		},
	}
	f := &TermInFilter{Key: "", Values: []string{"go"}}
	assert.True(t, f.Match(chunk), "expected match for empty key")
}

func TestRangeFilter_Match_OnlyMin(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 100},
		},
	}
	min := 50.0
	f := &RangeFilter{Key: "score", Min: &min}
	assert.True(t, f.Match(chunk), "expected match with only min")
}

func TestRangeFilter_Match_OnlyMax(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 10},
		},
	}
	max := 50.0
	f := &RangeFilter{Key: "score", Max: &max}
	assert.True(t, f.Match(chunk), "expected match with only max")
}

func TestRangeFilter_Match_NegativeValue(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: -10},
		},
	}
	min := -50.0
	max := 50.0
	f := &RangeFilter{Key: "score", Min: &min, Max: &max}
	assert.True(t, f.Match(chunk), "expected match for negative value")
}

func TestRangeFilter_Match_LargeValue(t *testing.T) {
	chunk := &core.Chunk{
		Metadata: map[string]core.Value{
			"score": core.Number{Value: 1e10},
		},
	}
	min := 0.0
	max := 1e12
	f := &RangeFilter{Key: "score", Min: &min, Max: &max}
	assert.True(t, f.Match(chunk), "expected match for large value")
}
