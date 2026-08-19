package eval

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDataset_SaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ds.json")

	ds := NewDataset("unit-test")
	ds.Version = "1"
	ds.Add(EvalQuery{
		Query:       "Capital of France?",
		RelevantIDs: []string{"c1", "c2"},
		Context:     "Paris is the capital of France.",
	})
	ds.Add(EvalQuery{
		ID:        "graded",
		Query:     "Who wrote it?",
		Relevance: map[string]int{"c3": 3, "c4": 1},
	})

	require.NoError(t, ds.Save(path))
	assert.Equal(t, 2, ds.Len())
	// ID should be filled from the query for the first one.
	assert.Equal(t, "Capital of France?", ds.Queries[0].ID)

	loaded, err := LoadDataset(path)
	require.NoError(t, err)
	assert.Equal(t, "unit-test", loaded.Name)
	assert.Equal(t, "1", loaded.Version)
	require.Len(t, loaded.Queries, 2)
	assert.Equal(t, []string{"c1", "c2"}, loaded.Queries[0].RelevantIDs)
	assert.Equal(t, "Paris is the capital of France.", loaded.Queries[0].Context)
	assert.Equal(t, map[string]int{"c3": 3, "c4": 1}, loaded.Queries[1].Relevance)
}

func TestLoadDataset_Missing(t *testing.T) {
	_, err := LoadDataset(filepath.Join(t.TempDir(), "nope.json"))
	require.Error(t, err)
}
