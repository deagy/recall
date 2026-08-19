package testutil_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/deagy/recall/index"
	"github.com/deagy/recall/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChunkID(t *testing.T) {
	assert.Equal(t, "d::chunk-0", testutil.ChunkID("d", 0))
	assert.Equal(t, "d::chunk-42", testutil.ChunkID("d", 42))
}

func TestNewFixtureStore_SingleChunks(t *testing.T) {
	f, err := testutil.NewFixtureStore(
		testutil.FixtureDoc{ID: "paris", Title: "Paris", Content: "Paris is the capital of France."},
		testutil.FixtureDoc{ID: "tokyo", Content: "Tokyo is the capital of Japan."},
		// No ID: should default to doc-2.
		testutil.FixtureDoc{Content: "Sydney is the largest city in Australia."},
	)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, f.Close()) })

	assert.Equal(t, 3, f.Store.Count())
	chunk, ok := f.Store.GetChunk(f.FirstChunkID("paris"))
	require.True(t, ok, "predictable chunk ID should exist")
	assert.Equal(t, "paris", chunk.DocumentRef)
	assert.Contains(t, chunk.Content, "capital of France")
	assert.NotNil(t, chunk.Embedding)
	assert.Len(t, chunk.Embedding, testutil.DefaultFixtureDim)

	defaulted, ok := f.Store.GetChunk(testutil.ChunkID("doc-2", 0))
	require.True(t, ok, "default doc ID should be index-based")
	assert.Equal(t, "doc-2", defaulted.DocumentRef)
}

func TestNewFixtureStore_Deterministic(t *testing.T) {
	docs := []testutil.FixtureDoc{
		{ID: "a", Content: "alpha beta gamma"},
		{ID: "b", Content: "delta epsilon zeta"},
	}
	f1, err := testutil.NewFixtureStore(docs...)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, f1.Close()) })
	f2, err := testutil.NewFixtureStore(docs...)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, f2.Close()) })

	r1, err := f1.Store.Search(context.Background(), "alpha", index.SearchOptions{TopK: 2})
	require.NoError(t, err)
	r2, err := f2.Store.Search(context.Background(), "alpha", index.SearchOptions{TopK: 2})
	require.NoError(t, err)
	require.Len(t, r1, 2)
	for i := range r1 {
		assert.Equal(t, r1[i].Chunk.ID, r2[i].Chunk.ID)
		assert.InDelta(t, r1[i].Score, r2[i].Score, 1e-6)
	}
}

func TestNewFixtureStore_UploadError(t *testing.T) {
	_, err := testutil.NewFixtureStore(testutil.FixtureDoc{ID: "x", Content: ""})
	require.Error(t, err)
}

func TestMockEmbedder_Deterministic(t *testing.T) {
	e := testutil.NewMockEmbedder(32)
	assert.Equal(t, 32, e.Dimension())

	v1 := testutil.DeterministicEmbed(e, "hello world")
	v2 := testutil.DeterministicEmbed(e, "hello world")
	assert.Equal(t, v1, v2, "same text must embed identically")
	assert.Len(t, v1, 32)

	var norm float64
	for _, x := range v1 {
		norm += float64(x) * float64(x)
	}
	assert.InDelta(t, 1.0, norm, 1e-5, "embeddings are L2-normalized")

	v3 := testutil.DeterministicEmbed(e, "different text")
	assert.NotEqual(t, v1, v3)
}

// fakeTB is a minimal testing.TB that records failures/skips so tests can
// assert on helper behavior without failing the surrounding test.
type fakeTB struct {
	testing.TB
	failed  bool
	skipped bool
	msgs    []string
}

func (f *fakeTB) Helper() {}

func (f *fakeTB) Errorf(format string, args ...any) {
	f.failed = true
	f.msgs = append(f.msgs, fmt.Sprintf(format, args...))
}

func (f *fakeTB) Fatalf(format string, args ...any) {
	f.failed = true
	f.msgs = append(f.msgs, fmt.Sprintf(format, args...))
}

func (f *fakeTB) Skipf(format string, args ...any) {
	f.skipped = true
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "golden.txt")
	require.NoError(t, os.WriteFile(p, []byte(content), 0o644))
	return p
}

func TestGolden_Update_WritesFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "g.txt")
	old := testutil.UpdateGolden
	testutil.UpdateGolden = true
	defer func() { testutil.UpdateGolden = old }()

	fake := &fakeTB{}
	testutil.Golden(fake, path, "v1")
	assert.True(t, fake.skipped)
	assert.False(t, fake.failed)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, "v1", string(data))
}

func TestGolden_Match(t *testing.T) {
	path := writeTemp(t, "same")
	fake := &fakeTB{}
	testutil.Golden(fake, path, "same")
	assert.False(t, fake.failed)
}

func TestGolden_Mismatch(t *testing.T) {
	path := writeTemp(t, "want")
	fake := &fakeTB{}
	testutil.Golden(fake, path, "got")
	assert.True(t, fake.failed)
	assert.Contains(t, fake.msgs[0], "mismatch")
}

func TestGolden_Missing(t *testing.T) {
	fake := &fakeTB{}
	testutil.Golden(fake, filepath.Join(t.TempDir(), "missing.txt"), "x")
	assert.True(t, fake.failed)
	assert.Contains(t, fake.msgs[0], "update flag")
}

func TestGoldenJSON_RoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "g.json")
	old := testutil.UpdateGolden
	testutil.UpdateGolden = true
	defer func() { testutil.UpdateGolden = old }()
	testutil.GoldenJSON(&fakeTB{}, path, map[string]int{"a": 1, "b": 2})

	// Turn update off so subsequent calls compare instead of rewriting.
	testutil.UpdateGolden = false

	fake := &fakeTB{}
	testutil.GoldenJSON(fake, path, map[string]int{"a": 1, "b": 2})
	assert.False(t, fake.failed)

	bad := &fakeTB{}
	testutil.GoldenJSON(bad, path, map[string]int{"a": 999})
	assert.True(t, bad.failed)
}

func TestGoldenDiff(t *testing.T) {
	assert.Equal(t, "", testutil.GoldenDiff("a\nb", "a\nb"))
	d := testutil.GoldenDiff("a\nb", "a\nc")
	assert.Contains(t, d, "-\tb")
	assert.Contains(t, d, "+\tc")
	// got has more lines than want
	long := testutil.GoldenDiff("a", "a\nextra")
	assert.Contains(t, long, "+\textra")
}

func TestGolden_Update_WriteErrors(t *testing.T) {
	old := testutil.UpdateGolden
	testutil.UpdateGolden = true
	defer func() { testutil.UpdateGolden = old }()

	// MkdirAll fails when the path traverses a regular file.
	file := writeTemp(t, "x")
	tricky := filepath.Join(file, "nested", "g.txt")
	fake1 := &fakeTB{}
	testutil.Golden(fake1, tricky, "v")
	assert.True(t, fake1.failed, "mkdir-all error must fail")

	// WriteFile fails when the destination is an existing directory.
	dir := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(dir, "sub"), 0o755))
	fake2 := &fakeTB{}
	testutil.Golden(fake2, filepath.Join(dir, "sub"), "v")
	assert.True(t, fake2.failed, "write-to-dir must fail")
}

func TestGoldenJSON_MarshalError(t *testing.T) {
	fake := &fakeTB{}
	testutil.GoldenJSON(fake, filepath.Join(t.TempDir(), "g.json"), make(chan int))
	assert.True(t, fake.failed)
}
