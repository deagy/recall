package loader

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDirectoryLoaderMixed(t *testing.T) {
	dir := t.TempDir()
	writeTempIn(t, dir, "a.txt", "alpha")
	writeTempIn(t, dir, "b.md", "# H\n\nbody\n")
	writeTempIn(t, dir, "c.json", `[{"content": "json!"}]`)
	writeTempIn(t, dir, "skip.pdf", "irrelevant")
	writeTempIn(t, filepath.Join(dir, "sub"), "d.txt", "delta")

	d, err := NewDirectoryLoader([]string{".txt", ".md", ".json"}, true, nil)
	require.NoError(t, err)
	docs, err := d.Load(context.Background(), dir)
	require.NoError(t, err)
	require.Len(t, docs, 4) // b.md yields 1 section, c.json 1 item

	var ids []string
	for _, doc := range docs {
		ids = append(ids, doc.ID)
	}
	assert.True(t, contains(ids, filepath.Join(dir, "a.txt")))
	assert.True(t, contains(ids, filepath.Join(dir, "b.md")+"#h"))
	assert.True(t, contains(ids, filepath.Join(dir, "c.json")+":0"))
	assert.True(t, contains(ids, filepath.Join(dir, "sub", "d.txt")))
}

func TestDirectoryLoaderNonRecursive(t *testing.T) {
	dir := t.TempDir()
	writeTempIn(t, dir, "top.txt", "top")
	writeTempIn(t, filepath.Join(dir, "sub"), "inner.txt", "inner")

	d, err := NewDirectoryLoader([]string{".txt"}, false, nil)
	require.NoError(t, err)
	docs, err := d.Load(context.Background(), dir)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "top", docs[0].Title)
}

func TestDirectoryLoaderPartialFailure(t *testing.T) {
	dir := t.TempDir()
	writeTempIn(t, dir, "good.txt", "good")
	writeTempIn(t, dir, "empty.txt", "") // TextLoader rejects empty files

	d, err := NewDirectoryLoader([]string{".txt"}, true, nil)
	require.NoError(t, err)
	docs, err := d.Load(context.Background(), dir)
	require.Error(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "good", docs[0].Content)
	assert.Contains(t, err.Error(), "empty.txt")
}

func TestDirectoryLoaderUnsupportedRegisteredExt(t *testing.T) {
	dir := t.TempDir()
	writeTempIn(t, dir, "x.foo", "data")
	d, err := NewDirectoryLoader([]string{"foo"}, true, nil)
	require.NoError(t, err)
	docs, err := d.Load(context.Background(), dir)
	require.Error(t, err)
	assert.Nil(t, docs)
	assert.Contains(t, err.Error(), "no default loader")
}

func TestDirectoryLoaderCustomLoader(t *testing.T) {
	dir := t.TempDir()
	writeTempIn(t, dir, "x.json", `[{"content": "via custom"}]`)
	// A TextLoader pretending to handle .json: proves the map wins over ForExtension.
	d, err := NewDirectoryLoader([]string{".json"}, true, map[string]Loader{".json": &TextLoader{}})
	require.NoError(t, err)
	docs, err := d.Load(context.Background(), dir)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.True(t, strings.HasPrefix(docs[0].Content, "["))
}

func TestDirectoryLoaderConfigValidation(t *testing.T) {
	_, err := NewDirectoryLoader(nil, true, nil)
	require.Error(t, err)
	_, err = NewDirectoryLoader([]string{".txt"}, true, map[string]Loader{".txt": nil})
	require.Error(t, err)
}

func TestDirectoryLoaderErrors(t *testing.T) {
	d, err := NewDirectoryLoader([]string{".txt"}, true, nil)
	require.NoError(t, err)

	// Missing directory.
	_, err = d.Load(context.Background(), "/no/such/dir")
	require.Error(t, err)

	// A file, not a directory.
	file := writeTemp(t, "file.txt", "x")
	_, err = d.Load(context.Background(), file)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not a directory")

	// No matches.
	empty := t.TempDir()
	_, err = d.Load(context.Background(), empty)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no matching files")

	// Canceled context.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = d.Load(ctx, empty)
	require.ErrorIs(t, err, context.Canceled)
}

func TestLoadersImplementInterface(t *testing.T) {
	var _ Loader = &TextLoader{}
	var _ Loader = &MarkdownLoader{}
	var _ Loader = &CSVLoader{}
	var _ Loader = &JSONLoader{}
	var _ Loader = &DirectoryLoader{}
	_ = errors.Join
}

func writeTempIn(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}

func contains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}
