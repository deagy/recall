package loader

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}

func TestTextLoader(t *testing.T) {
	path := writeTemp(t, "note.txt", "hello world\n")
	docs, err := (&TextLoader{}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, path, docs[0].ID)
	assert.Equal(t, path, docs[0].Source)
	assert.Equal(t, "note", docs[0].Title)
	assert.Equal(t, "hello world\n", docs[0].Content)
}

func TestTextLoaderTruncatesAtMaxBytes(t *testing.T) {
	path := writeTemp(t, "big.txt", "1234567890")
	docs, err := (&TextLoader{MaxBytes: 4}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "1234", docs[0].Content)
}

func TestTextLoaderEmptyFile(t *testing.T) {
	path := writeTemp(t, "empty.txt", "")
	_, err := (&TextLoader{}).Load(context.Background(), path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty")
}

func TestTextLoaderMissingFile(t *testing.T) {
	_, err := (&TextLoader{}).Load(context.Background(), "/definitely/missing.txt")
	require.Error(t, err)
}

func TestTextLoaderCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := (&TextLoader{}).Load(ctx, "whatever")
	require.ErrorIs(t, err, context.Canceled)
}
