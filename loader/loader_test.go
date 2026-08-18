package loader

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestForExtension(t *testing.T) {
	for ext, wantType := range map[string]string{
		".txt":      "*loader.TextLoader",
		".TEXT":     "*loader.TextLoader",
		".md":       "*loader.MarkdownLoader",
		".markdown": "*loader.MarkdownLoader",
		".csv":      "*loader.CSVLoader",
		".json":     "*loader.JSONLoader",
	} {
		ld, err := ForExtension(ext)
		require.NoError(t, err)
		assert.Equal(t, wantType, reflect.TypeOf(ld).String())
	}
	_, err := ForExtension(".pdf")
	require.Error(t, err)
	assert.IsType(t, &UnsupportedExtError{}, err)
}

func TestSlug(t *testing.T) {
	assert.Equal(t, "getting-started", slug("Getting Started"))
	assert.Equal(t, "a-b", slug("a  b"))
	assert.Equal(t, "hello-world", slug("# Hello, World! #"))
	assert.Equal(t, "x", slug("x"))
	assert.Equal(t, "", slug("!!!"))
}

func TestBaseName(t *testing.T) {
	assert.Equal(t, "notes", baseName("/tmp/notes.txt"))
	assert.Equal(t, "noext", baseName("noext"))
	// Dotfiles have no recognizable extension, so the name is kept as-is.
	assert.Equal(t, ".dotfile", baseName(".dotfile"))
}
