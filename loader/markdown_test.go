package loader

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const sampleMarkdown = `Welcome to the guide.

# Alpha

Intro to alpha.

## Beta

Beta body.

### Beta Deep

Deep body.

## Gamma

Gamma body.

`

func TestMarkdownLoaderSections(t *testing.T) {
	path := writeTemp(t, "guide.md", sampleMarkdown)
	docs, err := (&MarkdownLoader{}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 5)

	assert.Equal(t, "intro", docs[0].Title)
	assert.Equal(t, path+"#intro", docs[0].ID)

	assert.Equal(t, "Alpha", docs[1].Title)
	assert.Equal(t, path+"#alpha", docs[1].ID)
	assert.Equal(t, "Intro to alpha.", docs[1].Content)
	assert.Equal(t, "Alpha", core.ToString(docs[1].Metadata["heading"]))

	assert.Equal(t, "Beta", docs[2].Title)
	assert.Equal(t, path+"#alpha-beta", docs[2].ID)
	assert.Equal(t, "Alpha > Beta", core.ToString(docs[2].Metadata["section_path"]))
	got, ok := core.ToFloat64(docs[2].Metadata["level"])
	require.True(t, ok)
	assert.Equal(t, 2.0, got)

	// Deep section keeps the full breadcrumb.
	assert.Equal(t, path+"#alpha-beta-beta-deep", docs[3].ID)
	assert.Equal(t, "Alpha > Beta > Beta Deep", core.ToString(docs[3].Metadata["section_path"]))

	// Back up one level.
	assert.Equal(t, path+"#alpha-gamma", docs[4].ID)
	assert.Equal(t, "Alpha > Gamma", core.ToString(docs[4].Metadata["section_path"]))

	// Preamble carries no level attribute.
	_, hasLevel := docs[0].Metadata["level"]
	assert.False(t, hasLevel)
}

func TestMarkdownLoaderIncludeHeading(t *testing.T) {
	path := writeTemp(t, "h.md", "# Top\n\nbody\n")
	docs, err := (&MarkdownLoader{IncludeHeading: true}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "# Top\nbody", docs[0].Content)
}

func TestMarkdownLoaderDuplicateHeadings(t *testing.T) {
	path := writeTemp(t, "dup.md", "# One\n\na\n# One\n\nb\n# One\n\nc\n")
	docs, err := (&MarkdownLoader{}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 3)
	assert.Equal(t, path+"#one", docs[0].ID)
	assert.Equal(t, path+"#one-2", docs[1].ID)
	assert.Equal(t, path+"#one-3", docs[2].ID)
}

func TestMarkdownLoaderEmptySectionsDropped(t *testing.T) {
	path := writeTemp(t, "sparse.md", "# A\n\n# B\n\nbody\n\n# C\n")
	docs, err := (&MarkdownLoader{}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "B", docs[0].Title)
}

func TestMarkdownLoaderEmptyFile(t *testing.T) {
	path := writeTemp(t, "empty.md", "")
	_, err := (&MarkdownLoader{}).Load(context.Background(), path)
	require.Error(t, err)
}

func TestParseHeading(t *testing.T) {
	cases := []struct {
		line  string
		title string
		level int
		ok    bool
	}{
		{"# Title", "Title", 1, true},
		{"   ### Three spaces", "Three spaces", 3, true},
		{"#### No Space", "No Space", 4, true},
		{"####NoSpace", "", 0, false},
		{"####### seven", "", 0, false},
		{"####\t tab indent", "", 0, false},
		{"# closing #", "closing", 1, true},
		{"# a # b #", "a # b", 1, true},
		{"#tag", "", 0, false},
		{"just text", "", 0, false},
		{"#   spaced   ", "spaced", 1, true},
	}
	for _, c := range cases {
		title, level, ok := parseHeading(c.line)
		assert.Equal(t, c.ok, ok, "parseHeading(%q)", c.line)
		if ok {
			assert.Equal(t, c.title, title, "parseHeading(%q) title", c.line)
			assert.Equal(t, c.level, level, "parseHeading(%q) level", c.line)
		}
	}
}
