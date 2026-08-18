package loader

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONLoaderArray(t *testing.T) {
	path := writeTemp(t, "items.json", `[
		{"id": "a", "content": "first", "count": 2, "live": true},
		{"id": "b", "content": "second"}
	]`)
	docs, err := (&JSONLoader{IDField: "id"}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 2)
	assert.Equal(t, "a", docs[0].ID)
	assert.Equal(t, "first", docs[0].Content)
	assert.Equal(t, "a", docs[0].Title)
	n, ok := core.ToFloat64(docs[0].Metadata["count"])
	require.True(t, ok)
	assert.Equal(t, 2.0, n)
	b, ok := core.ToBool(docs[0].Metadata["live"])
	require.True(t, ok)
	assert.True(t, b)
	assert.Equal(t, "second", docs[1].Content)
}

func TestJSONLoaderSingleObjectWithNestedFields(t *testing.T) {
	path := writeTemp(t, "one.json", `{"uid": 42, "doc": {"body": "hello"}}`)
	docs, err := (&JSONLoader{IDField: "uid", ContentField: "doc.body"}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "42", docs[0].ID)
	assert.Equal(t, "hello", docs[0].Content)
}

func TestJSONLoaderIDFallback(t *testing.T) {
	path := writeTemp(t, "fb.json", `[{"content": "x"}, {"id": "", "content": "y"}]`)
	docs, err := (&JSONLoader{IDField: "id"}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 2)
	assert.Equal(t, path+":0", docs[0].ID)
	assert.Equal(t, path+":1", docs[1].ID)
}

func TestJSONLoaderErrors(t *testing.T) {
	ctx := context.Background()

	notJSON := writeTemp(t, "bad.json", `{oops`)
	_, err := (&JSONLoader{}).Load(ctx, notJSON)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse")

	scalar := writeTemp(t, "scalar.json", `"just a string"`)
	_, err = (&JSONLoader{}).Load(ctx, scalar)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "top level")

	emptyArray := writeTemp(t, "empty.json", `[]`)
	_, err = (&JSONLoader{}).Load(ctx, emptyArray)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no objects")

	mixedArray := writeTemp(t, "mixed.json", `["not-an-object", {"content": "x"}]`)
	_, err = (&JSONLoader{}).Load(ctx, mixedArray)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not an object")

	missingField := writeTemp(t, "missing.json", `[{"id": "a"}]`)
	_, err = (&JSONLoader{}).Load(ctx, missingField)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `missing field "content"`)

	wrongType := writeTemp(t, "wrong.json", `[{"content": 7}]`)
	_, err = (&JSONLoader{}).Load(ctx, wrongType)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not a string")
}
