package loader

import (
	"context"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCSVLoaderBasic(t *testing.T) {
	path := writeTemp(t, "people.csv", "name,city\nAda,London\nGrace,NYC\n")
	docs, err := (&CSVLoader{HeaderRow: true}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 2)

	assert.Equal(t, path+":row1", docs[0].ID)
	assert.Equal(t, "Ada | London", docs[0].Content)
	assert.Equal(t, "London", core.ToString(docs[0].Metadata["city"]))
	assert.Equal(t, "people row 1", docs[0].Title)
	assert.Equal(t, path+":row2", docs[1].ID)
}

func TestCSVLoaderColumnMapping(t *testing.T) {
	path := writeTemp(t, "map.csv", "id,desc,tags\nd1,first tag,two\nd2,x,y\n")
	docs, err := (&CSVLoader{HeaderRow: true, IDColumn: "id", ContentColumns: []string{"desc"}, Join: " "}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 2)
	assert.Equal(t, "d1", docs[0].ID)
	assert.Equal(t, "first tag", docs[0].Content)
	assert.Equal(t, "two", core.ToString(docs[0].Metadata["tags"]))
}

func TestCSVLoaderNoHeader(t *testing.T) {
	path := writeTemp(t, "raw.csv", "a,b\nc,d\n")
	docs, err := (&CSVLoader{}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 2)
	assert.Equal(t, "a | b", docs[0].Content)
	assert.Equal(t, "a", core.ToString(docs[0].Metadata["col_0"]))
}

func TestCSVLoaderCustomSeparator(t *testing.T) {
	path := writeTemp(t, "semi.csv", "k|v\n1|2\n")
	docs, err := (&CSVLoader{HeaderRow: true, Separator: '|'}).Load(context.Background(), path)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "1 | 2", docs[0].Content)
}

func TestCSVLoaderErrors(t *testing.T) {
	ctx := context.Background()

	// No header row (empty file).
	empty := writeTemp(t, "empty.csv", "")
	_, err := (&CSVLoader{HeaderRow: true}).Load(ctx, empty)
	require.Error(t, err)

	// Header only, no data rows.
	headerOnly := writeTemp(t, "h.csv", "a,b\n")
	_, err = (&CSVLoader{HeaderRow: true}).Load(ctx, headerOnly)
	require.Error(t, err)

	// Unknown ID column.
	bad := writeTemp(t, "bad.csv", "a,b\n1,2\n")
	_, err = (&CSVLoader{HeaderRow: true, IDColumn: "nope"}).Load(ctx, bad)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no column")

	// Unknown content column.
	_, err = (&CSVLoader{HeaderRow: true, ContentColumns: []string{"nope"}}).Load(ctx, bad)
	require.Error(t, err)

	// Single column file where the only column is the ID column.
	idOnly := writeTemp(t, "id.csv", "id\nx\n")
	_, err = (&CSVLoader{HeaderRow: true, IDColumn: "id"}).Load(ctx, idOnly)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no content columns")

	// Field count mismatch.
	mismatch := writeTemp(t, "mm.csv", "a,b\n1,2,3\n")
	_, err = (&CSVLoader{HeaderRow: true}).Load(ctx, mismatch)
	require.Error(t, err)

	// Missing file.
	_, err = (&CSVLoader{HeaderRow: true}).Load(ctx, "/nope/missing.csv")
	require.Error(t, err)
}
