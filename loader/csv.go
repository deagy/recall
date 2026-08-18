package loader

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/deagy/recall/core"
)

// CSVLoader loads a CSV file as one document per data row. The first row is
// treated as a header unless HeaderRow is false.
type CSVLoader struct {
	// Separator overrides the default comma.
	Separator rune

	// HeaderRow controls whether the first row names the columns.
	// Default true.
	HeaderRow bool

	// IDColumn selects the column used as the document ID. When empty,
	// IDs are generated as "<path>:row<N>" (1-based data row).
	IDColumn string

	// ContentColumns lists the columns joined into the document content.
	// When empty, all columns except IDColumn are included.
	ContentColumns []string

	// Join separates content column values. Default " | ".
	Join string
}

// Load reads the CSV file at path. Rows with a mismatched number of fields
// are an error (encoding/csv strict mode), which keeps column mapping sound.
func (l *CSVLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("loader: open %s: %w", path, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	if l.Separator != 0 {
		r.Comma = l.Separator
	}
	join := l.Join
	if join == "" {
		join = " | "
	}

	// Read the header (or peek the first data row to derive column names).
	var header []string
	var pending []string
	if l.HeaderRow {
		var err2 error
		header, err2 = r.Read()
		if errors.Is(err2, io.EOF) {
			return nil, fmt.Errorf("loader: %s has no header row", path)
		}
		if err2 != nil {
			return nil, fmt.Errorf("loader: read %s header: %w", path, err2)
		}
	} else {
		var err2 error
		pending, err2 = r.Read()
		if errors.Is(err2, io.EOF) {
			return nil, fmt.Errorf("loader: %s has no data rows", path)
		}
		if err2 != nil {
			return nil, fmt.Errorf("loader: read %s: %w", path, err2)
		}
		header = make([]string, len(pending))
		for i := range header {
			header[i] = fmt.Sprintf("col_%d", i)
		}
	}

	idIdx := -1
	if l.IDColumn != "" {
		idIdx = columnIndex(header, l.IDColumn)
		if idIdx < 0 {
			return nil, fmt.Errorf("loader: %s has no column %q", path, l.IDColumn)
		}
	}
	contentIdx := make([]int, 0, len(header))
	if len(l.ContentColumns) > 0 {
		for _, name := range l.ContentColumns {
			idx := columnIndex(header, name)
			if idx < 0 {
				return nil, fmt.Errorf("loader: %s has no column %q", path, name)
			}
			if idx != idIdx {
				contentIdx = append(contentIdx, idx)
			}
		}
	} else {
		for i := range header {
			if i != idIdx {
				contentIdx = append(contentIdx, i)
			}
		}
	}
	if len(contentIdx) == 0 {
		return nil, fmt.Errorf("loader: %s has no content columns", path)
	}

	docs := make([]*Document, 0)
	for rowNo := 1; ; rowNo++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var record []string
		if pending != nil {
			record, pending = pending, nil
		} else {
			var readErr error
			record, readErr = r.Read()
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				return nil, fmt.Errorf("loader: %s data row %d: %w", path, rowNo, readErr)
			}
		}
		meta := make(map[string]core.Value, len(record))
		for i, col := range header {
			if i < len(record) {
				meta[col] = core.String{Value: record[i]}
			}
		}
		id := fmt.Sprintf("%s:row%d", path, rowNo)
		if idIdx >= 0 && idIdx < len(record) && record[idIdx] != "" {
			id = record[idIdx]
		}
		var sb strings.Builder
		for n, i := range contentIdx {
			if n > 0 {
				sb.WriteString(join)
			}
			if i < len(record) {
				sb.WriteString(record[i])
			}
		}
		doc := NewDocument(id, fmt.Sprintf("%s row %d", baseName(path), rowNo), path, sb.String())
		doc.Metadata = meta
		docs = append(docs, doc)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("loader: %s has no data rows", path)
	}
	return docs, nil
}

func columnIndex(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}
