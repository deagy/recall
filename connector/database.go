package connector

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

// DatabaseConnector loads documents from SQL tables by running a configured
// query and mapping rows to documents. The database driver is injected
// (the *sql.DB is opened by the caller), keeping this package driver-free.
type DatabaseConnector struct {
	// DB is the open database handle. Required.
	DB *sql.DB

	// Query is the SELECT statement to run, e.g.
	// "SELECT id, title, body FROM docs". Required.
	Query string

	// IDColumn is the document ID column; default "id".
	IDColumn string

	// ContentColumn is the column holding the document body. Required.
	ContentColumn string

	// TitleColumn optionally provides a title; empty means ID is used.
	TitleColumn string

	// MetadataColumns are extra columns copied into document metadata
	// (string-valued).
	MetadataColumns []string

	// Limit, when > 0, is appended as "LIMIT n" to the query.
	Limit int
}

// Name implements Connector.
func (d *DatabaseConnector) Name() string { return "database" }

// Fetch runs the query and returns one document per row. The ref argument
// is currently ignored (the query is fixed at construction).
func (d *DatabaseConnector) Fetch(ctx context.Context, ref string) ([]*loader.Document, error) {
	if d.DB == nil {
		return nil, fmt.Errorf("database: nil DB handle")
	}
	if strings.TrimSpace(d.Query) == "" {
		return nil, fmt.Errorf("database: empty query")
	}
	idCol := d.IDColumn
	if idCol == "" {
		idCol = "id"
	}
	q := strings.TrimSpace(d.Query)
	if d.Limit > 0 && !strings.HasSuffix(strings.ToLower(strings.TrimSpace(q)), ";") {
		q += fmt.Sprintf(" LIMIT %d", d.Limit)
	}
	rows, err := d.DB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("database: query: %w", err)
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("database: columns: %w", err)
	}
	idx := make(map[string]int, len(cols))
	for i, c := range cols {
		idx[strings.ToLower(c)] = i
	}
	for _, name := range []string{idCol, strings.ToLower(d.ContentColumn)} {
		if _, ok := idx[strings.ToLower(name)]; !ok {
			return nil, fmt.Errorf("database: column %q not in result set %v", name, cols)
		}
	}

	docs := make([]*loader.Document, 0, 16)
	for rows.Next() {
		vals := make([]sql.RawBytes, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return docs, fmt.Errorf("database: scan: %w", err)
		}
		id := string(vals[idx[strings.ToLower(idCol)]])
		content := string(vals[idx[strings.ToLower(d.ContentColumn)]])
		if strings.TrimSpace(content) == "" {
			continue // skip empty rows
		}
		title := id
		if d.TitleColumn != "" {
			if ti, ok := idx[strings.ToLower(d.TitleColumn)]; ok {
				title = string(vals[ti])
			}
		}
		ld := loader.NewDocument(id, title, "sql:"+strings.TrimSpace(d.Query)[:min(64, len(strings.TrimSpace(d.Query)))], content)
		for _, mc := range d.MetadataColumns {
			if mi, ok := idx[strings.ToLower(mc)]; ok {
				ld.Metadata[strings.ToLower(mc)] = core.String{Value: string(vals[mi])}
			}
		}
		docs = append(docs, ld)
	}
	if err := rows.Err(); err != nil {
		return docs, fmt.Errorf("database: rows: %w", err)
	}
	return docs, nil
}
