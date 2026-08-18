package connector

import (
	"context"
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

// openTestDB returns an in-memory SQLite database with a docs table.
func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	stmts := []string{
		`CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, body TEXT, tag TEXT)`,
		`INSERT INTO docs VALUES ('d1', 'First', 'Body one', 'a')`,
		`INSERT INTO docs VALUES ('d2', 'Second', 'Body two', 'b')`,
		`INSERT INTO docs VALUES ('d3', 'Empty', '', 'c')`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			t.Fatalf("exec %q: %v", s, err)
		}
	}
	return db
}

func TestDatabaseConnector_Basic(t *testing.T) {
	db := openTestDB(t)
	c := &DatabaseConnector{
		DB:              db,
		Query:           "SELECT id, title, body, tag FROM docs",
		IDColumn:        "id",
		ContentColumn:   "body",
		TitleColumn:     "title",
		MetadataColumns: []string{"tag"},
	}
	docs, err := c.Fetch(context.Background(), "")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(docs) != 2 { // empty-body row skipped
		t.Fatalf("expected 2 docs, got %d", len(docs))
	}
	if docs[0].ID != "d1" || docs[0].Title != "First" || docs[0].Content != "Body one" {
		t.Errorf("doc0: %+v", docs[0])
	}
}

func TestDatabaseConnector_Limit(t *testing.T) {
	db := openTestDB(t)
	c := &DatabaseConnector{
		DB:            db,
		Query:         "SELECT id, body FROM docs WHERE body != ''",
		ContentColumn: "body",
		Limit:         1,
	}
	docs, err := c.Fetch(context.Background(), "")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
}

func TestDatabaseConnector_Errors(t *testing.T) {
	db := openTestDB(t)
	if _, err := (&DatabaseConnector{DB: db, Query: "SELECT nope FROM docs"}).Fetch(context.Background(), ""); err == nil {
		t.Error("expected query error")
	}
	if _, err := (&DatabaseConnector{DB: db, Query: "SELECT id FROM docs"}).Fetch(context.Background(), ""); err == nil {
		t.Error("expected missing content column error")
	}
	if _, err := (&DatabaseConnector{Query: "SELECT 1"}).Fetch(context.Background(), ""); err == nil {
		t.Error("expected nil db error")
	}
}
