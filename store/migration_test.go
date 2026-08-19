package store

import (
	"context"
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/deagy/recall/embedder"
	"github.com/stretchr/testify/require"
)

// openTestDB opens a single-connection in-memory SQLite database for testing
// the Migrator directly (bypassing the store).
func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	require.NoError(t, err)
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	return db
}

func TestMigrator_AppliesInOrder(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	m := NewMigrator(db, []Migration{
		{Version: 1, Name: "add-a", SQL: "CREATE TABLE a (id INTEGER PRIMARY KEY)"},
		{Version: 2, Name: "add-b", SQL: "CREATE TABLE b (id INTEGER PRIMARY KEY)"},
	})
	require.NoError(t, m.Migrate(ctx))

	v, err := m.Version(ctx)
	require.NoError(t, err)
	require.Equal(t, 2, v)

	// Both tables should now exist.
	for _, table := range []string{"a", "b"} {
		var n int
		require.NoError(t, db.QueryRowContext(ctx,
			`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?`, table).Scan(&n))
		require.Equal(t, 1, n, "table %s should exist", table)
	}
}

func TestMigrator_Idempotent(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	m := NewMigrator(db, []Migration{
		{Version: 1, Name: "add-a", SQL: "CREATE TABLE a (id INTEGER PRIMARY KEY)"},
	})
	require.NoError(t, m.Migrate(ctx))
	// Running again should be a no-op (already at version 1).
	require.NoError(t, m.Migrate(ctx))

	pending, err := m.Pending(ctx)
	require.NoError(t, err)
	require.Empty(t, pending)

	v, err := m.Version(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, v)
}

func TestMigrator_SortsOutOfOrderInput(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	// Provided out of order; should still apply 1 then 2.
	m := NewMigrator(db, []Migration{
		{Version: 2, Name: "add-b", SQL: "CREATE TABLE b (id INTEGER PRIMARY KEY)"},
		{Version: 1, Name: "add-a", SQL: "CREATE TABLE a (id INTEGER PRIMARY KEY)"},
	})
	require.NoError(t, m.Migrate(ctx))

	applied, err := m.Applied(ctx)
	require.NoError(t, err)
	require.Equal(t, "add-a", applied[1])
	require.Equal(t, "add-b", applied[2])
}

func TestMigrator_RollbackOnFailure(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	m := NewMigrator(db, []Migration{
		{Version: 1, Name: "ok", SQL: "CREATE TABLE good (id INTEGER PRIMARY KEY)"},
		{Version: 2, Name: "bad", SQL: "CREATE TABLE good (id INTEGER PRIMARY KEY)"}, // duplicate -> fails
	})
	err := m.Migrate(ctx)
	require.Error(t, err, "second migration should fail")

	// Version must remain at 1 (the failed migration rolled back).
	v, err := m.Version(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, v)

	applied, err := m.Applied(ctx)
	require.NoError(t, err)
	_, hasTwo := applied[2]
	require.False(t, hasTwo, "failed migration must not be recorded")
}

func TestMigrator_Pending(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	m := NewMigrator(db, []Migration{
		{Version: 1, Name: "one", SQL: "CREATE TABLE one (id INTEGER PRIMARY KEY)"},
		{Version: 2, Name: "two", SQL: "CREATE TABLE two (id INTEGER PRIMARY KEY)"},
		{Version: 3, Name: "three", SQL: "CREATE TABLE three (id INTEGER PRIMARY KEY)"},
	})
	// Before any migration, all three are pending.
	pending, err := m.Pending(ctx)
	require.NoError(t, err)
	require.Len(t, pending, 3)

	require.NoError(t, m.Migrate(ctx))
	pending, err = m.Pending(ctx)
	require.NoError(t, err)
	require.Empty(t, pending)
}

func TestSQLiteStore_AutoMigrationsAndVersion(t *testing.T) {
	s, err := NewSQLiteStore(Config{
		Namespace: "test",
		Embedder:  embedder.NewMockEmbedder(384),
		Migrations: []Migration{
			{Version: 1, Name: "add-notes", SQL: "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)"},
			{Version: 2, Name: "index-notes", SQL: "CREATE INDEX idx_notes_body ON notes(body)"},
		},
	}, ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() { s.Close() })

	ctx := context.Background()
	v, err := s.SchemaVersion(ctx)
	require.NoError(t, err)
	require.Equal(t, 2, v)

	// The migration-created table should be usable.
	_, err = s.db.ExecContext(ctx, `INSERT INTO notes (body) VALUES ('hello')`)
	require.NoError(t, err)
	var n int
	require.NoError(t, s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM notes`).Scan(&n))
	require.Equal(t, 1, n)

	// Explicit Migrate is a no-op now.
	require.NoError(t, s.Migrate(ctx))
}

func TestSQLiteStore_MigrateNoOpsWithoutMigrations(t *testing.T) {
	s := newTestSQLiteStore(t)
	require.NoError(t, s.Migrate(context.Background()))
	v, err := s.SchemaVersion(context.Background())
	require.NoError(t, err)
	require.Equal(t, 0, v)
}
