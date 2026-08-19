package store

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "modernc.org/sqlite"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/stretchr/testify/require"
)

// newTestFileStore creates a file-backed (WAL-mode) SQLite store in a temp dir,
// returning the store and its database path.
func newTestFileStore(t *testing.T) (*SQLiteStore, string) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	s, err := NewSQLiteStore(Config{
		Namespace: "test",
		Embedder:  embedder.NewMockEmbedder(32),
	}, dbPath)
	require.NoError(t, err)
	t.Cleanup(func() { s.Close() })
	return s, dbPath
}

// countChunks returns the number of rows in the chunks table.
func countChunks(t *testing.T, s *SQLiteStore) int {
	t.Helper()
	var n int
	require.NoError(t, s.db.QueryRowContext(context.Background(), `SELECT COUNT(*) FROM chunks`).Scan(&n))
	return n
}

// openTestFileDB opens a file-backed database for verification.
func openTestFileDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	return db, nil
}

func TestSQLiteStore_Checkpoint(t *testing.T) {
	s, _ := newTestFileStore(t)
	ctx := context.Background()

	// Write some data so the WAL has frames to checkpoint.
	doc := core.NewDocument("d1", "Doc", "t.txt")
	require.NoError(t, s.Upload(ctx, doc, "some content long enough to be chunked and written to the WAL for checkpointing"))

	// PASSIVE (default) and TRUNCATE should succeed.
	for _, mode := range []string{"", "PASSIVE", "TRUNCATE"} {
		_, err := s.Checkpoint(ctx, mode)
		require.NoError(t, err, "mode %q", mode)
	}

	// An unknown mode is rejected.
	_, err := s.Checkpoint(ctx, "BOGUS")
	require.Error(t, err)
}

func TestSQLiteStore_BackupAndRestore(t *testing.T) {
	s, _ := newTestFileStore(t)
	ctx := context.Background()

	doc := core.NewDocument("d1", "Doc", "t.txt")
	require.NoError(t, s.Upload(ctx, doc, "backup me please, this content is long enough to be chunked by the store"))
	require.Equal(t, 1, countChunks(t, s))

	backupPath := filepath.Join(t.TempDir(), "backup.db")
	require.NoError(t, s.Backup(ctx, backupPath))

	// The backup file must exist and be non-empty.
	info, err := os.Stat(backupPath)
	require.NoError(t, err)
	require.Greater(t, info.Size(), int64(0))

	// The backup must be a valid, queryable database with the same data.
	opened, err := openTestFileDB(backupPath)
	require.NoError(t, err)
	var n int
	require.NoError(t, opened.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&n))
	require.Equal(t, 1, n)
	opened.Close()

	// Empty destination is rejected.
	require.Error(t, s.Backup(ctx, ""))
}

func TestSQLiteStore_RestoreSQLite(t *testing.T) {
	s, dbPath := newTestFileStore(t)
	ctx := context.Background()

	require.NoError(t, s.Upload(ctx, core.NewDocument("d1", "Doc", "t.txt"), "version one content that is long enough to be chunked"))
	require.NoError(t, s.Upload(ctx, core.NewDocument("d2", "Doc2", "t2.txt"), "version two content that is long enough to be chunked"))
	require.Equal(t, 2, countChunks(t, s))

	// Take a backup at the 2-chunk state.
	backupPath := filepath.Join(t.TempDir(), "backup.db")
	require.NoError(t, s.Backup(ctx, backupPath))

	// Add more data, then close and restore the earlier backup.
	require.NoError(t, s.Upload(ctx, core.NewDocument("d3", "Doc3", "t3.txt"), "version three content that is long enough to chunk"))
	require.Equal(t, 3, countChunks(t, s))
	require.NoError(t, s.Close())

	require.NoError(t, RestoreSQLite(backupPath, dbPath))

	// Reopen: should see the restored 2-chunk state, not the 3-chunk state.
	reopened, err := NewSQLiteStore(Config{Namespace: "test", Embedder: embedder.NewMockEmbedder(32)}, dbPath)
	require.NoError(t, err)
	t.Cleanup(func() { reopened.Close() })
	require.Equal(t, 2, countChunks(t, reopened))
}

func TestSQLiteStore_QuoteLiteral(t *testing.T) {
	require.Equal(t, "'/tmp/x.db'", quoteSQLLiteral("/tmp/x.db"))
	require.Equal(t, "'/tmp/it''s.db'", quoteSQLLiteral("/tmp/it's.db"))
}

func TestSQLiteStore_IntegrityCheckOK(t *testing.T) {
	s, _ := newTestFileStore(t)
	ctx := context.Background()
	require.NoError(t, s.Upload(ctx, core.NewDocument("d1", "Doc", "t.txt"), "intact data content that is long enough to be chunked"))

	rep, err := s.IntegrityCheck(ctx)
	require.NoError(t, err)
	require.True(t, rep.OK)
	require.Empty(t, rep.Issues)
	require.Empty(t, rep.ForeignKeys)
}

func TestSQLiteStore_IntegrityCheckDetectsFKViolation(t *testing.T) {
	s, _ := newTestFileStore(t)
	ctx := context.Background()

	// Insert an orphan embeddings row whose chunk does not exist in chunks.
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO embeddings (chunk_id, namespace, embedding) VALUES ('orphan-id', 'test', X'00000000')`)
	require.NoError(t, err)

	rep, err := s.IntegrityCheck(ctx)
	require.NoError(t, err)
	require.False(t, rep.OK, "should detect the orphan embedding")
	require.NotEmpty(t, rep.ForeignKeys)
}

func TestSQLiteStore_Repair(t *testing.T) {
	s, _ := newTestFileStore(t)
	ctx := context.Background()
	require.NoError(t, s.Upload(ctx, core.NewDocument("d1", "Doc", "t.txt"), "the quick brown fox jumps over the lazy dog near the riverbank"))

	ftsCount := func(t *testing.T) int {
		t.Helper()
		var n int
		require.NoError(t, s.db.QueryRowContext(ctx,
			`SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'quick'`).Scan(&n))
		return n
	}
	require.Greater(t, ftsCount(t), 0, "FTS should match before corruption")

	// Corrupt the FTS index by clearing all entries.
	_, err := s.db.ExecContext(ctx, `INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')`)
	require.NoError(t, err)
	require.Equal(t, 0, ftsCount(t), "FTS should be empty after corruption")

	// Repair rebuilds the FTS index from the content table.
	require.NoError(t, s.Repair(ctx))
	require.Greater(t, ftsCount(t), 0, "FTS should match again after repair")
}

func TestSQLiteStore_StartAutoCheckpoint(t *testing.T) {
	ctx := context.Background()

	// A non-positive interval is a no-op (no goroutine started).
	s, _ := newTestFileStore(t)
	require.NoError(t, s.Upload(ctx, core.NewDocument("d1", "Doc", "t.txt"), "auto checkpoint data that is long enough to be chunked"))
	s.StartAutoCheckpoint(ctx, 0)
	require.NoError(t, s.Close()) // must not hang

	// A positive interval starts a loop that Close stops cleanly.
	s2, _ := newTestFileStore(t)
	s2.StartAutoCheckpoint(context.Background(), time.Millisecond)
	// Give it a moment to run at least one checkpoint tick.
	time.Sleep(10 * time.Millisecond)
	require.NoError(t, s2.Close())
}

func TestResilientStoreInterface(t *testing.T) {
	var _ ResilientStore = (*SQLiteStore)(nil)
}
