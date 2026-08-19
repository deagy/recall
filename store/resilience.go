package store

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"
)

// CheckpointStats reports the outcome of a WAL checkpoint.
type CheckpointStats struct {
	// Busy is true when a concurrent reader prevented a full checkpoint.
	Busy bool

	// LogFrames is the total number of frames in the WAL at checkpoint time.
	LogFrames int

	// CheckpointedFrames is the number of frames written back to the database.
	CheckpointedFrames int
}

// Checkpoint forces a WAL checkpoint, flushing committed transactions from the
// write-ahead log back into the main database file. mode is one of "PASSIVE"
// (the default), "FULL", "TRUNCATE", or "RESTART".
func (s *SQLiteStore) Checkpoint(ctx context.Context, mode string) (CheckpointStats, error) {
	m := strings.ToUpper(strings.TrimSpace(mode))
	switch m {
	case "":
		m = "PASSIVE"
	case "PASSIVE", "FULL", "TRUNCATE", "RESTART":
	default:
		return CheckpointStats{}, fmt.Errorf("store: unknown checkpoint mode %q", mode)
	}
	var busy, log, ckpt int
	if err := s.db.QueryRowContext(ctx, `PRAGMA wal_checkpoint(`+m+`)`).Scan(&busy, &log, &ckpt); err != nil {
		return CheckpointStats{}, fmt.Errorf("checkpoint: %w", err)
	}
	return CheckpointStats{Busy: busy != 0, LogFrames: log, CheckpointedFrames: ckpt}, nil
}

// StartAutoCheckpoint runs a PASSIVE WAL checkpoint every interval in the
// background, bounding WAL growth for long-running writers. It stops when the
// store is Closed or when ctx is cancelled. A non-positive interval disables
// it. Calling it again replaces any previously started checkpoint loop.
func (s *SQLiteStore) StartAutoCheckpoint(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		return
	}
	cctx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	if s.autoCheckpointCancel != nil {
		s.autoCheckpointCancel()
	}
	s.autoCheckpointCancel = cancel
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-cctx.Done():
				return
			case <-ticker.C:
				_, _ = s.Checkpoint(cctx, "PASSIVE")
			}
		}
	}()
}

// Backup creates a consistent point-in-time copy of the database at destPath
// using SQLite's online VACUUM INTO, which is safe to run while the store is
// actively in use.
func (s *SQLiteStore) Backup(ctx context.Context, destPath string) error {
	if destPath == "" {
		return fmt.Errorf("store: backup destination path is required")
	}
	if _, err := s.db.ExecContext(ctx, `VACUUM INTO `+quoteSQLLiteral(destPath)); err != nil {
		return fmt.Errorf("backup: %w", err)
	}
	return nil
}

// quoteSQLLiteral safely quotes a SQL string literal by doubling embedded
// single quotes, preventing injection via file paths.
func quoteSQLLiteral(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

// RestoreSQLite replaces the database file at destPath with a byte-for-byte
// copy of the backup at srcPath, atomically (write to a temp file, then
// rename). It also removes the WAL/SHM sidecar files so the restored database
// starts clean. The destination store must be closed before restoring.
func RestoreSQLite(srcPath, destPath string) error {
	data, err := os.ReadFile(srcPath)
	if err != nil {
		return fmt.Errorf("reading backup %q: %w", srcPath, err)
	}
	tmp := destPath + ".restore-tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("writing restore temp file: %w", err)
	}
	if err := os.Rename(tmp, destPath); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("renaming backup into place: %w", err)
	}
	// Drop stale WAL/SHM sidecars from the previous database.
	_ = os.Remove(destPath + "-wal")
	_ = os.Remove(destPath + "-shm")
	return nil
}

// IntegrityReport is the result of a database integrity scan.
type IntegrityReport struct {
	// OK is true when no problems were found.
	OK bool

	// Issues lists each structural problem reported by PRAGMA
	// integrity_check (empty when the database is structurally sound).
	Issues []string

	// ForeignKeys lists rows that violate a foreign key constraint.
	ForeignKeys []string
}

// IntegrityCheck detects corrupted data by running PRAGMA integrity_check
// (structural/page-level) and PRAGMA foreign_key_check (referential
// integrity). It is read-only and safe to run while the store is in use.
func (s *SQLiteStore) IntegrityCheck(ctx context.Context) (*IntegrityReport, error) {
	rep := &IntegrityReport{}

	rows, err := s.db.QueryContext(ctx, `PRAGMA integrity_check`)
	if err != nil {
		return nil, fmt.Errorf("integrity_check: %w", err)
	}
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			rows.Close()
			return nil, err
		}
		if line != "ok" {
			rep.Issues = append(rep.Issues, line)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	fkRows, err := s.db.QueryContext(ctx, `PRAGMA foreign_key_check`)
	if err != nil {
		return rep, fmt.Errorf("foreign_key_check: %w", err)
	}
	for fkRows.Next() {
		var table, rowid, parent, fkID string
		if err := fkRows.Scan(&table, &rowid, &parent, &fkID); err != nil {
			fkRows.Close()
			return rep, err
		}
		rep.ForeignKeys = append(rep.ForeignKeys, fmt.Sprintf("%s(rowid=%s) -> %s", table, rowid, parent))
	}
	fkRows.Close()
	if err := fkRows.Err(); err != nil {
		return rep, err
	}

	rep.OK = len(rep.Issues) == 0 && len(rep.ForeignKeys) == 0
	return rep, nil
}

// Repair attempts to repair common, safe corruption: it rebuilds the FTS5
// index from the content table so keyword search matches the stored chunks
// again. It does not repair structural (page-level) corruption, which
// requires restoring from a backup (see Backup and RestoreSQLite).
func (s *SQLiteStore) Repair(ctx context.Context) error {
	if _, err := s.db.ExecContext(ctx, `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`); err != nil {
		return fmt.Errorf("rebuilding FTS index: %w", err)
	}
	return nil
}

// ResilientStore is satisfied by stores that support production resilience
// operations: WAL checkpointing, point-in-time backup, integrity checks, and
// repair. SQLiteStore implements it; in-memory stores do not.
type ResilientStore interface {
	// Checkpoint forces a WAL checkpoint.
	Checkpoint(ctx context.Context, mode string) (CheckpointStats, error)
	// Backup writes a consistent point-in-time copy to destPath.
	Backup(ctx context.Context, destPath string) error
	// IntegrityCheck scans for corrupted data.
	IntegrityCheck(ctx context.Context) (*IntegrityReport, error)
	// Repair repairs safe, common corruption (rebuilds the FTS index).
	Repair(ctx context.Context) error
}

var _ ResilientStore = (*SQLiteStore)(nil)
