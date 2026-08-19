package store

import (
	"context"
	"database/sql"
	"fmt"
	"sort"
	"strconv"
)

// Migration is a single, ordered schema change.
type Migration struct {
	// Version is the monotonically increasing schema version this migration
	// produces. Migrations are applied in ascending Version order.
	Version int

	// Name is a short human-readable identifier recorded in the migrations
	// bookkeeping table.
	Name string

	// SQL is the DDL/DML to run when this migration is applied. It executes
	// inside a single transaction.
	SQL string
}

// Migrator applies schema migrations to a SQLite database. Applied versions
// are tracked in both PRAGMA user_version and a schema_migrations table, so
// Migrate is idempotent and safe to call repeatedly (e.g. on every open).
type Migrator struct {
	db         *sql.DB
	migrations []Migration
}

// NewMigrator creates a Migrator for db with the given migrations. Migrations
// are sorted by Version ascending so the supplied order does not matter.
func NewMigrator(db *sql.DB, migrations []Migration) *Migrator {
	sorted := make([]Migration, len(migrations))
	copy(sorted, migrations)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Version < sorted[j].Version })
	return &Migrator{db: db, migrations: sorted}
}

// ensureMigrationsTable creates the bookkeeping table if it does not exist.
func (m *Migrator) ensureMigrationsTable(ctx context.Context) error {
	_, err := m.db.ExecContext(ctx, `
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);`)
	return err
}

// Version returns the current schema version (PRAGMA user_version).
func (m *Migrator) Version(ctx context.Context) (int, error) {
	var v int
	err := m.db.QueryRowContext(ctx, `PRAGMA user_version`).Scan(&v)
	return v, err
}

// Applied returns the applied migration versions mapped to their names.
func (m *Migrator) Applied(ctx context.Context) (map[int]string, error) {
	if err := m.ensureMigrationsTable(ctx); err != nil {
		return nil, fmt.Errorf("ensuring migrations table: %w", err)
	}
	rows, err := m.db.QueryContext(ctx, `SELECT version, name FROM schema_migrations ORDER BY version`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	applied := make(map[int]string)
	for rows.Next() {
		var v int
		var name string
		if err := rows.Scan(&v, &name); err != nil {
			return nil, err
		}
		applied[v] = name
	}
	return applied, rows.Err()
}

// Pending returns the migrations not yet applied, in ascending order.
func (m *Migrator) Pending(ctx context.Context) ([]Migration, error) {
	current, err := m.Version(ctx)
	if err != nil {
		return nil, fmt.Errorf("reading schema version: %w", err)
	}
	var pending []Migration
	for _, mig := range m.migrations {
		if mig.Version > current {
			pending = append(pending, mig)
		}
	}
	return pending, nil
}

// Migrate applies all pending migrations in order, each in its own
// transaction. It is a no-op when the schema is already up to date.
func (m *Migrator) Migrate(ctx context.Context) error {
	if err := m.ensureMigrationsTable(ctx); err != nil {
		return fmt.Errorf("ensuring migrations table: %w", err)
	}
	current, err := m.Version(ctx)
	if err != nil {
		return fmt.Errorf("reading schema version: %w", err)
	}
	for _, mig := range m.migrations {
		if mig.Version <= current {
			continue
		}
		if err := m.apply(ctx, mig); err != nil {
			return fmt.Errorf("applying migration v%d (%s): %w", mig.Version, mig.Name, err)
		}
		current = mig.Version
	}
	return nil
}

// apply runs a single migration's SQL, records it, and bumps the schema
// version, all atomically.
func (m *Migrator) apply(ctx context.Context, mig Migration) error {
	tx, err := m.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()
	if _, err = tx.ExecContext(ctx, mig.SQL); err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, `INSERT INTO schema_migrations (version, name) VALUES (?, ?)`,
		mig.Version, mig.Name); err != nil {
		return err
	}
	// mig.Version is an int, so interpolation cannot inject SQL.
	if _, err = tx.ExecContext(ctx, `PRAGMA user_version = `+strconv.Itoa(mig.Version)); err != nil {
		return err
	}
	return tx.Commit()
}
