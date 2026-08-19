package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/deagy/recall/config"
)

func TestLocalStoreMigrate(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)
	mustRunCLI(t, "--config", cfgPath, "upload", writeNotesFile(t)) // create the database

	migFile := filepath.Join(t.TempDir(), "m.sql")
	content := "-- recall-migration: version=1 name=cli_test_probe\n" +
		"CREATE TABLE IF NOT EXISTS cli_test_probe (id INTEGER PRIMARY KEY);\n"
	if err := os.WriteFile(migFile, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	out := mustRunCLI(t, "--config", cfgPath, "store", "migrate", migFile, "-o", "json")
	var mo migrateOutput
	decodeJSON(t, out, &mo)
	if mo.VersionBefore != 0 || mo.VersionAfter != 1 {
		t.Errorf("version before=%d after=%d, want 0 -> 1", mo.VersionBefore, mo.VersionAfter)
	}
	if len(mo.Applied) != 1 || mo.Applied[0] != "v1 cli_test_probe" {
		t.Errorf("applied = %v", mo.Applied)
	}

	// The schema version is reflected in store info.
	out = mustRunCLI(t, "--config", cfgPath, "store", "info", "-o", "json")
	var info storeInfoOutput
	decodeJSON(t, out, &info)
	if info.SchemaVersion != 1 {
		t.Errorf("schema version after migrate = %d, want 1", info.SchemaVersion)
	}

	// Second run is a no-op.
	out = mustRunCLI(t, "--config", cfgPath, "store", "migrate", migFile)
	if !strings.Contains(out, "up to date") {
		t.Errorf("idempotent migrate output: %s", out)
	}

	// Missing migration file errors.
	if _, err := runCLI(t, "--config", cfgPath, "store", "migrate", filepath.Join(t.TempDir(), "missing.sql")); err == nil {
		t.Error("missing migration file: expected error")
	}
}

func TestLocalStoreBackupRestore(t *testing.T) {
	cfgPath, dbPath := writeSQLiteConfig(t)
	mustRunCLI(t, "--config", cfgPath, "upload", writeNotesFile(t))

	backup := filepath.Join(t.TempDir(), "backup.db")
	out := mustRunCLI(t, "--config", cfgPath, "store", "backup", backup, "-o", "json")
	var bo backupOutput
	decodeJSON(t, out, &bo)
	if bo.Database != dbPath || bo.Backup != backup || bo.SizeBytes == 0 {
		t.Errorf("backup output = %+v", bo)
	}

	// Existing destination requires --force.
	if _, err := runCLI(t, "--config", cfgPath, "store", "backup", backup); err == nil || !strings.Contains(err.Error(), "exists") {
		t.Errorf("expected destination-exists error, got %v", err)
	}

	// Restore over the live database (atomic rename, needs --force).
	out = mustRunCLI(t, "--config", cfgPath, "store", "restore", backup, "--force")
	if !strings.Contains(out, "restored") {
		t.Errorf("restore output: %s", out)
	}

	// The store still answers with its data.
	out = mustRunCLI(t, "--config", cfgPath, "store", "info", "-o", "json")
	var info storeInfoOutput
	decodeJSON(t, out, &info)
	if info.Chunks == 0 || !info.OK {
		t.Errorf("store after restore = %+v", info)
	}

	// Restore of a missing backup errors.
	if _, err := runCLI(t, "--config", cfgPath, "store", "restore", filepath.Join(t.TempDir(), "nope.db")); err == nil {
		t.Error("missing backup: expected error")
	}
}

func TestLocalStoreRequiresSQLite(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.Config{}
	cfg.WithDefaults() // memory backend
	cfgPath := filepath.Join(dir, "memory.json")
	if err := config.Save(cfgPath, cfg); err != nil {
		t.Fatal(err)
	}

	_, err := runCLI(t, "--config", cfgPath, "store", "backup", filepath.Join(dir, "b.db"))
	if err == nil || !strings.Contains(err.Error(), "sqlite") {
		t.Errorf("backup on memory backend: expected sqlite requirement error, got %v", err)
	}
	_, err = runCLI(t, "--config", cfgPath, "graph", "list")
	if err == nil || !strings.Contains(err.Error(), "sqlite") {
		t.Errorf("graph list on memory backend: expected sqlite requirement error, got %v", err)
	}
}

func TestLocalGraphAndReason(t *testing.T) {
	cfgPath, dbPath := writeSQLiteConfig(t)
	seedSQLiteGraph(t, dbPath)

	out := mustRunCLI(t, "--config", cfgPath, "graph", "list")
	if !strings.Contains(out, "3 entit(y/ies), 2 relation(s)") {
		t.Fatalf("graph list output: %s", out)
	}

	// Entity lookup by ID, with neighbors and relations.
	out = mustRunCLI(t, "--config", cfgPath, "graph", "alice", "-o", "json")
	var ge graphEntityOutput
	decodeJSON(t, out, &ge)
	if ge.Mode != "local" || ge.Entity.ID != "alice" {
		t.Fatalf("graph entity = %+v", ge)
	}
	if len(ge.Neighbors) != 1 || ge.Neighbors[0].ID != "acme" {
		t.Errorf("neighbors = %+v", ge.Neighbors)
	}
	if len(ge.Relations) != 1 || ge.Relations[0].Type != "works_at" {
		t.Errorf("relations = %+v", ge.Relations)
	}

	// Entity lookup by unique label.
	out = mustRunCLI(t, "--config", cfgPath, "graph", "Acme", "-o", "json")
	decodeJSON(t, out, &ge)
	if ge.Entity.ID != "acme" {
		t.Errorf("label lookup = %+v", ge.Entity)
	}

	// Unknown entity errors.
	if _, err := runCLI(t, "--config", cfgPath, "graph", "nobody"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected entity-not-found error, got %v", err)
	}

	// Path exploration between two entities.
	out = mustRunCLI(t, "--config", cfgPath, "reason", "--from", "alice", "--to", "berlin", "-o", "json")
	var ro reasonOutput
	decodeJSON(t, out, &ro)
	if len(ro.Paths) == 0 {
		t.Fatalf("expected at least one path, got %+v", ro)
	}
	path := ro.Paths[0]
	if strings.Join(path.Entities, ",") != "alice,acme,berlin" {
		t.Errorf("path entities = %v", path.Entities)
	}

	// Table rendering of paths.
	out = mustRunCLI(t, "--config", cfgPath, "reason", "--from", "alice", "--to", "berlin")
	if !strings.Contains(out, "paths (") {
		t.Errorf("reason table output: %s", out)
	}

	// Natural-language reasoning runs without error.
	mustRunCLI(t, "--config", cfgPath, "reason", "Alice")

	// Neither query nor endpoints is an error.
	if _, err := runCLI(t, "--config", cfgPath, "reason"); err == nil || !strings.Contains(err.Error(), "provide a query") {
		t.Errorf("expected argument error, got %v", err)
	}
}
