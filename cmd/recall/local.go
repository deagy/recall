package main

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite" // registers the pure-Go "sqlite" driver

	"github.com/deagy/recall/app"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/loader"
	"github.com/deagy/recall/store"
)

// supportedExtensions are the file types `recall upload` accepts.
var supportedExtensions = []string{
	".txt", ".text", ".md", ".markdown", ".csv", ".json", ".html", ".htm",
	".pdf", ".docx",
}

// loaderForPath returns the loader that reads path (a file or directory).
func loaderForPath(path string, recursive bool) (loader.Loader, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return loader.NewDirectoryLoader(supportedExtensions, recursive, nil)
	}
	return loader.ForExtension(filepath.Ext(path))
}

// toCoreDocument maps a loader document to the store's document type.
func toCoreDocument(d *loader.Document, namespace string) *core.Document {
	cd := core.NewDocument(d.ID, d.Title, d.Source)
	cd.Namespace = namespace
	if len(d.Metadata) > 0 {
		cd.Metadata = make(map[string]core.Value, len(d.Metadata))
		for k, v := range d.Metadata {
			cd.Metadata[k] = v
		}
	}
	return cd
}

// metadataToAny converts typed core metadata into plain values for the
// JSON wire format (server mode).
func metadataToAny(meta map[string]core.Value) map[string]any {
	if len(meta) == 0 {
		return nil
	}
	out := make(map[string]any, len(meta))
	for k, v := range meta {
		if v == nil {
			continue
		}
		switch val := v.(type) {
		case core.String:
			out[k] = val.Value
		case core.Number:
			out[k] = val.Value
		case core.Boolean:
			out[k] = val.Value
		case core.URI:
			out[k] = val.Value
		case core.Literal:
			out[k] = val.Value
		default:
			out[k] = v.String()
		}
	}
	return out
}

// openLocalStore opens the in-process store described by the configuration.
// In local mode the store's namespace is the effective namespace so uploads
// land where the user asked.
func (o *globalOptions) openLocalStore() (store.Store, error) {
	sc := o.cfg.Store
	sc.Namespace = o.effectiveNamespace()
	return app.BuildStore(sc)
}

// namespaceFilter returns a metadata filter restricting results to ns, or
// nil when ns is empty.
func namespaceFilter(ns string) []index.Filter {
	if ns == "" {
		return nil
	}
	return []index.Filter{&index.TermInFilter{Key: core.MetadataKeyNamespace, Values: []string{ns}}}
}

// localDBPath returns the SQLite path for local maintenance commands,
// honoring an optional --db override.
func (o *globalOptions) localDBPath(override string) (string, error) {
	path := override
	if path == "" {
		path = o.cfg.Store.Path
	}
	if path == "" {
		return "", errors.New("no database path: set store.path in the config or pass --db")
	}
	return path, nil
}

// requireSQLiteLocal ensures a local operation on the SQLite file is
// possible (backend + path resolvable).
func (o *globalOptions) requireSQLiteLocal(override string) (string, error) {
	if o.cli != nil {
		return "", errors.New("this command is local-only; rerun without --server (and clear cli.url)")
	}
	if o.cfg.Store.Backend != config.BackendSQLite {
		return "", fmt.Errorf("this command requires store.backend: sqlite (configured backend: %q)", o.cfg.Store.Backend)
	}
	return o.localDBPath(override)
}

// openSQLiteDB opens a raw SQLite database with a single serialized
// connection — the pattern for maintenance operations that must not go
// through the full store (and its embedder).
func openSQLiteDB(path string) (*sql.DB, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("database not found at %s: %w", path, err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("enabling WAL: %w", err)
	}
	return db, nil
}

// quoteLiteral safely quotes a SQL string literal by doubling embedded
// single quotes (same technique as store.Backup).
func quoteLiteral(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}
