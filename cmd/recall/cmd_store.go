package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/store"
)

// storeInfoOutput is the result of `recall store info`.
type storeInfoOutput struct {
	Mode          string   `json:"mode" yaml:"mode"`
	Backend       string   `json:"backend" yaml:"backend"`
	Path          string   `json:"path,omitempty" yaml:"path,omitempty"`
	Status        string   `json:"status" yaml:"status"`
	OK            bool     `json:"ok" yaml:"ok"`
	Connected     bool     `json:"connected" yaml:"connected"`
	Chunks        int      `json:"chunks" yaml:"chunks"`
	Namespaces    []string `json:"namespaces,omitempty" yaml:"namespaces,omitempty"`
	SchemaVersion int      `json:"schema_version,omitempty" yaml:"schema_version,omitempty"`
	IntegrityOK   *bool    `json:"integrity_ok,omitempty" yaml:"integrity_ok,omitempty"`
	Issues        []string `json:"issues,omitempty" yaml:"issues,omitempty"`
	CheckedAt     string   `json:"checked_at,omitempty" yaml:"checked_at,omitempty"`
}

func newStoreCmd(o *globalOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "store",
		Short: "Store maintenance and inspection",
	}
	cmd.AddCommand(newStoreInfoCmd(o), newStoreMigrateCmd(o), newStoreBackupCmd(o), newStoreRestoreCmd(o))
	return cmd
}

func newStoreInfoCmd(o *globalOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "info",
		Short: "Display store statistics and health",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()

			if o.cli != nil {
				d, err := o.cli.Diagnostics(ctx)
				if err != nil {
					return err
				}
				h := d.Health
				out := &storeInfoOutput{
					Mode: "server", Backend: h.Backend, Status: h.Status, OK: h.OK, Connected: h.Connected,
					Chunks: h.Count, Namespaces: h.Namespaces, Issues: h.Issues,
					CheckedAt: h.CheckedAt.Format(time.RFC3339),
				}
				if h.Integrity != nil {
					ok := h.Integrity.OK
					out.IntegrityOK = &ok
					out.Issues = append(out.Issues, h.Integrity.Issues...)
				}
				return emitStoreInfo(cmd, o, out)
			}

			st, err := o.openLocalStore()
			if err != nil {
				return err
			}
			defer st.Close()

			rep, err := store.HealthCheck(ctx, st)
			if err != nil {
				return err
			}
			out := &storeInfoOutput{
				Mode: "local", Backend: rep.Backend, Path: o.cfg.Store.Path, Status: rep.Status, OK: rep.OK,
				Connected: rep.Connected, Chunks: rep.Count, Namespaces: rep.Namespaces, Issues: rep.Issues,
				CheckedAt: rep.CheckedAt.Format(time.RFC3339),
			}
			if sq, ok := st.(*store.SQLiteStore); ok {
				v, err := sq.SchemaVersion(ctx)
				if err == nil {
					out.SchemaVersion = v
				}
			}
			if rep.Integrity != nil {
				ok := rep.Integrity.OK
				out.IntegrityOK = &ok
			}
			return emitStoreInfo(cmd, o, out)
		},
	}
}

func emitStoreInfo(cmd *cobra.Command, o *globalOptions, out *storeInfoOutput) error {
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		tw := p.table()
		fmt.Fprintf(tw, "mode:\t%s\n", out.Mode)
		fmt.Fprintf(tw, "backend:\t%s\n", out.Backend)
		if out.Path != "" {
			fmt.Fprintf(tw, "path:\t%s\n", out.Path)
		}
		fmt.Fprintf(tw, "status:\t%s (ok=%v, connected=%v)\n", out.Status, out.OK, out.Connected)
		fmt.Fprintf(tw, "chunks:\t%d\n", out.Chunks)
		if len(out.Namespaces) > 0 {
			fmt.Fprintf(tw, "namespaces:\t%s\n", joinStrings(out.Namespaces))
		}
		if out.SchemaVersion > 0 {
			fmt.Fprintf(tw, "schema_version:\t%d\n", out.SchemaVersion)
		}
		if out.IntegrityOK != nil {
			fmt.Fprintf(tw, "integrity:\t%v\n", *out.IntegrityOK)
		}
		for _, issue := range out.Issues {
			fmt.Fprintf(tw, "issue:\t%s\n", issue)
		}
		if out.CheckedAt != "" {
			fmt.Fprintf(tw, "checked_at:\t%s\n", out.CheckedAt)
		}
		tw.Flush()
	})
}

func joinStrings(ss []string) string {
	out := ""
	for i, s := range ss {
		if i > 0 {
			out += ", "
		}
		out += s
	}
	return out
}

// migrateOutput is the result of `recall store migrate`.
type migrateOutput struct {
	Database      string   `json:"database" yaml:"database"`
	VersionBefore int      `json:"version_before" yaml:"version_before"`
	VersionAfter  int      `json:"version_after" yaml:"version_after"`
	Applied       []string `json:"applied" yaml:"applied"`
}

// Migration files carry one or more versioned migrations. Each migration
// starts with a header line:
//
//	-- recall-migration: version=1 name=add_review_column
//
// and runs until the next header or end of file.
func newStoreMigrateCmd(o *globalOptions) *cobra.Command {
	var dbPath string
	cmd := &cobra.Command{
		Use:   "migrate <file.sql>...",
		Short: "Apply versioned schema migrations to the SQLite store",
		Long: `Apply versioned schema migrations to the SQLite store. Migrations are
read from SQL files, each migration marked by a header comment:

  -- recall-migration: version=1 name=add_review_column
  ALTER TABLE chunks ADD COLUMN review_status TEXT;

Migrations run in ascending version order inside transactions and are
idempotent (already-applied versions are skipped).`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runStoreMigrate(cmd, o, args, dbPath)
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", "", "database path (default: store.path from config)")
	return cmd
}

func runStoreMigrate(cmd *cobra.Command, o *globalOptions, files []string, dbPath string) error {
	ctx := cmd.Context()
	path, err := o.requireSQLiteLocal(dbPath)
	if err != nil {
		return err
	}

	var migs []store.Migration
	for _, f := range files {
		parsed, err := parseMigrationFile(f)
		if err != nil {
			return err
		}
		migs = append(migs, parsed...)
	}
	if len(migs) == 0 {
		return fmt.Errorf("no migrations found in %v", files)
	}

	db, err := openSQLiteDB(path)
	if err != nil {
		return err
	}
	defer db.Close()

	m := store.NewMigrator(db, migs)
	before, err := m.Version(ctx)
	if err != nil {
		return fmt.Errorf("reading schema version: %w", err)
	}
	if err := m.Migrate(ctx); err != nil {
		return fmt.Errorf("applying migrations: %w", err)
	}
	after, err := m.Version(ctx)
	if err != nil {
		return fmt.Errorf("reading schema version: %w", err)
	}

	applied, err := m.Applied(ctx)
	if err != nil {
		return fmt.Errorf("listing applied migrations: %w", err)
	}
	out := &migrateOutput{Database: path, VersionBefore: before, VersionAfter: after, Applied: []string{}}
	for v := before + 1; v <= after; v++ {
		if name, ok := applied[v]; ok {
			out.Applied = append(out.Applied, fmt.Sprintf("v%d %s", v, name))
		} else {
			out.Applied = append(out.Applied, fmt.Sprintf("v%d", v))
		}
	}
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		fmt.Fprintf(p.w, "database:       %s\n", out.Database)
		fmt.Fprintf(p.w, "schema version: %d -> %d\n", out.VersionBefore, out.VersionAfter)
		if len(out.Applied) == 0 {
			fmt.Fprintln(p.w, "up to date (no migrations applied)")
			return
		}
		fmt.Fprintln(p.w, "applied:")
		for _, a := range out.Applied {
			fmt.Fprintf(p.w, "  %s\n", a)
		}
	})
}

// parseMigrationFile reads versioned migrations from a SQL file.
func parseMigrationFile(path string) ([]store.Migration, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	const header = "-- recall-migration:"
	var migs []store.Migration
	var cur *store.Migration
	var sqlLines []string

	flush := func() {
		if cur == nil {
			return
		}
		cur.SQL = joinNonEmpty(sqlLines)
		migs = append(migs, *cur)
		cur = nil
		sqlLines = nil
	}

	for _, line := range strings.Split(string(data), "\n") {
		trimmed := strings.TrimSpace(line)
		if v, ok := strings.CutPrefix(trimmed, header); ok {
			flush()
			mig, err := parseMigrationHeader(strings.TrimSpace(v))
			if err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			cur = &mig
			continue
		}
		if cur != nil {
			sqlLines = append(sqlLines, line)
		}
		// Lines before the first header are ignored (file-level comments).
	}
	flush()
	if len(migs) == 0 {
		return nil, fmt.Errorf("%s: no %q headers found", path, header)
	}
	return migs, nil
}

// joinNonEmpty trims each line and joins the non-blank ones with newlines.
func joinNonEmpty(lines []string) string {
	var b strings.Builder
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(line)
	}
	return b.String()
}

// parseMigrationHeader parses "version=N name=..." (extra keys ignored).
func parseMigrationHeader(s string) (store.Migration, error) {
	var mig store.Migration
	hasVersion := false
	for _, part := range strings.Fields(s) {
		k, v, ok := strings.Cut(part, "=")
		if !ok {
			return mig, fmt.Errorf("malformed header field %q (want key=value)", part)
		}
		switch k {
		case "version":
			n, err := strconv.Atoi(v)
			if err != nil || n <= 0 {
				return mig, fmt.Errorf("invalid version %q", v)
			}
			mig.Version = n
			hasVersion = true
		case "name":
			mig.Name = v
		}
	}
	if !hasVersion {
		return mig, fmt.Errorf("header is missing version=N: %q", s)
	}
	if mig.Name == "" {
		mig.Name = fmt.Sprintf("migration-%d", mig.Version)
	}
	return mig, nil
}

// backupOutput is the result of `recall store backup`.
type backupOutput struct {
	Database  string `json:"database" yaml:"database"`
	Backup    string `json:"backup" yaml:"backup"`
	SizeBytes int64  `json:"size_bytes" yaml:"size_bytes"`
}

func newStoreBackupCmd(o *globalOptions) *cobra.Command {
	var (
		dbPath string
		force  bool
	)
	cmd := &cobra.Command{
		Use:   "backup <destination>",
		Short: "Create an online backup of the SQLite store (VACUUM INTO)",
		Long: `Create an online backup of the SQLite store using VACUUM INTO. The
backup is a self-contained database file safe to copy or restore.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runStoreBackup(cmd, o, args[0], dbPath, force)
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", "", "database path (default: store.path from config)")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite the destination if it exists")
	return cmd
}

func runStoreBackup(cmd *cobra.Command, o *globalOptions, dest, dbPath string, force bool) error {
	ctx := cmd.Context()
	path, err := o.requireSQLiteLocal(dbPath)
	if err != nil {
		return err
	}
	if _, err := os.Stat(dest); err == nil && !force {
		return fmt.Errorf("destination %s exists (use --force to overwrite)", dest)
	}
	if dir := dest[:len(dest)-len(baseName(dest))]; dir != "" {
		if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
			return fmt.Errorf("destination directory %s does not exist", dir)
		}
	}

	db, err := openSQLiteDB(path)
	if err != nil {
		return err
	}
	defer db.Close()
	//nolint:gosec // G202: dest is embedded as a quoted SQL literal (quoteLiteral escapes it); VACUUM INTO has no parameterized form
	if _, err := db.ExecContext(ctx, "VACUUM INTO "+quoteLiteral(dest)); err != nil {
		return fmt.Errorf("backup: %w", err)
	}

	fi, err := os.Stat(dest)
	if err != nil {
		return err
	}
	out := &backupOutput{Database: path, Backup: dest, SizeBytes: fi.Size()}
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		fmt.Fprintf(p.w, "backed up %s -> %s (%d bytes)\n", out.Database, out.Backup, out.SizeBytes)
	})
}

func baseName(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == '\\' {
			return p[i+1:]
		}
	}
	return p
}

// restoreOutput is the result of `recall store restore`.
type restoreOutput struct {
	Backup string `json:"backup" yaml:"backup"`
	Target string `json:"target" yaml:"target"`
}

func newStoreRestoreCmd(o *globalOptions) *cobra.Command {
	var (
		to    string
		force bool
	)
	cmd := &cobra.Command{
		Use:   "restore <backup>",
		Short: "Restore a backup over the store database",
		Long: `Restore a backup created by ` + "`recall store backup`" + ` over the store
database. The destination defaults to store.path from the config (or --to).
The restore is atomic (temp file + rename) and requires --force when the
destination already exists. The destination store must be closed.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runStoreRestore(cmd, o, args[0], to, force)
		},
	}
	cmd.Flags().StringVar(&to, "to", "", "destination database path (default: store.path from config)")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite the destination if it exists")
	return cmd
}

func runStoreRestore(cmd *cobra.Command, o *globalOptions, backup, to string, force bool) error {
	if err := o.requireLocal("recall store restore"); err != nil {
		return err
	}
	if _, err := os.Stat(backup); err != nil {
		return fmt.Errorf("backup not found at %s: %w", backup, err)
	}
	target := to
	if target == "" {
		target = o.cfg.Store.Path
	}
	if target == "" {
		return fmt.Errorf("no destination: set store.path in the config or pass --to")
	}
	if _, err := os.Stat(target); err == nil && !force {
		return fmt.Errorf("destination %s exists (use --force to overwrite)", target)
	}

	if err := store.RestoreSQLite(backup, target); err != nil {
		return err
	}
	out := &restoreOutput{Backup: backup, Target: target}
	p := newPrinter(cmd, o.output)
	return p.emit(out, func(p *printer) {
		fmt.Fprintf(p.w, "restored %s -> %s\n", out.Backup, out.Target)
	})
}
