package loader

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// DirectoryLoader walks a directory tree and loads files by extension,
// dispatching each file to a Loader. Matching files that fail to load are
// reported via errors.Join while successfully loaded documents are still
// returned, so one bad file does not discard the whole directory.
type DirectoryLoader struct {
	// Extensions lists the file extensions to load, e.g. [".txt", ".md"].
	// Required. Matching is case-insensitive.
	Extensions []string

	// Recursive controls whether subdirectories are descended into.
	// Default true.
	Recursive bool

	// Loaders optionally overrides the loader used for a given extension.
	// Unlisted extensions fall back to ForExtension defaults.
	Loaders map[string]Loader
}

// NewDirectoryLoader validates and normalizes a DirectoryLoader config.
func NewDirectoryLoader(extensions []string, recursive bool, loaders map[string]Loader) (*DirectoryLoader, error) {
	if len(extensions) == 0 {
		return nil, errors.New("loader: DirectoryLoader requires at least one extension")
	}
	norm := make([]string, 0, len(extensions))
	for _, e := range extensions {
		e = strings.ToLower(e)
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
		norm = append(norm, e)
	}
	l := &DirectoryLoader{
		Extensions: norm,
		Recursive:  recursive,
		Loaders:    make(map[string]Loader, len(loaders)),
	}
	for e, ld := range loaders {
		if ld == nil {
			return nil, fmt.Errorf("loader: nil loader for extension %q", e)
		}
		l.Loaders[strings.ToLower(e)] = ld
	}
	return l, nil
}

// Load walks the directory at dir (a file path is an error) and loads every
// matching file. Results are ordered by path for determinism.
func (d *DirectoryLoader) Load(ctx context.Context, dir string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("loader: stat %s: %w", dir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("loader: %s is not a directory", dir)
	}
	allowed := make(map[string]bool, len(d.Extensions))
	for _, e := range d.Extensions {
		allowed[e] = true
	}
	var (
		docs  []*Document
		fails []error
	)
	walkErr := filepath.WalkDir(dir, func(path string, de fs.DirEntry, err error) error {
		if err != nil {
			fails = append(fails, fmt.Errorf("loader: %s: %w", path, err))
			return nil
		}
		if de.IsDir() {
			if !d.Recursive {
				if path != dir {
					return filepath.SkipDir
				}
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if !allowed[ext] {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		ld, ok := d.Loaders[ext]
		if !ok {
			ld, err = ForExtension(ext)
			if err != nil {
				fails = append(fails, fmt.Errorf("loader: %s: %w", path, err))
				return nil
			}
		}
		loaded, err := ld.Load(ctx, path)
		if err != nil {
			fails = append(fails, err)
			return nil
		}
		docs = append(docs, loaded...)
		return nil
	})
	if walkErr != nil {
		return docs, errors.Join(walkErr, errors.Join(fails...))
	}
	if len(fails) > 0 {
		return docs, errors.Join(fails...)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("loader: no matching files in %s (extensions %v)", dir, d.Extensions)
	}
	return docs, nil
}
