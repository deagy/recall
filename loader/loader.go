// Package loader reads documents from file-based sources into a uniform
// in-memory representation ready for chunking and upload.
//
// Loaders are pure readers: they do not embed, chunk, or store anything.
// The stdlib-only loaders (text, markdown, CSV, JSON, directory) cover the
// common local-file sources; binary formats (PDF, DOCX, HTML) are separate
// concerns that may require additional dependencies.
package loader

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/deagy/recall/core"
)

// Document is the unit produced by a Loader: one piece of source content
// with stable identity and metadata. It is designed to feed directly into
// store.Upload (Content) with core.Document fields mapped one-to-one.
type Document struct {
	// ID is a stable, source-scoped identifier for this document.
	ID string

	// Title is a human-readable title.
	Title string

	// Source is the origin the document was loaded from (file path, etc.).
	Source string

	// Content is the raw text content.
	Content string

	// Metadata carries loader-specific structured attributes (heading,
	// row/column fields, ...).
	Metadata map[string]core.Value
}

// NewDocument creates a loader.Document with empty metadata initialized.
func NewDocument(id, title, source, content string) *Document {
	return &Document{
		ID:       id,
		Title:    title,
		Source:   source,
		Content:  content,
		Metadata: make(map[string]core.Value),
	}
}

// Loader reads documents from a source. The ref parameter is loader-specific
// (a file path for most loaders, a directory for DirectoryLoader).
type Loader interface {
	Load(ctx context.Context, ref string) ([]*Document, error)
}

// ForExtension returns the default Loader for a file extension such as
// ".txt" or ".md". Extensions are case-insensitive. It returns an error for
// extensions with no default loader; callers can register their own via
// DirectoryLoader.Loaders.
func ForExtension(ext string) (Loader, error) {
	switch strings.ToLower(ext) {
	case ".txt", ".text":
		return &TextLoader{}, nil
	case ".md", ".markdown":
		return &MarkdownLoader{}, nil
	case ".csv":
		return &CSVLoader{}, nil
	case ".json":
		return &JSONLoader{}, nil
	default:
		return nil, &UnsupportedExtError{Ext: ext}
	}
}

// UnsupportedExtError is returned when no loader is registered for an extension.
type UnsupportedExtError struct {
	Ext string
}

func (e *UnsupportedExtError) Error() string {
	return "loader: no default loader for extension " + e.Ext
}

// slug converts a heading into a stable URL-style identifier:
// lowercase, runs of non-alphanumerics collapsed to "-".
func slug(s string) string {
	var b strings.Builder
	prevDash := false
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		switch {
		case r >= 'a' && r <= 'z' || r >= '0' && r <= '9':
			b.WriteRune(r)
			prevDash = false
		default:
			if !prevDash && b.Len() > 0 {
				b.WriteByte('-')
				prevDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

// baseName is the file name with extension, used as a document title.
func baseName(path string) string {
	base := filepath.Base(path)
	if i := strings.LastIndex(base, "."); i > 0 {
		return base[:i]
	}
	return base
}
