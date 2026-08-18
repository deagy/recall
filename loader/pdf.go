package loader

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/deagy/recall/core"
	"github.com/ledongthuc/pdf"
)

// PDFLoader loads a PDF file as a single document by extracting its plain
// text. Encrypted or scanned (image-only) PDFs yield an error respectively.
type PDFLoader struct{}

// Load reads and extracts text from the PDF at path.
func (l *PDFLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("loader: open %s: %w", path, err)
	}
	defer f.Close()
	size, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return nil, fmt.Errorf("loader: stat %s: %w", path, err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("loader: seek %s: %w", path, err)
	}
	r, err := pdf.NewReader(f, size)
	if err != nil {
		return nil, fmt.Errorf("loader: open %s: %w", path, err)
	}
	numPages := r.NumPage()
	if numPages == 0 {
		return nil, fmt.Errorf("loader: %s has no pages", path)
	}
	pr, err := r.GetPlainText()
	if err != nil {
		return nil, fmt.Errorf("loader: extract %s: %w", path, err)
	}
	raw, err := io.ReadAll(pr)
	if err != nil {
		return nil, fmt.Errorf("loader: read %s text: %w", path, err)
	}
	text := normalizeWhitespace(string(raw))
	if text == "" {
		return nil, fmt.Errorf("loader: %s has no extractable text (scanned or encrypted?)", path)
	}
	d := NewDocument(path, baseName(path), path, text)
	d.Metadata["pages"] = core.Number{Value: float64(numPages)}
	return []*Document{d}, nil
}
