package loader

import (
	"archive/zip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"strings"

	"github.com/deagy/recall/core"
)

// DocxLoader loads a .docx file (a ZIP of OOXML parts) as a single document
// by extracting the paragraph text of word/document.xml. Only the standard
// library is used: no third-party Office dependency is required for text
// extraction.
type DocxLoader struct {
	// Parts limits how many ZIP entries are opened; 0 means no limit.
	Parts int
}

// Load reads the docx at path and extracts its document text.
func (l *DocxLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("loader: open %s: %w", path, err)
	}
	defer zr.Close()

	var doc *zip.File
	for i, f := range zr.File {
		if l.Parts > 0 && i >= l.Parts {
			break
		}
		if f.Name == "word/document.xml" {
			doc = f
			break
		}
	}
	if doc == nil {
		return nil, fmt.Errorf("loader: %s is not a valid docx (missing word/document.xml)", path)
	}
	rc, err := doc.Open()
	if err != nil {
		return nil, fmt.Errorf("loader: open %s: %w", path, err)
	}
	defer rc.Close()

	text, err := extractDocxText(rc)
	if err != nil {
		return nil, fmt.Errorf("loader: %s: %w", path, err)
	}
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("loader: %s has no extractable text", path)
	}
	d := NewDocument(path, baseName(path), path, text)
	d.Metadata["format"] = core.String{Value: "docx"}
	return []*Document{d}, nil
}

// extractDocxText streams word/document.xml and collects text runs:
// each <w:p> paragraph starts a new line, each <w:t> contributes text,
// and table cell breaks (<w:tab/>) become spaces.
func extractDocxText(r io.Reader) (string, error) {
	dec := xml.NewDecoder(r)
	var sb strings.Builder
	inPara := false
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("xml: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "p":
				if inPara {
					sb.WriteByte('\n')
				}
				inPara = true
			case "t":
				var text string
				if err := dec.DecodeElement(&text, &t); err != nil {
					return "", fmt.Errorf("xml: %w", err)
				}
				sb.WriteString(text)
			case "tab":
				sb.WriteByte('\t')
			case "br":
				sb.WriteByte('\n')
			}
		}
	}
	return normalizeWhitespace(sb.String()), nil
}
