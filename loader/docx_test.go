package loader

import (
	"archive/zip"
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTempDocx builds a minimal docx (ZIP with word/document.xml) and
// returns its path.
func writeTempDocx(t *testing.T, docXML string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "note.docx")
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create("word/document.xml")
	if err != nil {
		t.Fatalf("zip create: %v", err)
	}
	if _, err := w.Write([]byte(docXML)); err != nil {
		t.Fatalf("zip write: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write docx: %v", err)
	}
	return p
}

const docxNS = `xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"`

func TestDocxLoader_Basic(t *testing.T) {
	xml := `<w:document ` + docxNS + `><w:body>` +
		`<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>world</w:t></w:r></w:p>` +
		`<w:p><w:r><w:t>Second para</w:t></w:r></w:p>` +
		`</w:body></w:document>`
	p := writeTempDocx(t, xml)
	docs, err := (&DocxLoader{}).Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if !strings.Contains(docs[0].Content, "Hello world") || !strings.Contains(docs[0].Content, "Second para") {
		t.Errorf("unexpected content %q", docs[0].Content)
	}
}

func TestDocxLoader_NotZip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.docx")
	if err := os.WriteFile(p, []byte("not a zip"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := (&DocxLoader{}).Load(context.Background(), p); err == nil {
		t.Fatal("expected error for non-zip file")
	}
}

func TestDocxLoader_MissingPart(t *testing.T) {
	p := writeTempDocx(t, `<root/>`)
	if _, err := (&DocxLoader{}).Load(context.Background(), p); err == nil {
		t.Fatal("expected error for docx without word/document.xml")
	}
}

func TestForExtension_Docx(t *testing.T) {
	l, err := ForExtension(".docx")
	if err != nil || l == nil {
		t.Fatalf("ForExtension .docx: %v", err)
	}
	if _, ok := l.(*DocxLoader); !ok {
		t.Errorf("got %T", l)
	}
}
