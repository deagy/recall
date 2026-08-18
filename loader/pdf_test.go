package loader

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// pdfEscape escapes text for embedding in a PDF literal string.
func pdfEscape(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "(", "\\(")
	s = strings.ReplaceAll(s, ")", "\\)")
	return s
}

// writeTempPDF builds a minimal but valid single/multi-page PDF with a
// correct xref table and returns its path.
func writeTempPDF(t *testing.T, pages ...string) string {
	t.Helper()
	n := len(pages)
	// Layout: 1=catalog, 2=pages tree, 3=font,
	// then per page i: (4+2i)=page obj, (5+2i)=content stream.
	total := 3 + 2*n
	var body bytes.Buffer
	body.WriteString("%PDF-1.4\n")
	offsets := make([]int, total+1) // 1-based
	obj := func(num int, content []byte) {
		offsets[num] = body.Len()
		body.WriteString(fmt.Sprintf("%d 0 obj\n", num))
		body.Write(content)
		body.WriteString("\nendobj\n")
	}
	obj(1, []byte("<< /Type /Catalog /Pages 2 0 R >>"))
	var kids strings.Builder
	for i := 0; i < n; i++ {
		kids.WriteString(fmt.Sprintf("%d 0 R ", 4+2*i))
	}
	obj(2, []byte(fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", kids.String(), n)))
	obj(3, []byte("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
	for i := 0; i < n; i++ {
		pageNum, contentNum := 4+2*i, 5+2*i
		obj(pageNum, []byte(fmt.Sprintf(
			"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>",
			contentNum)))
		stream := fmt.Sprintf("BT /F1 12 Tf 72 720 Td (%s) Tj ET", pdfEscape(pages[i]))
		obj(contentNum, []byte(fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream)))
	}
	xrefOff := body.Len()
	body.WriteString(fmt.Sprintf("xref\n0 %d\n", total+1))
	body.WriteString("0000000000 65535 f \n")
	for i := 1; i <= total; i++ {
		body.WriteString(fmt.Sprintf("%010d 00000 n \n", offsets[i]))
	}
	body.WriteString(fmt.Sprintf("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", total+1, xrefOff))

	p := filepath.Join(t.TempDir(), "doc.pdf")
	if err := os.WriteFile(p, body.Bytes(), 0o600); err != nil {
		t.Fatalf("write pdf: %v", err)
	}
	return p
}

func TestPDFLoader_Basic(t *testing.T) {
	p := writeTempPDF(t, "Hello PDF", "Second page text")
	docs, err := (&PDFLoader{}).Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	d := docs[0]
	if !strings.Contains(d.Content, "Hello PDF") || !strings.Contains(d.Content, "Second page text") {
		t.Errorf("unexpected content %q", d.Content)
	}
}

func TestPDFLoader_NotPDF(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.pdf")
	if err := os.WriteFile(p, []byte("not a pdf"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := (&PDFLoader{}).Load(context.Background(), p); err == nil {
		t.Fatal("expected error for non-pdf file")
	}
}

func TestPDFLoader_NoText(t *testing.T) {
	p := writeTempPDF(t, "")
	if _, err := (&PDFLoader{}).Load(context.Background(), p); err == nil {
		t.Fatal("expected error for textless page")
	}
}

func TestForExtension_PDF(t *testing.T) {
	l, err := ForExtension(".pdf")
	if err != nil || l == nil {
		t.Fatalf("ForExtension .pdf: %v", err)
	}
	if _, ok := l.(*PDFLoader); !ok {
		t.Errorf("got %T", l)
	}
}
