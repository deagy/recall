package loader

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTempHTML writes content to a temp .html file and returns its path.
func writeTempHTML(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "page.html")
	if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
		t.Fatalf("write html: %v", err)
	}
	return p
}

func TestHTMLLoader_Basic(t *testing.T) {
	p := writeTempHTML(t, `<!DOCTYPE html>
<html>
<head><title>My Title</title><style>body{color:red}</style></head>
<body>
<script>var x = "should not appear";</script>
<nav>Menu Item</nav>
<header>Site Header</header>
<h1>Heading One</h1>
<p>First   paragraph.</p>
<p>Second paragraph.</p>
<footer>Footer Text</footer>
</body>
</html>`)
	docs, err := (&HTMLLoader{}).Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	d := docs[0]
	if d.Title != "My Title" {
		t.Errorf("title: got %q", d.Title)
	}
	for _, want := range []string{"Heading One", "First paragraph.", "Second paragraph."} {
		if !strings.Contains(d.Content, want) {
			t.Errorf("content missing %q; got %q", want, d.Content)
		}
	}
	for _, banned := range []string{"should not appear", "Menu Item", "Site Header", "Footer Text", "color:red"} {
		if strings.Contains(d.Content, banned) {
			t.Errorf("content unexpectedly contains %q", banned)
		}
	}
}

func TestHTMLLoader_KeepChrome(t *testing.T) {
	p := writeTempHTML(t, `<html><body><nav>NavText</nav><p>Body</p></body></html>`)
	docs, err := (&HTMLLoader{KeepChrome: true}).Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !strings.Contains(docs[0].Content, "NavText") {
		t.Errorf("KeepChrome should retain nav text, got %q", docs[0].Content)
	}
}

func TestHTMLLoader_Empty(t *testing.T) {
	p := writeTempHTML(t, `<html><body><script>only script</script></body></html>`)
	if _, err := (&HTMLLoader{}).Load(context.Background(), p); err == nil {
		t.Fatal("expected error for script-only page")
	}
}

func TestHTMLLoader_MissingFile(t *testing.T) {
	if _, err := (&HTMLLoader{}).Load(context.Background(), "/definitely/missing.html"); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestForExtension_HTML(t *testing.T) {
	for _, ext := range []string{".html", ".htm"} {
		l, err := ForExtension(ext)
		if err != nil || l == nil {
			t.Fatalf("ForExtension %s: %v", ext, err)
		}
		if _, ok := l.(*HTMLLoader); !ok {
			t.Errorf("got %T", l)
		}
	}
}

func TestNormalizeWhitespace(t *testing.T) {
	got := normalizeWhitespace("  a   b  \n\n\n\n  c  \n")
	if got != "a b\nc" {
		t.Errorf("got %q", got)
	}
}
