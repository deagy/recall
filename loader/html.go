package loader

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/deagy/recall/core"
	"golang.org/x/net/html"
)

// htmlAlwaysSkipped contains elements that never contribute readable text.
var htmlAlwaysSkipped = map[string]bool{
	"script": true, "style": true, "noscript": true, "template": true,
	"iframe": true, "svg": true, "canvas": true, "object": true,
}

// htmlChromeSkipped are page-chrome elements dropped unless KeepChrome is set.
var htmlChromeSkipped = map[string]bool{
	"nav": true, "header": true, "footer": true, "aside": true, "form": true,
}

// htmlBlockElements produce a line break boundary in the extracted text.
var htmlBlockElements = map[string]bool{
	"p": true, "div": true, "br": true, "li": true, "tr": true, "table": true,
	"section": true, "article": true, "blockquote": true, "pre": true,
	"ul": true, "ol": true, "h1": true, "h2": true, "h3": true,
	"h4": true, "h5": true, "h6": true, "main": true, "figure": true,
}

// HTMLLoader loads an HTML file as a single document with its visible text
// extracted: script/style/nav/header/footer content is dropped, block
// elements become line breaks, and whitespace runs are normalized.
type HTMLLoader struct {
	// KeepChrome, when true, does not drop nav/header/footer/aside/form
	// elements (their text is kept).
	KeepChrome bool
}

// Load parses the HTML file at path.
func (l *HTMLLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("loader: read %s: %w", path, err)
	}
	text, title, err := ExtractHTML(data, l.KeepChrome)
	if err != nil {
		return nil, fmt.Errorf("loader: %s: %w", path, err)
	}
	d := NewDocument(path, baseName(path), path, text)
	if title != "" {
		d.Title = title
		d.Metadata["title"] = core.String{Value: title}
	}
	return []*Document{d}, nil
}

// ExtractHTML parses raw HTML and returns the normalized visible text plus
// the document title ("" when absent). keepChrome controls whether
// nav/header/footer/aside/form elements are retained.
func ExtractHTML(data []byte, keepChrome bool) (content, title string, err error) {
	doc, err := html.Parse(bytes.NewReader(data))
	if err != nil {
		return "", "", err
	}
	content = extractHTMLText(doc, keepChrome)
	if strings.TrimSpace(content) == "" {
		return "", "", fmt.Errorf("no extractable text")
	}
	title = extractHTMLTitle(doc)
	return content, title, nil
}

// extractHTMLText walks a parsed HTML tree, collecting text nodes outside
// skipped elements and emitting newlines at block boundaries.
func extractHTMLText(root *html.Node, keepChrome bool) string {
	var sb strings.Builder
	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			if htmlAlwaysSkipped[n.Data] || (!keepChrome && htmlChromeSkipped[n.Data]) {
				return // drop the entire subtree
			}
			if htmlBlockElements[n.Data] {
				sb.WriteByte('\n')
			}
		}
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(root)
	return normalizeWhitespace(sb.String())
}

// extractHTMLTitle returns the text of the first <title> element, if any.
func extractHTMLTitle(root *html.Node) string {
	var title string
	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		if title != "" {
			return
		}
		if n.Type == html.ElementNode && n.Data == "title" {
			var sb strings.Builder
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.TextNode {
					sb.WriteString(c.Data)
				}
			}
			title = strings.TrimSpace(sb.String())
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(root)
	return title
}

// normalizeWhitespace collapses each line's internal whitespace, drops
// blank lines, and trims the overall result.
func normalizeWhitespace(s string) string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line == "" {
			continue
		}
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}
