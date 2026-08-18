package loader

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/deagy/recall/core"
)

// MarkdownLoader loads a Markdown file, splitting it into one document per
// ATX heading section. The section before the first heading (if non-empty)
// is emitted as an "intro" document. Each document carries its heading,
// level, and breadcrumb path in metadata, and gets a slug-based ID suitable
// for cross-run stability.
type MarkdownLoader struct {
	// IncludeHeading, when true, prepends the heading line to each section's
	// content so embedded text is self-describing. Default false.
	IncludeHeading bool
}

type mdSection struct {
	title   string
	level   int
	path    []string // ancestor headings, oldest first (excludes own title)
	content string
}

// Load reads the file and splits it into heading sections.
func (l *MarkdownLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("loader: read %s: %w", path, err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("loader: %s is empty", path)
	}
	sections := splitMarkdown(string(data))
	if len(sections) == 0 {
		return nil, fmt.Errorf("loader: %s has no non-empty sections", path)
	}
	docs := make([]*Document, 0, len(sections))
	used := make(map[string]int)
	for _, s := range sections {
		title := s.title
		if title == "" {
			title = "intro"
		}
		idSlug := slug(strings.Join(append(append([]string{}, s.path...), title), "-"))
		if n, dup := used[idSlug]; dup {
			used[idSlug] = n + 1
			idSlug = fmt.Sprintf("%s-%d", idSlug, n+1)
		} else {
			used[idSlug] = 1
		}
		content := s.content
		if l.IncludeHeading && s.title != "" {
			hashes := strings.Repeat("#", s.level)
			content = hashes + " " + s.title + "\n" + strings.TrimLeft(s.content, " \n")
		}
		doc := NewDocument(path+"#"+idSlug, title, path, strings.TrimSpace(content))
		crumbs := strings.Join(append(append([]string{}, s.path...), title), " > ")
		doc.Metadata["heading"] = core.String{Value: title}
		doc.Metadata["section_path"] = core.String{Value: crumbs}
		if s.title != "" {
			doc.Metadata["level"] = core.Number{Value: float64(s.level)}
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

// splitMarkdown splits text into sections at ATX headings (1–3 leading
// spaces, "#"+1..6, space, title). Blank-only sections are dropped.
func splitMarkdown(text string) []*mdSection {
	var (
		sections []*mdSection
		stack    []mdSection // open sections, indexed by level
		cur      *mdSection
	)
	openSection := func(title string, level int) {
		s := &mdSection{title: title, level: level}
		s.path = make([]string, len(stack))
		for i, st := range stack {
			s.path[i] = st.title
		}
		sections = append(sections, s)
		cur = s
	}
	for _, line := range strings.Split(text, "\n") {
		if title, level, ok := parseHeading(line); ok {
			for len(stack) > 0 && stack[len(stack)-1].level >= level {
				stack = stack[:len(stack)-1]
			}
			openSection(title, level)
			stack = append(stack, *cur)
		} else if cur != nil {
			cur.content += line + "\n"
		} else {
			// Preamble line: remember it if it is non-blank.
			trimmed := strings.TrimSpace(line)
			if trimmed != "" && cur == nil && len(sections) == 0 {
				openSection("", 0)
				cur.content += line + "\n"
			}
		}
	}
	// Drop sections whose content (excluding heading) is blank.
	kept := sections[:0]
	for _, s := range sections {
		if strings.TrimSpace(s.content) != "" {
			kept = append(kept, s)
		}
	}
	return kept
}

// parseHeading reports whether line is an ATX heading and returns its title
// and level. Setext headings, headings deeper than level 6, and missing
// space after hashes are not treated as headings.
func parseHeading(line string) (title string, level int, ok bool) {
	indent := 0
	for indent < len(line) && line[indent] == ' ' {
		indent++
		if indent > 3 {
			return "", 0, false
		}
	}
	rest := line[indent:]
	level = 0
	for level < len(rest) && rest[level] == '#' {
		level++
	}
	if level < 1 || level > 6 || level == len(rest) || rest[level] != ' ' {
		return "", 0, false
	}
	title = strings.TrimSpace(rest[level+1:])
	// Strip a closing hash sequence: a trailing run of only '#' preceded by a space.
	if i := strings.LastIndexByte(title, '#'); i > 0 {
		allHash := true
		for _, c := range title[i:] {
			if c != '#' {
				allHash = false
				break
			}
		}
		if allHash && title[i-1] == ' ' {
			title = strings.TrimRight(title[:i], " ")
		}
	}
	if title == "" {
		return "", 0, false
	}
	return title, level, true
}
