package chunker

import (
	"strings"

	"github.com/deagy/recall/core"
)

// DefaultBoundary is the separator that marks a document/section break
// inside a concatenated content string.
const DefaultBoundary = "\n---\n"

// DocumentAwareChunker wraps an inner chunker and enforces document
// boundaries: content is first split on an explicit boundary marker
// (default "---" on its own line), and each segment is chunked
// independently. No chunk ever contains text from two different
// documents/sections, and overlap never crosses a boundary. Each
// chunk is tagged with MetaSectionIndex / MetaSectionCount.
type DocumentAwareChunker struct {
	Inner Chunker

	// Boundary marks a document/section break. Empty means
	// DefaultBoundary.
	Boundary string
}

// NewDocumentAware creates a DocumentAwareChunker around inner.
func NewDocumentAware(inner Chunker) *DocumentAwareChunker {
	return &DocumentAwareChunker{Inner: inner, Boundary: DefaultBoundary}
}

// Chunk splits content on boundaries and chunks each segment with the
// inner chunker.
func (d *DocumentAwareChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}
	boundary := d.Boundary
	if boundary == "" {
		boundary = DefaultBoundary
	}
	segments := strings.Split(content, boundary)

	var chunks []*core.Chunk
	for i, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		segChunks, err := d.Inner.Chunk(doc, seg)
		if err != nil {
			return nil, err
		}
		for _, c := range segChunks {
			if c.Metadata == nil {
				c.Metadata = make(map[string]core.Value)
			}
			c.Metadata[MetaSectionIndex] = core.Number{Value: float64(i)}
			c.Metadata[MetaSectionCount] = core.Number{Value: float64(len(segments))}
			chunks = append(chunks, c)
		}
	}
	return chunks, nil
}
