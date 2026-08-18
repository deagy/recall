package chunker

import (
	"strings"
	"sync"

	"github.com/deagy/recall/core"
)

// Metadata keys linking child (search) chunks to their parent (context)
// chunks.
const (
	// MetaParentID holds the ID of the parent chunk on each child.
	MetaParentID = "parent_id"

	// MetaSectionIndex holds the zero-based section index on chunks
	// produced by DocumentAwareChunker.
	MetaSectionIndex = "section_index"

	// MetaSectionCount holds the total section count.
	MetaSectionCount = "section_count"
)

// ParentChildChunker implements parent-document chunking: documents are
// chunked at two granularities. Small "child" chunks are what gets
// embedded and searched (fine-grained matching), while large "parent"
// chunks carry the full context handed to the generator. Each child is
// annotated with MetaParentID so callers can expand hits back to their
// parent before context assembly.
type ParentChildChunker struct {
	parent Chunker
	child  Chunker

	mu      sync.RWMutex
	parents map[string]map[string]*core.Chunk // docID -> parentID -> parent
}

// NewParentChild creates a ParentChildChunker. The parent chunker
// produces large context chunks; the child chunker produces the small
// chunks used for retrieval.
func NewParentChild(parent, child Chunker) *ParentChildChunker {
	return &ParentChildChunker{
		parent:  parent,
		child:   child,
		parents: make(map[string]map[string]*core.Chunk),
	}
}

// Chunk splits the document, returns the child chunks (each tagged with
// MetaParentID), and caches the parent chunks for later expansion.
func (pc *ParentChildChunker) Chunk(doc *core.Document, content string) ([]*core.Chunk, error) {
	if content == "" {
		return nil, nil
	}
	parents, err := pc.parent.Chunk(doc, content)
	if err != nil {
		return nil, err
	}
	children, err := pc.child.Chunk(doc, content)
	if err != nil {
		return nil, err
	}

	byID := make(map[string]*core.Chunk, len(parents))
	for _, p := range parents {
		byID[p.ID] = p
	}
	pc.mu.Lock()
	pc.parents[doc.ID] = byID
	pc.mu.Unlock()

	for _, c := range children {
		if p := pc.matchParent(parents, c); p != nil {
			if c.Metadata == nil {
				c.Metadata = make(map[string]core.Value)
			}
			c.Metadata[MetaParentID] = core.String{Value: p.ID}
		}
	}
	return children, nil
}

// matchParent finds the parent chunk that best contains a child. It
// anchors on the child's leading 64 characters (trimmed) and picks the
// earliest parent containing it; if the anchor is unique enough to
// match several parents (overlap regions), the first one in document
// order wins. Returns nil when no parent contains the anchor.
func (pc *ParentChildChunker) matchParent(parents []*core.Chunk, child *core.Chunk) *core.Chunk {
	anchor := child.Content
	if len(anchor) > 64 {
		anchor = anchor[:64]
	}
	anchor = strings.TrimSpace(anchor)
	if anchor == "" {
		return nil
	}
	for _, p := range parents {
		if strings.Contains(p.Content, anchor) {
			return p
		}
	}
	// Fallback for children whose leading text was altered by chunking
	// (e.g. separator trims): anchor on the first 16 characters.
	if len(child.Content) > 16 {
		short := strings.TrimSpace(child.Content[:16])
		for _, p := range parents {
			if short != "" && strings.Contains(p.Content, short) {
				return p
			}
		}
	}
	return nil
}

// ParentFor returns the parent chunk for a child produced by Chunk.
func (pc *ParentChildChunker) ParentFor(docID string, child *core.Chunk) (*core.Chunk, bool) {
	if child == nil || child.Metadata == nil {
		return nil, false
	}
	pid := child.GetMetadataString(MetaParentID)
	if pid == "" {
		return nil, false
	}
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	docParents, ok := pc.parents[docID]
	if !ok {
		return nil, false
	}
	p, ok := docParents[pid]
	return p, ok
}

// Parents returns the cached parent chunks for a document in document
// order.
func (pc *ParentChildChunker) Parents(docID string) []*core.Chunk {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return orderedParents(pc.parents[docID])
}

// orderedParents orders a parent map by ChunkIndex.
func orderedParents(m map[string]*core.Chunk) []*core.Chunk {
	if len(m) == 0 {
		return nil
	}
	out := make([]*core.Chunk, 0, len(m))
	for _, p := range m {
		out = append(out, p)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].ChunkIndex < out[j-1].ChunkIndex; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// ExpandChunks maps a set of child chunks to their de-duplicated parent
// chunks, preserving first-seen order. Children without a cached parent
// (e.g. from an older document) are passed through as-is so no result
// is lost.
func (pc *ParentChildChunker) ExpandChunks(docID string, children []*core.Chunk) []*core.Chunk {
	seen := make(map[string]bool)
	var out []*core.Chunk
	for _, c := range children {
		parent, ok := pc.ParentFor(docID, c)
		if !ok {
			out = append(out, c)
			continue
		}
		if !seen[parent.ID] {
			seen[parent.ID] = true
			out = append(out, parent)
		}
	}
	return out
}
