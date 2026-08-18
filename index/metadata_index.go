package index

import (
	"sort"
	"sync"

	"github.com/deagy/recall/core"
)

// MetadataIndex is an inverted index over chunk metadata that answers
// "which chunks could match these filters?" in set operations instead
// of scanning every chunk. It complements vector indexes: query the
// metadata index first to get a candidate set, then restrict the
// vector search to it (or use Candidates directly for pure metadata
// lookups).
type MetadataIndex struct {
	mu sync.RWMutex

	// chunks keeps the full chunk so generic filters can fall back
	// to evaluating Match on candidates.
	chunks map[string]*core.Chunk

	// terms is the inverted index: key -> value -> chunk IDs. Values are
	// read through GetMetadataString, so every metadata entry has at
	// most one indexed value per key.
	terms map[string]map[string]map[string]struct{}
}

// NewMetadataIndex creates an empty MetadataIndex.
func NewMetadataIndex() *MetadataIndex {
	return &MetadataIndex{
		chunks: make(map[string]*core.Chunk),
		terms:  make(map[string]map[string]map[string]struct{}),
	}
}

// Add indexes a chunk's metadata. Re-adding an existing ID replaces the
// previous entry.
func (mi *MetadataIndex) Add(chunk *core.Chunk) error {
	if chunk == nil {
		return core.ErrInvalidChunk
	}
	mi.mu.Lock()
	defer mi.mu.Unlock()

	// Remove any previous postings for this ID before replacing.
	mi.removePostingsLocked(chunk.ID)

	mi.chunks[chunk.ID] = chunk
	for key, val := range chunk.Metadata {
		if val == nil {
			continue
		}
		str := val.String()
		bucket := mi.terms[key]
		if bucket == nil {
			bucket = make(map[string]map[string]struct{})
			mi.terms[key] = bucket
		}
		ids := bucket[str]
		if ids == nil {
			ids = make(map[string]struct{})
			bucket[str] = ids
		}
		ids[chunk.ID] = struct{}{}
	}
	return nil
}

// Remove deletes a chunk and its postings.
func (mi *MetadataIndex) Remove(id string) error {
	mi.mu.Lock()
	defer mi.mu.Unlock()
	if _, ok := mi.chunks[id]; !ok {
		return core.ErrNotFound
	}
	delete(mi.chunks, id)
	mi.removePostingsLocked(id)
	return nil
}

// removePostingsLocked drops id from every posting bucket. Caller holds
// the write lock.
func (mi *MetadataIndex) removePostingsLocked(id string) {
	for key, bucket := range mi.terms {
		for val, ids := range bucket {
			if _, ok := ids[id]; !ok {
				continue
			}
			delete(ids, id)
			if len(ids) == 0 {
				delete(bucket, val)
			}
		}
		if len(bucket) == 0 {
			delete(mi.terms, key)
		}
	}
}

// Get returns a stored chunk by ID.
func (mi *MetadataIndex) Get(id string) (*core.Chunk, bool) {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	c, ok := mi.chunks[id]
	return c, ok
}

// Count returns the number of indexed chunks.
func (mi *MetadataIndex) Count() int {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	return len(mi.chunks)
}

// Values lists the indexed metadata values for a key, sorted.
func (mi *MetadataIndex) Values(key string) []string {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	bucket := mi.terms[key]
	if bucket == nil {
		return nil
	}
	out := make([]string, 0, len(bucket))
	for v := range bucket {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// Candidates evaluates the given filters and returns the set of chunk
// IDs that could match, or (nil, false) when there are no filters
// (meaning "no pre-filter needed").
//
// TermFilter and TermInFilter are answered with posting-list
// intersections/unions; any other Filter is evaluated by scanning the
// current candidate set with Filter.Match.
func (mi *MetadataIndex) Candidates(filters []Filter) (map[string]struct{}, bool) {
	if len(filters) == 0 {
		return nil, false
	}
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	candidates := make(map[string]struct{}, len(mi.chunks))
	for id := range mi.chunks {
		candidates[id] = struct{}{}
	}

	for _, f := range filters {
		switch tf := f.(type) {
		case *TermFilter:
			postings := mi.postingsLocked(tf.Key, tf.Value)
			candidates = intersect(candidates, postings)
		case *TermInFilter:
			union := make(map[string]struct{})
			for _, v := range tf.Values {
				for id := range mi.postingsLocked(tf.Key, v) {
					union[id] = struct{}{}
				}
			}
			candidates = intersect(candidates, union)
		default:
			// Generic fallback: evaluate Match on candidates.
			narrowed := make(map[string]struct{}, len(candidates))
			for id := range candidates {
				if f.Match(mi.chunks[id]) {
					narrowed[id] = struct{}{}
				}
			}
			candidates = narrowed
		}
		if len(candidates) == 0 {
			break
		}
	}
	return candidates, true
}

// postingsLocked returns the posting list for key/value (never nil).
func (mi *MetadataIndex) postingsLocked(key, value string) map[string]struct{} {
	if bucket := mi.terms[key]; bucket != nil {
		if ids := bucket[value]; ids != nil {
			return ids
		}
	}
	return make(map[string]struct{})
}

// intersect returns the intersection of two ID sets.
func intersect(a, b map[string]struct{}) map[string]struct{} {
	if len(b) < len(a) {
		a, b = b, a
	}
	out := make(map[string]struct{}, len(b))
	for id := range b {
		if _, ok := a[id]; ok {
			out[id] = struct{}{}
		}
	}
	return out
}

// SortedIDs converts an ID set to a sorted slice for deterministic
// iteration.
func SortedIDs(set map[string]struct{}) []string {
	out := make([]string, 0, len(set))
	for id := range set {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}
