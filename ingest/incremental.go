package ingest

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// Incremental tracks per-document content hashes so re-runs only ingest
// new or changed documents. State is keyed by document ID and persisted as
// JSON so incremental ingestion survives process restarts.
type Incremental struct {
	mu     sync.Mutex
	path   string
	hashes map[string]string
	dirty  bool
}

// NewIncremental creates an incremental tracker that persists to path
// ("" means in-memory only). Existing state is loaded when present.
func NewIncremental(path string) (*Incremental, error) {
	inc := &Incremental{path: path, hashes: make(map[string]string)}
	if path == "" {
		return inc, nil
	}
	if err := inc.Load(); err != nil {
		return nil, err
	}
	return inc, nil
}

// Load (re)reads the state file, replacing any in-memory state.
func (inc *Incremental) Load() error {
	data, err := os.ReadFile(inc.path)
	if err != nil {
		if os.IsNotExist(err) {
			inc.mu.Lock()
			inc.hashes = make(map[string]string)
			inc.mu.Unlock()
			return nil
		}
		return fmt.Errorf("ingest: load incremental %s: %w", inc.path, err)
	}
	var st struct {
		Version   int               `json:"version"`
		Documents map[string]string `json:"documents"`
	}
	if err := json.Unmarshal(data, &st); err != nil {
		return fmt.Errorf("ingest: decode incremental %s: %w", inc.path, err)
	}
	inc.mu.Lock()
	inc.hashes = st.Documents
	if inc.hashes == nil {
		inc.hashes = make(map[string]string)
	}
	inc.mu.Unlock()
	return nil
}

// ShouldSkip reports whether docID has already been ingested with this
// content hash.
func (inc *Incremental) ShouldSkip(docID, contentHash string) bool {
	inc.mu.Lock()
	defer inc.mu.Unlock()
	prev, ok := inc.hashes[docID]
	return ok && prev == contentHash
}

// Mark records that docID was ingested with the given content hash.
func (inc *Incremental) Mark(docID, contentHash string) {
	inc.mu.Lock()
	inc.hashes[docID] = contentHash
	inc.dirty = true
	inc.mu.Unlock()
}

// Forget removes docID from the tracked state (e.g. after deletion).
func (inc *Incremental) Forget(docID string) {
	inc.mu.Lock()
	delete(inc.hashes, docID)
	inc.dirty = true
	inc.mu.Unlock()
}

// Len returns the number of tracked documents.
func (inc *Incremental) Len() int {
	inc.mu.Lock()
	defer inc.mu.Unlock()
	return len(inc.hashes)
}

// Save writes the state file if it changed since the last save.
func (inc *Incremental) Save() error {
	inc.mu.Lock()
	if inc.path == "" {
		inc.mu.Unlock()
		return nil
	}
	if !inc.dirty {
		inc.mu.Unlock()
		return nil
	}
	ids := make([]string, 0, len(inc.hashes))
	for id := range inc.hashes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	st := struct {
		Version   int               `json:"version"`
		Documents map[string]string `json:"documents"`
	}{Version: 1, Documents: inc.hashes}
	inc.dirty = false
	inc.mu.Unlock()
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return fmt.Errorf("ingest: encode incremental: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(inc.path, data, 0o600); err != nil {
		return fmt.Errorf("ingest: save incremental %s: %w", inc.path, err)
	}
	return nil
}
