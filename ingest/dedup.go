package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// ContentHash returns the hex-encoded SHA-256 of a document's content. It
// is the canonical fingerprint used by both Deduplicator and Incremental.
func ContentHash(content string) string {
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

// Deduplicator tracks content hashes already ingested within (and across)
// runs. It is safe for concurrent use and can be persisted to a JSON file
// so duplicate suppression survives restarts.
type Deduplicator struct {
	mu   sync.Mutex
	seen map[string]bool
}

// NewDeduplicator returns an empty deduplicator.
func NewDeduplicator() *Deduplicator {
	return &Deduplicator{seen: make(map[string]bool)}
}

// LoadDeduplicator loads a persisted deduplicator from path; a missing file
// yields an empty one.
func LoadDeduplicator(path string) (*Deduplicator, error) {
	d := NewDeduplicator()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return d, nil
		}
		return nil, fmt.Errorf("ingest: load dedup %s: %w", path, err)
	}
	var hashes []string
	if err := json.Unmarshal(data, &hashes); err != nil {
		return nil, fmt.Errorf("ingest: decode dedup %s: %w", path, err)
	}
	for _, h := range hashes {
		d.seen[h] = true
	}
	return d, nil
}

// Save persists the seen hashes to path (pretty-printed JSON array).
func (d *Deduplicator) Save(path string) error {
	d.mu.Lock()
	hashes := make([]string, 0, len(d.seen))
	for h := range d.seen {
		hashes = append(hashes, h)
	}
	d.mu.Unlock()
	data, err := json.MarshalIndent(hashes, "", "  ")
	if err != nil {
		return fmt.Errorf("ingest: encode dedup: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("ingest: save dedup %s: %w", path, err)
	}
	return nil
}

// IsDuplicate reports whether the content has been seen before.
func (d *Deduplicator) IsDuplicate(content string) bool {
	h := ContentHash(content)
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.seen[h]
}

// Mark records the content as seen.
func (d *Deduplicator) Mark(content string) {
	d.mu.Lock()
	d.seen[ContentHash(content)] = true
	d.mu.Unlock()
}

// Len returns the number of distinct hashes tracked.
func (d *Deduplicator) Len() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.seen)
}
