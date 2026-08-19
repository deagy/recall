package embedder

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/deagy/recall/embedder/onnx"
)

// DefaultHFBaseURL is the default base for HuggingFace model downloads.
// It can be overridden per-cache (CacheBaseURL) to point at a mirror or a
// local file server for offline use.
const DefaultHFBaseURL = "https://huggingface.co"

// DefaultHFRepo is the default HuggingFace repo used when none is given:
// the canonical sentence-transformers MiniLM ONNX export.
const DefaultHFRepo = "sentence-transformers/all-MiniLM-L6-v2"

// DefaultHFFile is the default ONNX file within a HuggingFace repo.
const DefaultHFFile = "onnx/model.onnx"

// ModelCache is a small on-disk cache for ONNX model files. Entries are
// keyed by the SHA-256 of the source URL and validated by content hash
// (the filename), so identical content from different URLs shares a file
// only when explicitly requested; the primary key is the URL hash.
//
// The cache is safe for concurrent use. Downloads are deduplicated: while
// one goroutine is fetching a URL, others wait for the same result.
type ModelCache struct {
	dir     string
	ttl     time.Duration
	http    *http.Client
	baseURL string

	mu       sync.Mutex
	inflight map[string]*inflightFetch
}

// inflightFetch is the shared result of an in-progress download. Multiple
// goroutines waiting on the same URL all read path/err once done is closed,
// so the result is broadcast to every waiter (a single-value channel would
// only serve the first).
type inflightFetch struct {
	done chan struct{}
	path string
	err  error
}

// NewModelCache creates a cache rooted at dir. ttl is the maximum age of a
// cached file before it is re-downloaded (0 means cache forever). The file
// is stored under the name sha256(url).onnx.
func NewModelCache(dir string, ttl time.Duration) (*ModelCache, error) {
	if dir == "" {
		return nil, fmt.Errorf("embedder: model cache requires a directory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("embedder: model cache: %w", err)
	}
	return &ModelCache{
		dir:      dir,
		ttl:      ttl,
		http:     &http.Client{Timeout: 10 * time.Minute},
		baseURL:  DefaultHFBaseURL,
		inflight: make(map[string]*inflightFetch),
	}, nil
}

// SetBaseURL overrides the default HuggingFace base URL (for example to
// point at a mirror or a local file server for offline use). An empty
// string restores the default.
func (c *ModelCache) SetBaseURL(base string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if base == "" {
		base = DefaultHFBaseURL
	}
	c.baseURL = base
}

// HFURL builds the download URL for an ONNX file in a HuggingFace repo
// using this cache's base URL: <base>/<repo>/resolve/main/<file>.
func (c *ModelCache) HFURL(repo, file string) string {
	c.mu.Lock()
	base := c.baseURL
	c.mu.Unlock()
	return hfURL(base, repo, file)
}

// Path returns the on-disk path a given URL would be cached to.
func (c *ModelCache) Path(rawURL string) string {
	sum := sha256.Sum256([]byte(rawURL))
	return filepath.Join(c.dir, hex.EncodeToString(sum[:])+".onnx")
}

// Get returns the on-disk path for the model at rawURL, downloading it if
// the cache is empty or the entry is older than the TTL. The returned path
// is suitable for onnx.LoadFile.
func (c *ModelCache) Get(ctx context.Context, rawURL string) (string, error) {
	if _, err := url.Parse(rawURL); err != nil {
		return "", fmt.Errorf("embedder: model cache: invalid URL %q: %w", rawURL, err)
	}
	path := c.Path(rawURL)
	if c.isFresh(path) {
		return path, nil
	}
	// Deduplicate concurrent fetches of the same URL.
	c.mu.Lock()
	if in, ok := c.inflight[rawURL]; ok {
		c.mu.Unlock()
		return c.await(ctx, in)
	}
	in := &inflightFetch{done: make(chan struct{})}
	c.inflight[rawURL] = in
	c.mu.Unlock()

	p, err := c.download(ctx, rawURL, path)
	in.path, in.err = p, err
	close(in.done)
	c.mu.Lock()
	delete(c.inflight, rawURL)
	c.mu.Unlock()
	if err != nil {
		return "", err
	}
	return p, nil
}

// await waits for another goroutine's in-flight fetch, honoring ctx.
func (c *ModelCache) await(ctx context.Context, in *inflightFetch) (string, error) {
	select {
	case <-in.done:
		return in.path, in.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// isFresh reports whether path exists and is younger than the TTL.
func (c *ModelCache) isFresh(path string) bool {
	st, err := os.Stat(path)
	if err != nil {
		return false
	}
	if c.ttl <= 0 {
		return true
	}
	return time.Since(st.ModTime()) < c.ttl
}

// download fetches rawURL into path atomically (temp file + rename).
func (c *ModelCache) download(ctx context.Context, rawURL, path string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return "", fmt.Errorf("embedder: model cache: %w", err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("embedder: model cache: fetch %q: %w", rawURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("embedder: model cache: fetch %q: HTTP %d", rawURL, resp.StatusCode)
	}
	tmp, err := os.CreateTemp(c.dir, ".download-*")
	if err != nil {
		return "", fmt.Errorf("embedder: model cache: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return "", fmt.Errorf("embedder: model cache: write %q: %w", rawURL, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return "", fmt.Errorf("embedder: model cache: close %q: %w", rawURL, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return "", fmt.Errorf("embedder: model cache: move %q into place: %w", rawURL, err)
	}
	return path, nil
}

// hfURL builds the download URL for an ONNX file in a HuggingFace repo:
// <base>/<repo>/resolve/main/<file>. An empty file defaults to
// DefaultHFFile and an empty base defaults to DefaultHFBaseURL.
func hfURL(base, repo, file string) string {
	if file == "" {
		file = DefaultHFFile
	}
	if base == "" {
		base = DefaultHFBaseURL
	}
	return base + "/" + repo + "/resolve/main/" + file
}

// HuggingFaceURL builds the download URL for an ONNX file in a
// HuggingFace repo using the default base URL.
func HuggingFaceURL(repo, file string) string {
	return hfURL(DefaultHFBaseURL, repo, file)
}

// LoadHFModel downloads (or reuses a cached copy of) the ONNX model for the
// given HuggingFace repo and returns a ready-to-run *onnx.Model. repo and
// file may be empty to use the defaults (sentence-transformers
// all-MiniLM-L6-v2, onnx/model.onnx). cache may be nil, in which case the
// model is fetched without caching.
func LoadHFModel(ctx context.Context, cache *ModelCache, repo, file string) (*onnx.Model, error) {
	if repo == "" {
		repo = DefaultHFRepo
	}
	var rawURL string
	if cache != nil {
		rawURL = cache.HFURL(repo, file)
		path, err := cache.Get(ctx, rawURL)
		if err != nil {
			return nil, err
		}
		m, err := onnx.LoadFile(path)
		if err != nil {
			return nil, err
		}
		return m, nil
	}
	rawURL = HuggingFaceURL(repo, file)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("embedder: load hf model: %w", err)
	}
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedder: load hf model: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedder: load hf model: HTTP %d for %q", resp.StatusCode, rawURL)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("embedder: load hf model: read %q: %w", rawURL, err)
	}
	m, err := onnx.Load(data)
	if err != nil {
		return nil, err
	}
	return m, nil
}
