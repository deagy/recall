package connector

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

// DefaultRateLimit requests per second when a connector's RateLimit is zero.
const DefaultRateLimit = 4.0

// WebConnector fetches documents from HTTP(S) URLs. HTML responses are
// converted to extracted text by default; plain text, markdown, and JSON
// pass through raw.
type WebConnector struct {
	// Client is the HTTP client to use; default http.DefaultClient.
	Client *http.Client

	// RateLimit is the maximum sustained requests per second across all
	// Fetch calls (0 means DefaultRateLimit).
	RateLimit float64

	// MaxBytes caps the response body; 0 means 10 MiB.
	MaxBytes int64

	// RawHTML, when true, keeps raw HTML bodies verbatim instead of
	// extracting visible text (the default).
	RawHTML bool

	// AcceptContentTypes filters which responses are loaded; empty means
	// the built-in text types.
	AcceptContentTypes []string

	mu   sync.Mutex
	next time.Time
}

// Name implements Connector.
func (w *WebConnector) Name() string { return "web" }

// Fetch retrieves the document at the given URL.
func (w *WebConnector) Fetch(ctx context.Context, ref string) ([]*loader.Document, error) {
	u, err := url.Parse(ref)
	if err != nil {
		return nil, fmt.Errorf("web: invalid URL %q: %w", ref, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("web: unsupported scheme %q", u.Scheme)
	}
	if err := w.wait(ctx); err != nil {
		return nil, err
	}
	client := w.Client
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ref, nil)
	if err != nil {
		return nil, fmt.Errorf("web: %w", err)
	}
	req.Header.Set("User-Agent", "recall-ingest/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("web: fetch %s: %w", ref, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("web: %s returned %s", ref, resp.Status)
	}
	maxBytes := w.MaxBytes
	if maxBytes == 0 {
		maxBytes = 10 << 20
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("web: read %s: %w", ref, err)
	}
	if int64(len(body)) > maxBytes {
		return nil, fmt.Errorf("web: %s exceeds %d byte limit", ref, maxBytes)
	}
	contentType := resp.Header.Get("Content-Type")
	if !w.accepted(contentType) {
		return nil, fmt.Errorf("web: %s has unsupported content type %q", ref, contentType)
	}
	content := string(body)
	title := u.Host
	if !w.RawHTML && strings.Contains(strings.ToLower(contentType), "html") {
		text, htmlTitle, err := loader.ExtractHTML(body, false)
		if err != nil {
			return nil, fmt.Errorf("web: extract %s: %w", ref, err)
		}
		content = text
		if htmlTitle != "" {
			title = htmlTitle
		}
	}
	d := loader.NewDocument(ref, title, ref, content)
	d.Metadata["content_type"] = core.String{Value: contentType}
	d.Metadata["final_url"] = core.String{Value: resp.Request.URL.String()}
	return []*loader.Document{d}, nil
}

// accepted reports whether contentType passes the accept filter.
func (w *WebConnector) accepted(contentType string) bool {
	accepted := w.AcceptContentTypes
	if len(accepted) == 0 {
		accepted = []string{"text/html", "text/plain", "text/markdown", "application/json", "application/xml", "text/xml"}
	}
	ct := strings.ToLower(strings.TrimSpace(contentType))
	if i := strings.IndexByte(ct, ';'); i >= 0 {
		ct = strings.TrimSpace(ct[:i])
	}
	for _, a := range accepted {
		if ct == strings.ToLower(a) || strings.HasPrefix(ct, strings.ToLower(a)+"/") {
			return true
		}
	}
	return false
}

// wait blocks until the rate limit allows another request.
func (w *WebConnector) wait(ctx context.Context) error {
	rate := w.RateLimit
	if rate <= 0 {
		rate = DefaultRateLimit
	}
	minGap := time.Second / time.Duration(rate)
	w.mu.Lock()
	now := time.Now()
	start := now
	if now.Before(w.next) {
		start = w.next
	}
	w.next = start.Add(minGap)
	w.mu.Unlock()
	if !start.After(time.Now()) {
		return nil
	}
	timer := time.NewTimer(time.Until(start))
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
