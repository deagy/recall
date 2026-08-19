package analytics

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"sync"
	"time"
)

// Sink receives each recorded query. Implement this to export analytics to a
// file, an HTTP endpoint, a message queue, etc. Write must be safe for
// concurrent use and must not block indefinitely.
type Sink interface {
	// Write receives a single query record.
	Write(rec QueryRecord) error
	// Close releases any resources held by the sink.
	Close() error
}

// FileSink appends query records to a file as NDJSON (one JSON object per
// line).
type FileSink struct {
	mu      sync.Mutex
	f       *os.File
	encoder *json.Encoder
	closed  bool
}

// NewFileSink opens (creating if needed) the file at path for appending and
// returns a FileSink writing NDJSON to it.
func NewFileSink(path string) (*FileSink, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &FileSink{f: f, encoder: json.NewEncoder(f)}, nil
}

// Write appends the record as a single JSON line.
func (s *FileSink) Write(rec QueryRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return os.ErrClosed
	}
	return s.encoder.Encode(rec)
}

// Close closes the underlying file.
func (s *FileSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.f.Close()
}

// HTTPSink POSTs each query record as a JSON document to a fixed URL. It is
// suitable for forwarding analytics to a collector or message-queue HTTP
// bridge.
type HTTPSink struct {
	mu      sync.Mutex
	url     string
	client  *http.Client
	closed  bool
	lastErr error
}

// NewHTTPSink returns an HTTPSink that POSTs records to url. If client is nil,
// a client with a 2-second timeout is used so the query path is never blocked
// for long.
func NewHTTPSink(url string, client *http.Client) *HTTPSink {
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Second}
	}
	return &HTTPSink{url: url, client: client}
}

// Write POSTs the record as JSON to the sink's URL.
func (s *HTTPSink) Write(rec QueryRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return os.ErrClosed
	}
	body, err := json.Marshal(rec)
	if err != nil {
		s.lastErr = err
		return err
	}
	req, err := http.NewRequest(http.MethodPost, s.url, bytes.NewReader(body))
	if err != nil {
		s.lastErr = err
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		s.lastErr = err
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		err = &httpStatusError{code: resp.StatusCode}
		s.lastErr = err
		return err
	}
	return nil
}

// Close marks the sink closed.
func (s *HTTPSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

// LastError returns the most recent write error, if any.
func (s *HTTPSink) LastError() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastErr
}

// httpStatusError is returned when the collector responds with a non-2xx status.
type httpStatusError struct {
	code int
}

func (e *httpStatusError) Error() string {
	return "analytics sink returned status " + http.StatusText(e.code)
}
