package analytics

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFileSink_WritesNDJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.ndjson")
	s, err := NewFileSink(path)
	require.NoError(t, err)
	require.NoError(t, s.Write(QueryRecord{Query: "hello", Results: 2, TopScore: 0.9}))
	require.NoError(t, s.Write(QueryRecord{Query: "world", Results: 1, TopScore: 0.5}))
	require.NoError(t, s.Close())

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	require.Len(t, lines, 2)

	var r0 QueryRecord
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &r0))
	assert.Equal(t, "hello", r0.Query)
	assert.Equal(t, 2, r0.Results)

	// Writing after Close fails.
	require.Error(t, s.Write(QueryRecord{Query: "x"}))
}

func TestHTTPSink_PostsToURL(t *testing.T) {
	var mu sync.Mutex
	var received []QueryRecord
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var rec QueryRecord
		if err := json.NewDecoder(r.Body).Decode(&rec); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		mu.Lock()
		received = append(received, rec)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	s := NewHTTPSink(srv.URL, nil)
	require.NoError(t, s.Write(QueryRecord{Query: "ping", Results: 1}))
	require.NoError(t, s.Close())

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, received, 1)
	assert.Equal(t, "ping", received[0].Query)
}

func TestHTTPSink_ErrorOnBadStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	s := NewHTTPSink(srv.URL, nil)
	require.Error(t, s.Write(QueryRecord{Query: "x"}))
	require.Error(t, s.LastError())
}
