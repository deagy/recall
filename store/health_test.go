package store

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sampleContent returns text long enough for the default chunker to produce a
// chunk (MinChunkSize is 50 characters).
func sampleContent() string {
	return "Recall is a retrieval-augmented generation SDK. " +
		"It provides structured storage, embedding-based similarity search, " +
		"metadata filtering, and persistent storage."
}

func TestHealthCheck_MemoryStore(t *testing.T) {
	s, err := NewMemoryStore(Config{})
	require.NoError(t, err)
	ctx := context.Background()
	require.NoError(t, s.Upload(ctx, core.NewDocument("d1", "T", "s.md"), sampleContent()))

	rep, err := HealthCheck(ctx, s)
	require.NoError(t, err)
	assert.True(t, rep.OK)
	assert.Equal(t, StatusHealthy, rep.Status)
	assert.Equal(t, "memory", rep.Backend)
	assert.True(t, rep.Connected)
	assert.GreaterOrEqual(t, rep.Count, 1)
}

func TestHealthCheck_SQLiteStore(t *testing.T) {
	s, err := NewSQLiteStore(Config{}, ":memory:")
	require.NoError(t, err)
	defer s.Close()
	ctx := context.Background()
	require.NoError(t, s.Upload(ctx, core.NewDocument("d1", "T", "s.md"), sampleContent()))

	rep, err := HealthCheck(ctx, s)
	require.NoError(t, err)
	assert.True(t, rep.OK)
	assert.Equal(t, StatusHealthy, rep.Status)
	assert.Equal(t, "sqlite", rep.Backend)
	assert.True(t, rep.Connected)
	assert.GreaterOrEqual(t, rep.Count, 1)
	require.NotNil(t, rep.Integrity)
	assert.True(t, rep.Integrity.OK)
}

func TestHealthCheck_SQLiteDown(t *testing.T) {
	s, err := NewSQLiteStore(Config{}, ":memory:")
	require.NoError(t, err)
	// Closing the db makes the connectivity probe fail.
	require.NoError(t, s.Close())

	rep, err := HealthCheck(context.Background(), s)
	require.NoError(t, err)
	assert.False(t, rep.OK)
	assert.Equal(t, StatusDown, rep.Status)
	assert.False(t, rep.Connected)
	assert.Equal(t, "sqlite", rep.Backend)
	assert.NotEmpty(t, rep.Issues)
}

func TestHealthHandler(t *testing.T) {
	s, err := NewMemoryStore(Config{})
	require.NoError(t, err)
	h := HealthHandler(s)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/healthz", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var rep HealthReport
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &rep))
	assert.True(t, rep.OK)
	assert.Equal(t, "memory", rep.Backend)

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/diagnostics", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var diag Diagnostics
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &diag))
	assert.Equal(t, "memory", diag.Health.Backend)
	assert.False(t, diag.GeneratedAt.IsZero())

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/nope", nil))
	assert.Equal(t, http.StatusNotFound, rec.Code)
}

func TestHealthHandler_DownReturns503(t *testing.T) {
	s, err := NewSQLiteStore(Config{}, ":memory:")
	require.NoError(t, err)
	require.NoError(t, s.Close())
	h := HealthHandler(s)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/healthz", nil))
	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
}
