package distributed

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestShardDistribution(t *testing.T) {
	c := newClusterWithNodes(t, "n1", "n2")
	sm := NewShardManager(c)
	_, err := sm.CreateShardWithID("n1", "s1")
	require.NoError(t, err)
	_, err = sm.CreateShardWithID("n2", "s2")
	require.NoError(t, err)
	_, err = sm.CreateShardWithID("n2", "s3")
	require.NoError(t, err)

	stats := ShardDistribution(sm)
	assert.Equal(t, 3, stats.Total)
	assert.Equal(t, 3, stats.Active)
	assert.Equal(t, 0, stats.Inactive)
	assert.Equal(t, 0, stats.Degraded)
	assert.Equal(t, 1, stats.PerNode["n1"])
	assert.Equal(t, 2, stats.PerNode["n2"])
}

func TestShardDistribution_Nil(t *testing.T) {
	stats := ShardDistribution(nil)
	assert.Equal(t, 0, stats.Total)
	assert.Nil(t, stats.PerNode)
}

func TestDiagnostics(t *testing.T) {
	c := newClusterWithNodes(t, "n1", "n2")
	sm := NewShardManager(c)
	_, err := sm.CreateShardWithID("n1", "s1")
	require.NoError(t, err)

	d := Diagnostics(c, sm)
	assert.Equal(t, "healthy", d.Health.Overall)
	assert.Equal(t, 2, d.Health.Total)
	assert.Equal(t, 1, d.Shards.Total)
	assert.False(t, d.GeneratedAt.IsZero())
}

func TestClusterHealthHandler(t *testing.T) {
	c := newClusterWithNodes(t, "n1", "n2")
	sm := NewShardManager(c)
	_, err := sm.CreateShardWithID("n1", "s1")
	require.NoError(t, err)
	h := HealthHandler(c, sm)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/healthz", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var ch ClusterHealth
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &ch))
	assert.Equal(t, "healthy", ch.Overall)

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/diagnostics", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var d ClusterDiagnostics
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &d))
	assert.Equal(t, 1, d.Shards.Total)

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/nope", nil))
	assert.Equal(t, http.StatusNotFound, rec.Code)
}

func TestClusterHealthHandler_DownReturns503(t *testing.T) {
	c := newClusterWithNodes(t, "n1")
	require.NoError(t, c.SetNodeStatus("n1", "offline"))
	sm := NewShardManager(c)
	h := HealthHandler(c, sm)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/healthz", nil))
	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
}
