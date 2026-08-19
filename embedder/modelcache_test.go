package embedder

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/deagy/recall/embedder/onnx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testONNXBytes builds a small valid ONNX model (y = x + 10) to serve from a
// fake HTTP server.
func testONNXBytes(t *testing.T) []byte {
	t.Helper()
	c, err := onnx.NewTensor([]int64{1}, onnx.Float32, []float32{10})
	require.NoError(t, err)
	nodes := []onnx.NodeSpec{{Op: "Add", Inputs: []string{"x", "c"}, Outputs: []string{"y"}}}
	inits := map[string]*onnx.Tensor{"c": c}
	inputs := []onnx.NamedType{{Name: "x", Dtype: onnx.Float32}}
	outputs := []onnx.NamedType{{Name: "y", Dtype: onnx.Float32}}
	return onnx.Encode(nodes, inits, inputs, outputs)
}

func TestModelCache_PathDeterministic(t *testing.T) {
	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	p1 := c.Path("https://example.com/a.onnx")
	p2 := c.Path("https://example.com/a.onnx")
	p3 := c.Path("https://example.com/b.onnx")
	assert.Equal(t, p1, p2)
	assert.NotEqual(t, p1, p3)
	assert.True(t, filepath.IsAbs(p1))
	assert.Equal(t, ".onnx", filepath.Ext(p1))
}

func TestModelCache_PathEmptyDir(t *testing.T) {
	_, err := NewModelCache("", 0)
	assert.Error(t, err)
}

func TestModelCache_Get_DownloadsAndLoads(t *testing.T) {
	data := testONNXBytes(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(data)
	}))
	defer srv.Close()

	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	path, err := c.Get(context.Background(), srv.URL+"/model.onnx")
	require.NoError(t, err)

	st, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, int64(len(data)), st.Size())

	m, err := onnx.LoadFile(path)
	require.NoError(t, err)
	assert.NotNil(t, m)
}

func TestModelCache_Get_CacheHit(t *testing.T) {
	data := testONNXBytes(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write(data)
	}))
	defer srv.Close()

	c, err := NewModelCache(t.TempDir(), 0) // ttl 0 = cache forever
	require.NoError(t, err)
	url := srv.URL + "/model.onnx"
	_, err = c.Get(context.Background(), url)
	require.NoError(t, err)
	_, err = c.Get(context.Background(), url)
	require.NoError(t, err)
	assert.Equal(t, int32(1), hits.Load(), "second Get must be served from cache")
}

func TestModelCache_Get_TTLExpiry(t *testing.T) {
	data := testONNXBytes(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write(data)
	}))
	defer srv.Close()

	c, err := NewModelCache(t.TempDir(), time.Nanosecond) // expires immediately
	require.NoError(t, err)
	url := srv.URL + "/model.onnx"
	_, err = c.Get(context.Background(), url)
	require.NoError(t, err)
	time.Sleep(2 * time.Millisecond)
	_, err = c.Get(context.Background(), url)
	require.NoError(t, err)
	assert.Equal(t, int32(2), hits.Load(), "expired entry must be re-downloaded")
}

func TestModelCache_Get_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	_, err = c.Get(context.Background(), srv.URL+"/model.onnx")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTP 500")
}

func TestModelCache_Get_InvalidURL(t *testing.T) {
	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	_, err = c.Get(context.Background(), "")
	assert.Error(t, err)
}

func TestModelCache_Get_ConcurrentDedup(t *testing.T) {
	data := testONNXBytes(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		time.Sleep(150 * time.Millisecond) // let all goroutines pile into inflight
		w.Write(data)
	}))
	defer srv.Close()

	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	url := srv.URL + "/model.onnx"

	const n = 8
	paths := make([]string, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			paths[i], errs[i] = c.Get(context.Background(), url)
		}(i)
	}
	wg.Wait()
	for i := 0; i < n; i++ {
		require.NoError(t, errs[i])
		assert.Equal(t, paths[0], paths[i])
	}
	assert.Equal(t, int32(1), hits.Load(), "concurrent Gets must deduplicate to one download")
}

func TestHuggingFaceURL(t *testing.T) {
	assert.Equal(t,
		DefaultHFBaseURL+"/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
		HuggingFaceURL("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx"),
	)
	// empty file defaults
	assert.Equal(t,
		DefaultHFBaseURL+"/some/repo/resolve/main/"+DefaultHFFile,
		HuggingFaceURL("some/repo", ""),
	)
	// custom base via hfURL
	assert.Equal(t,
		"https://mirror.example/some/repo/resolve/main/model.onnx",
		hfURL("https://mirror.example", "some/repo", "model.onnx"),
	)
}

func TestLoadHFModel_WithCache(t *testing.T) {
	data := testONNXBytes(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(data)
	}))
	defer srv.Close()

	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	c.SetBaseURL(srv.URL)

	m, err := LoadHFModel(context.Background(), c, "test/repo", "model.onnx")
	require.NoError(t, err)
	require.NotNil(t, m)

	// The returned model must actually run.
	x, err := onnx.NewTensor([]int64{1}, onnx.Float32, []float32{3})
	require.NoError(t, err)
	outs, err := m.Run(context.Background(), map[string]*onnx.Tensor{"x": x})
	require.NoError(t, err)
	f, err := outs["y"].AsFloat64()
	require.NoError(t, err)
	assert.InDelta(t, 13, f[0], 1e-6)
}

func TestLoadHFModel_CacheReused(t *testing.T) {
	data := testONNXBytes(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write(data)
	}))
	defer srv.Close()

	c, err := NewModelCache(t.TempDir(), 0)
	require.NoError(t, err)
	c.SetBaseURL(srv.URL)
	_, err = LoadHFModel(context.Background(), c, "test/repo", "model.onnx")
	require.NoError(t, err)
	_, err = LoadHFModel(context.Background(), c, "test/repo", "model.onnx")
	require.NoError(t, err)
	assert.Equal(t, int32(1), hits.Load())
}
