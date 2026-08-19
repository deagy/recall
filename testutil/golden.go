package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// UpdateGolden, when true, makes Golden and GoldenJSON write the file at the
// given path with the current value (and skip the assertion) instead of
// failing on a mismatch. Tests typically wire it to a custom flag, e.g.:
//
//	var update = flag.Bool("update", false, "update golden files")
//
//	func TestX(t *testing.T) {
//	    if *update {
//	        testutil.UpdateGolden = true
//	    }
//	    ...
//	}
var UpdateGolden = false

// Golden compares got against the contents of the golden file at path (relative
// to the test working directory, conventionally testdata/). If the file does
// not exist, the test fails with a hint to run with the update flag — unless
// UpdateGolden is true, in which case the file is written and the test is
// skipped. On a mismatch the test fails and reports both values.
func Golden(t testing.TB, path, got string) {
	t.Helper()
	if UpdateGolden {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("testutil.Golden: create dir: %v", err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatalf("testutil.Golden: write %s: %v", path, err)
		}
		t.Skipf("golden file %s updated", path)
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("testutil.Golden: read %s: %v (run with the update flag to create it)", path, err)
	}
	if string(want) != got {
		t.Errorf("testutil.Golden: %s mismatch (run with the update flag to refresh)\n--- want ---\n%s\n--- got ---\n%s",
			path, string(want), got)
	}
}

// GoldenJSON is Golden for structured values: it serializes v with
// json.MarshalIndent (two-space indent) and compares against the file.
func GoldenJSON(t testing.TB, path string, v any) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("testutil.GoldenJSON: marshal: %v", err)
	}
	Golden(t, path, string(data))
}

// GoldenDiff returns a compact line-based diff summary between want and got,
// for use in custom error messages. It returns "" when the values are equal.
func GoldenDiff(want, got string) string {
	if want == got {
		return ""
	}
	wl := strings.Split(want, "\n")
	gl := strings.Split(got, "\n")
	n := len(wl)
	if len(gl) > n {
		n = len(gl)
	}
	var b strings.Builder
	for i := 0; i < n; i++ {
		w, g := "", ""
		if i < len(wl) {
			w = wl[i]
		}
		if i < len(gl) {
			g = gl[i]
		}
		if w != g {
			b.WriteString("-\t" + w + "\n")
			b.WriteString("+\t" + g + "\n")
		}
	}
	return b.String()
}
