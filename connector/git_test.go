package connector

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// haveGit reports whether the git CLI is available for tests.
func haveGit(t *testing.T) bool {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git CLI not available")
	}
	return true
}

// initRepo creates a small committed repository at dir and returns its path.
func initRepo(t *testing.T, dir string) string {
	t.Helper()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	git("init", "-q")
	git("config", "user.email", "test@example.com")
	git("config", "user.name", "Test")
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# Test Repo\nSome docs."), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("plain notes"), 0o600); err != nil {
		t.Fatal(err)
	}
	git("add", "-A")
	git("commit", "-q", "-m", "init")
	return dir
}

func TestGitConnector_CloneLocal(t *testing.T) {
	if !haveGit(t) {
		return
	}
	src := initRepo(t, filepath.Join(t.TempDir(), "src"))

	docs, err := (&GitConnector{Depth: 1}).Fetch(context.Background(), src)
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	var found []string
	for _, d := range docs {
		found = append(found, d.Title)
	}
	joined := strings.Join(found, ",")
	// MarkdownLoader titles the README doc from its H1 ("# Test Repo").
	if !strings.Contains(joined, "Test Repo") || !strings.Contains(joined, "notes") {
		t.Errorf("expected README and notes docs, got %v", found)
	}
}

func TestGitConnector_BadRef(t *testing.T) {
	if !haveGit(t) {
		return
	}
	if _, err := (&GitConnector{}).Fetch(context.Background(), filepath.Join(t.TempDir(), "does-not-exist.git")); err == nil {
		t.Error("expected clone error for missing repo")
	}
}
