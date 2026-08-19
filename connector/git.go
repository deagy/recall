package connector

import (
	"context"
	"fmt"
	"os"
	"os/exec"

	"github.com/deagy/recall/loader"
)

// GitConnector clones a git repository to a temporary directory and loads
// its files with a DirectoryLoader. The git CLI is invoked via exec; the
// binary can be overridden for testing.
type GitConnector struct {
	// GitBinary is the git executable; default "git" (resolved via PATH).
	GitBinary string

	// Depth sets --depth N for shallow clones; 0 means full clone.
	Depth int

	// ExtraArgs are appended to the clone command (e.g. ["--branch", "v1"]).
	ExtraArgs []string

	// DirLoader loads the cloned tree; default is a recursive
	// DirectoryLoader for common text extensions.
	DirLoader *loader.DirectoryLoader
}

// defaultGitExtensions are loaded from cloned repositories.
var defaultGitExtensions = []string{".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}

// Name implements Connector.
func (g *GitConnector) Name() string { return "git" }

// Fetch clones the repository at ref (a git URL or local path) and returns
// the loaded documents. The temporary clone is removed on return.
func (g *GitConnector) Fetch(ctx context.Context, ref string) ([]*loader.Document, error) {
	bin := g.GitBinary
	if bin == "" {
		bin = "git"
	}
	tmp, err := os.MkdirTemp("", "recall-git-*")
	if err != nil {
		return nil, fmt.Errorf("git: %w", err)
	}
	defer os.RemoveAll(tmp)

	args := []string{"clone"}
	if g.Depth > 0 {
		args = append(args, "--depth", fmt.Sprintf("%d", g.Depth))
	}
	args = append(args, g.ExtraArgs...)
	args = append(args, "--", ref, tmp)
	//nolint:gosec // G204: exec.Command does not invoke a shell; bin is the git executable and ref is protected by the "--" separator from option injection
	cmd := exec.CommandContext(ctx, bin, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("git clone %s: %w: %s", ref, err, out)
	}
	dl := g.DirLoader
	if dl == nil {
		dl, err = loader.NewDirectoryLoader(defaultGitExtensions, true, nil)
		if err != nil {
			return nil, err
		}
	}
	docs, err := dl.Load(ctx, tmp)
	if err != nil {
		return docs, fmt.Errorf("git: load %s: %w", ref, err)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("git: no loadable files in %s", ref)
	}
	return docs, nil
}
