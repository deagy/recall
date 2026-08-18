package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

// DefaultGitHubAPI is the base URL for the GitHub REST API.
const DefaultGitHubAPI = "https://api.github.com"

// GitHubConnector fetches a repository's README and (optionally) its issues
// as documents via the GitHub REST API.
type GitHubConnector struct {
	// BaseURL overrides the API base; default https://api.github.com.
	BaseURL string

	// Token is an optional GitHub personal access token.
	Token string

	// Client is the HTTP client; default http.DefaultClient.
	Client *http.Client

	// IncludeIssues, when true, also fetches issues (pull requests are
	// excluded).
	IncludeIssues bool

	// MaxIssues caps the number of issues fetched; 0 means 100.
	MaxIssues int

	// State filters issues: "open", "closed", or "all"; default "all".
	State string
}

// Name implements Connector.
func (g *GitHubConnector) Name() string { return "github" }

// Fetch retrieves documents for a repository ref of the form "owner/repo".
func (g *GitHubConnector) Fetch(ctx context.Context, ref string) ([]*loader.Document, error) {
	base := g.BaseURL
	if base == "" {
		base = DefaultGitHubAPI
	}
	base = strings.TrimSuffix(base, "/")
	if !strings.Contains(ref, "/") {
		return nil, fmt.Errorf("github: ref must be owner/repo, got %q", ref)
	}
	docs := make([]*loader.Document, 0, 2)
	readme, err := g.fetchReadme(ctx, base, ref)
	if err != nil && !isNotFound(err) {
		return nil, err
	}
	if readme != nil {
		docs = append(docs, readme)
	}
	if g.IncludeIssues {
		issues, err := g.fetchIssues(ctx, base, ref)
		if err != nil {
			return nil, err
		}
		docs = append(docs, issues...)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("github: no documents available for %s", ref)
	}
	return docs, nil
}

// fetchReadme downloads the raw README; a 404 yields (nil, nil).
func (g *GitHubConnector) fetchReadme(ctx context.Context, base, ref string) (*loader.Document, error) {
	body, err := g.get(ctx, base+"/repos/"+ref+"/readme", "application/vnd.github.raw+json")
	if err != nil {
		if isNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	d := loader.NewDocument("github.com/"+ref+"/README", "README", "github.com/"+ref, string(body))
	d.Metadata["source"] = core.String{Value: "github"}
	return d, nil
}

// githubIssue is the subset of the GitHub issue API used here.
type githubIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	Labels []struct {
		Name string `json:"name"`
	} `json:"labels"`
	URL         string    `json:"url"`
	PullRequest *struct{} `json:"pull_request"`
}

// fetchIssues lists issues (excluding pull requests) as documents.
func (g *GitHubConnector) fetchIssues(ctx context.Context, base, ref string) ([]*loader.Document, error) {
	state := g.State
	if state == "" {
		state = "all"
	}
	max := g.MaxIssues
	if max <= 0 || max > 100 {
		max = 100
	}
	body, err := g.get(ctx, fmt.Sprintf("%s/repos/%s/issues?state=%s&per_page=%d", base, ref, state, max), "")
	if err != nil {
		return nil, err
	}
	var issues []githubIssue
	if err := json.Unmarshal(body, &issues); err != nil {
		return nil, fmt.Errorf("github: decode issues: %w", err)
	}
	docs := make([]*loader.Document, 0, len(issues))
	for _, iss := range issues {
		if iss.PullRequest != nil {
			continue // GitHub lists PRs in the issues endpoint
		}
		content := iss.Title
		if iss.Body != "" {
			content += "\n\n" + iss.Body
		}
		labels := make([]string, 0, len(iss.Labels))
		for _, l := range iss.Labels {
			labels = append(labels, l.Name)
		}
		d := loader.NewDocument(fmt.Sprintf("github.com/%s/issues/%d", ref, iss.Number), iss.Title, iss.URL, content)
		d.Metadata["number"] = core.Number{Value: float64(iss.Number)}
		d.Metadata["state"] = core.String{Value: iss.State}
		d.Metadata["labels"] = core.String{Value: strings.Join(labels, ",")}
		docs = append(docs, d)
	}
	return docs, nil
}

// get performs an authenticated GET and returns the body.
func (g *GitHubConnector) get(ctx context.Context, urlStr, accept string) ([]byte, error) {
	client := g.Client
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return nil, err
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	if g.Token != "" {
		req.Header.Set("Authorization", "Bearer "+g.Token)
	}
	req.Header.Set("User-Agent", "recall-ingest/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, &apiError{status: resp.StatusCode}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("github: GET %s returned %s", urlStr, resp.Status)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return body, nil
}

// apiError wraps non-2xx API responses so callers can detect 404.
type apiError struct{ status int }

func (e *apiError) Error() string { return fmt.Sprintf("github: API error %d", e.status) }

func isNotFound(err error) bool {
	ae, ok := err.(*apiError)
	return ok && ae.status == http.StatusNotFound
}
