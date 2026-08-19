// Package client is an HTTP client for the Recall REST API (see package
// api). It wraps every endpoint with typed request/response structs and a
// structured error type for non-2xx responses. It is the transport behind
// the recall CLI's server mode.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Config configures a Client.
type Config struct {
	// BaseURL is the server base URL (e.g. "http://localhost:8080").
	// Required.
	BaseURL string

	// APIKey, when set, authenticates every request. It is sent as both
	// "Authorization: Bearer <key>" and "X-API-Key: <key>" so servers with
	// either convention accept it.
	APIKey string

	// Timeout bounds each HTTP request. Defaults to 30s.
	Timeout time.Duration
}

// Client talks to a recall-server. It is safe for concurrent use.
type Client struct {
	base string
	key  string
	http *http.Client
}

// New creates a Client, validating and normalizing the configuration.
func New(cfg Config) (*Client, error) {
	u, err := url.Parse(cfg.BaseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("client: invalid base URL %q (want e.g. http://localhost:8080)", cfg.BaseURL)
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	return &Client{
		base: strings.TrimSuffix(u.String(), "/"),
		key:  cfg.APIKey,
		http: &http.Client{Timeout: cfg.Timeout},
	}, nil
}

// BaseURL returns the normalized base URL the client sends requests to.
func (c *Client) BaseURL() string { return c.base }

// Error is a non-2xx response from the server carrying the standard JSON
// error envelope ({"code","message"}).
type Error struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int
	// Code is the API error code (e.g. "not_found"). Empty when the
	// response body was not a valid error envelope.
	Code string
	// Message is the human-readable error message.
	Message string
}

func (e *Error) Error() string {
	if e.Code == "" {
		return fmt.Sprintf("recall server: HTTP %d: %s", e.StatusCode, e.Message)
	}
	return fmt.Sprintf("recall server: HTTP %d (%s): %s", e.StatusCode, e.Code, e.Message)
}

// do performs an HTTP request, JSON-encoding body (when non-nil) and
// decoding the response into out (when non-nil). Non-2xx responses are
// returned as *Error with the envelope decoded.
func (c *Client) do(ctx context.Context, method, path string, body, out any) error {
	var rd io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("client: encoding request: %w", err)
		}
		rd = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, rd)
	if err != nil {
		return fmt.Errorf("client: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
		req.Header.Set("X-API-Key", c.key)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("client: %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("client: reading %s %s response: %w", method, path, err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var env struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		}
		msg := strings.TrimSpace(string(data))
		if json.Unmarshal(data, &env) == nil && env.Message != "" {
			msg = env.Message
		}
		if env.Code == "" && msg == "" {
			msg = http.StatusText(resp.StatusCode)
		}
		return &Error{StatusCode: resp.StatusCode, Code: env.Code, Message: msg}
	}

	if out != nil {
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("client: decoding %s %s response: %w", method, path, err)
		}
	}
	return nil
}

// UploadRequest is the body of POST /upload.
type UploadRequest struct {
	// ID is the document ID. The server generates one when empty.
	ID string `json:"id,omitempty"`
	// Title is the document title.
	Title string `json:"title,omitempty"`
	// Author is the document author.
	Author string `json:"author,omitempty"`
	// Source is the origin (file path, URL, ...).
	Source string `json:"source,omitempty"`
	// Namespace optionally overrides the store default.
	Namespace string `json:"namespace,omitempty"`
	// Tags are arbitrary labels.
	Tags []string `json:"tags,omitempty"`
	// Metadata carries arbitrary structured attributes.
	Metadata map[string]any `json:"metadata,omitempty"`
	// Content is the document text (required).
	Content string `json:"content"`
}

// UploadResult is the response of POST /upload.
type UploadResult struct {
	ID        string `json:"id"`
	Title     string `json:"title,omitempty"`
	Namespace string `json:"namespace,omitempty"`
	Chunks    int    `json:"chunks"`
	CreatedAt string `json:"created_at,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty"`
}

// Upload sends a document to POST /upload.
func (c *Client) Upload(ctx context.Context, req UploadRequest) (*UploadResult, error) {
	var out UploadResult
	if err := c.do(ctx, http.MethodPost, "/upload", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// SearchOptions configures a search request.
type SearchOptions struct {
	// TopK is the maximum number of results. Defaults to 10 when <= 0.
	TopK int
	// MinScore is the minimum relevance score.
	MinScore float64
	// BM25Weight is the keyword weight for hybrid search (0-1). Zero means
	// the server default (0.5).
	BM25Weight float64
	// EfSearch controls HNSW search width (0 = server default).
	EfSearch int
}

// Result is a single search result.
type Result struct {
	ID         string         `json:"id"`
	Document   string         `json:"document"`
	ChunkIndex int            `json:"chunk_index"`
	Content    string         `json:"content"`
	Score      float64        `json:"score"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// SearchResults is the response of GET /search and POST /hybrid-search.
type SearchResults struct {
	Query   string   `json:"query"`
	Count   int      `json:"count"`
	Results []Result `json:"results"`
}

// Search performs a vector-similarity search via GET /search.
func (c *Client) Search(ctx context.Context, query string, opts SearchOptions) (*SearchResults, error) {
	q := url.Values{}
	q.Set("q", query)
	q.Set("k", strconv.Itoa(orDefault(opts.TopK, 10)))
	q.Set("min_score", strconv.FormatFloat(opts.MinScore, 'g', -1, 64))
	if opts.EfSearch > 0 {
		q.Set("ef_search", strconv.Itoa(opts.EfSearch))
	}
	var out SearchResults
	if err := c.do(ctx, http.MethodGet, "/search?"+q.Encode(), nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// HybridSearch performs a vector + BM25 search via POST /hybrid-search.
func (c *Client) HybridSearch(ctx context.Context, query string, opts SearchOptions) (*SearchResults, error) {
	req := struct {
		Query      string   `json:"query"`
		TopK       int      `json:"k"`
		MinScore   float64  `json:"min_score"`
		BM25Weight *float64 `json:"bm25_weight,omitempty"`
		EfSearch   int      `json:"ef_search"`
	}{
		Query:    query,
		TopK:     orDefault(opts.TopK, 10),
		MinScore: opts.MinScore,
		EfSearch: opts.EfSearch,
	}
	if opts.BM25Weight > 0 {
		w := opts.BM25Weight
		req.BM25Weight = &w
	}
	var out SearchResults
	if err := c.do(ctx, http.MethodPost, "/hybrid-search", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func orDefault(v, def int) int {
	if v <= 0 {
		return def
	}
	return v
}

// Citation is a ranked reference to a source chunk in RAG responses.
type Citation struct {
	Number   int     `json:"number"`
	ChunkID  string  `json:"chunk_id"`
	Document string  `json:"document,omitempty"`
	Score    float64 `json:"score"`
	Snippet  string  `json:"snippet,omitempty"`
}

// RAGResponse is the response of POST /rag.
type RAGResponse struct {
	Query     string     `json:"query"`
	Answer    string     `json:"answer"`
	Context   string     `json:"context"`
	Tokens    int        `json:"tokens"`
	Sources   []Result   `json:"sources"`
	Citations []Citation `json:"citations,omitempty"`
}

// RAG runs a RAG query via POST /rag. When hybrid is true the server uses
// hybrid retrieval.
func (c *Client) RAG(ctx context.Context, query string, hybrid bool) (*RAGResponse, error) {
	req := struct {
		Query  string `json:"query"`
		Hybrid bool   `json:"hybrid"`
	}{Query: query, Hybrid: hybrid}
	var out RAGResponse
	if err := c.do(ctx, http.MethodPost, "/rag", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Entity is a graph entity in API responses.
type Entity struct {
	ID           string            `json:"id"`
	Label        string            `json:"label"`
	Type         string            `json:"type"`
	Properties   map[string]string `json:"properties,omitempty"`
	SourceChunks []string          `json:"source_chunks,omitempty"`
}

// Relation is a graph relation in API responses.
type Relation struct {
	From   string  `json:"from"`
	To     string  `json:"to"`
	Type   string  `json:"type"`
	Weight float64 `json:"weight"`
}

// EntityDetail is the response of GET /graph/{entity}.
type EntityDetail struct {
	Entity    Entity     `json:"entity"`
	Neighbors []Entity   `json:"neighbors"`
	Relations []Relation `json:"relations"`
}

// GraphEntity fetches an entity (by ID, or unique label) with its neighbors
// and relations via GET /graph/{entity}.
func (c *Client) GraphEntity(ctx context.Context, id string) (*EntityDetail, error) {
	var out EntityDetail
	if err := c.do(ctx, http.MethodGet, "/graph/"+url.PathEscape(id), nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ReasonRequest is the body of POST /graph/reason. Either Query (natural
// language reasoning) or both From and To (path exploration) is required.
type ReasonRequest struct {
	Query   string `json:"query,omitempty"`
	From    string `json:"from,omitempty"`
	To      string `json:"to,omitempty"`
	MaxHops int    `json:"max_hops,omitempty"`
}

// InferredRelation is an inferred relation in POST /graph/reason responses.
type InferredRelation struct {
	From       string  `json:"from"`
	To         string  `json:"to"`
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence"`
	Rule       string  `json:"rule"`
	Hops       int     `json:"hops"`
}

// Path is a discovered entity path in POST /graph/reason responses.
type Path struct {
	Entities  []string `json:"entities"`
	Relations []string `json:"relations"`
}

// ReasonResponse is the response of POST /graph/reason.
type ReasonResponse struct {
	Inferences []InferredRelation `json:"inferences"`
	Paths      []Path             `json:"paths"`
}

// Reason runs multi-hop reasoning via POST /graph/reason.
func (c *Client) Reason(ctx context.Context, req ReasonRequest) (*ReasonResponse, error) {
	var out ReasonResponse
	if err := c.do(ctx, http.MethodPost, "/graph/reason", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Integrity mirrors store.IntegrityReport (the store package serves these
// fields without JSON tags, so the keys are the Go field names).
type Integrity struct {
	OK          bool     `json:"OK"`
	Issues      []string `json:"Issues"`
	ForeignKeys []string `json:"ForeignKeys"`
}

// StoreHealth mirrors the store HealthReport served at /healthz and
// embedded in /diagnostics.
type StoreHealth struct {
	OK         bool       `json:"ok"`
	Status     string     `json:"status"`
	Backend    string     `json:"backend"`
	Connected  bool       `json:"connected"`
	Count      int        `json:"count"`
	Namespaces []string   `json:"namespaces,omitempty"`
	Integrity  *Integrity `json:"integrity,omitempty"`
	Issues     []string   `json:"issues,omitempty"`
	CheckedAt  time.Time  `json:"checked_at"`
}

// StoreDiagnostics mirrors the GET /diagnostics response of a
// recall-server.
type StoreDiagnostics struct {
	Health      StoreHealth `json:"health"`
	GeneratedAt time.Time   `json:"generated_at"`
}

// Health fetches the store health report via GET /healthz.
func (c *Client) Health(ctx context.Context) (*StoreHealth, error) {
	var out StoreHealth
	if err := c.do(ctx, http.MethodGet, "/healthz", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Diagnostics fetches the full diagnostics snapshot via GET /diagnostics.
func (c *Client) Diagnostics(ctx context.Context) (*StoreDiagnostics, error) {
	var out StoreDiagnostics
	if err := c.do(ctx, http.MethodGet, "/diagnostics", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ClusterNodeHealth mirrors the distributed package's ClusterHealth
// (served by distributed.HealthHandler; the fields are exported without
// JSON tags, so the keys are the Go field names).
type ClusterNodeHealth struct {
	Total    int    `json:"Total"`
	Online   int    `json:"Online"`
	Degraded int    `json:"Degraded"`
	Offline  int    `json:"Offline"`
	Overall  string `json:"Overall"`
}

// ClusterShardStats mirrors the distributed package's ShardStats.
type ClusterShardStats struct {
	Total    int            `json:"total"`
	Active   int            `json:"active"`
	Inactive int            `json:"inactive"`
	Degraded int            `json:"degraded"`
	Chunks   int            `json:"chunks"`
	PerNode  map[string]int `json:"per_node,omitempty"`
}

// ClusterDiagnostics mirrors the GET /diagnostics response of a
// distributed cluster node (distributed.HealthHandler).
type ClusterDiagnostics struct {
	Health      ClusterNodeHealth `json:"health"`
	Shards      ClusterShardStats `json:"shards"`
	GeneratedAt time.Time         `json:"generated_at"`
}

// ProbeClusterNode fetches the /diagnostics snapshot served by a
// distributed cluster node at baseURL. A non-2xx or unparseable response
// returns an error; callers report such nodes as unreachable.
func ProbeClusterNode(ctx context.Context, baseURL string, timeout time.Duration) (*ClusterDiagnostics, error) {
	u, err := url.Parse(baseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("client: invalid node URL %q (want e.g. http://node1:9000)", baseURL)
	}
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	c := &Client{
		base: strings.TrimSuffix(u.String(), "/"),
		http: &http.Client{Timeout: timeout},
	}
	var out ClusterDiagnostics
	if err := c.do(ctx, http.MethodGet, "/diagnostics", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
