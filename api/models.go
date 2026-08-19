// Package api exposes Recall as an HTTP service using only the standard
// library. It provides a REST API over any store.Store, with optional
// RAG pipeline, knowledge graph, and reasoning support, plus API-key and
// JWT authentication.
//
// The server is built as an http.Handler (Server.Handler), so it can be
// embedded in any routing setup or served standalone via
// Server.ListenAndServe. The OpenAPI 3.0 specification is served at
// GET /openapi.json.
package api

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/deagy/recall/core"
)

// Error is the standard JSON error envelope returned by all endpoints.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e Error) Error() string { return e.Message }

// Error codes returned by the API.
const (
	ErrCodeBadRequest   = "bad_request"
	ErrCodeNotFound     = "not_found"
	ErrCodeUnauthorized = "unauthorized"
	ErrCodeForbidden    = "forbidden"
	ErrCodeMethod       = "method_not_allowed"
	ErrCodeInternal     = "internal_error"
)

func jsonMarshal(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	return enc.Encode(v)
}

// writeJSON writes v as a JSON response with the given HTTP status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = jsonMarshal(w, v)
}

// writeError writes a JSON error envelope with the given HTTP status.
func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, Error{Code: code, Message: message})
}

// uploadRequest is the body of POST /upload.
type uploadRequest struct {
	ID        string         `json:"id"`
	Title     string         `json:"title"`
	Author    string         `json:"author"`
	Source    string         `json:"source"`
	Namespace string         `json:"namespace"`
	Tags      []string       `json:"tags"`
	Metadata  map[string]any `json:"metadata"`
	Content   string         `json:"content"`
}

// uploadResponse is the response of POST /upload.
type uploadResponse struct {
	ID        string `json:"id"`
	Title     string `json:"title,omitempty"`
	Namespace string `json:"namespace,omitempty"`
	Chunks    int    `json:"chunks"`
	CreatedAt string `json:"created_at,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty"`
}

// searchRequest is the body of POST /hybrid-search.
type searchRequest struct {
	Query      string   `json:"query"`
	TopK       int      `json:"k"`
	MinScore   float64  `json:"min_score"`
	BM25Weight *float64 `json:"bm25_weight"`
	EfSearch   int      `json:"ef_search"`
}

// resultDTO is a single search result in API responses.
type resultDTO struct {
	ID         string         `json:"id"`
	Document   string         `json:"document"`
	ChunkIndex int            `json:"chunk_index"`
	Content    string         `json:"content"`
	Score      float64        `json:"score"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// searchResponse is the response of GET /search and POST /hybrid-search.
type searchResponse struct {
	Query   string      `json:"query"`
	Count   int         `json:"count"`
	Results []resultDTO `json:"results"`
}

// ragRequest is the body of POST /rag. The pipeline's retrieval options
// (topK, min score, max tokens) are fixed at server construction time;
// Hybrid selects QueryHybrid over Query.
type ragRequest struct {
	Query  string `json:"query"`
	Hybrid bool   `json:"hybrid"`
}

// citationDTO is a citation reference in RAG responses.
type citationDTO struct {
	Number   int     `json:"number"`
	ChunkID  string  `json:"chunk_id"`
	Document string  `json:"document,omitempty"`
	Score    float64 `json:"score"`
	Snippet  string  `json:"snippet,omitempty"`
}

// ragResponse is the response of POST /rag.
type ragResponse struct {
	Query     string        `json:"query"`
	Answer    string        `json:"answer"`
	Context   string        `json:"context"`
	Tokens    int           `json:"tokens"`
	Sources   []resultDTO   `json:"sources"`
	Citations []citationDTO `json:"citations,omitempty"`
}

// reasonRequest is the body of POST /graph/reason. Either Query (NL
// reasoning) or From+To (path exploration) must be provided.
type reasonRequest struct {
	Query   string `json:"query"`
	From    string `json:"from"`
	To      string `json:"to"`
	MaxHops int    `json:"max_hops"`
}

// inferredRelationDTO is an inferred relation in POST /graph/reason responses.
type inferredRelationDTO struct {
	From       string  `json:"from"`
	To         string  `json:"to"`
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence"`
	Rule       string  `json:"rule"`
	Hops       int     `json:"hops"`
}

// graphPathDTO is a discovered path in POST /graph/reason responses.
type graphPathDTO struct {
	Entities  []string `json:"entities"`
	Relations []string `json:"relations"`
}

// reasonResponse is the response of POST /graph/reason.
type reasonResponse struct {
	Inferences []inferredRelationDTO `json:"inferences"`
	Paths      []graphPathDTO        `json:"paths"`
}

// entityDTO is a graph entity in API responses.
type entityDTO struct {
	ID           string            `json:"id"`
	Label        string            `json:"label"`
	Type         string            `json:"type"`
	Properties   map[string]string `json:"properties,omitempty"`
	SourceChunks []string          `json:"source_chunks,omitempty"`
}

// relationDTO is a graph relation in API responses.
type relationDTO struct {
	From   string  `json:"from"`
	To     string  `json:"to"`
	Type   string  `json:"type"`
	Weight float64 `json:"weight"`
}

// entityResponse is the response of GET /graph/{entity}.
type entityResponse struct {
	Entity    entityDTO     `json:"entity"`
	Neighbors []entityDTO   `json:"neighbors"`
	Relations []relationDTO `json:"relations"`
}

// chunkMetadata converts core metadata (typed Values) into a JSON-friendly
// map.
func chunkMetadata(c *core.Chunk) map[string]any {
	if c.Metadata == nil {
		return nil
	}
	out := make(map[string]any, len(c.Metadata))
	for k, v := range c.Metadata {
		if v == nil {
			continue
		}
		switch val := v.(type) {
		case core.String:
			out[k] = val.Value
		case core.Number:
			out[k] = val.Value
		case core.Boolean:
			out[k] = val.Value
		case core.URI:
			out[k] = val.Value
		case core.Literal:
			out[k] = val.Value
		default:
			out[k] = v.String()
		}
	}
	return out
}
