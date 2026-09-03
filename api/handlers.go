package api

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/store"
)

// openapiSpec is the embedded OpenAPI 3.0 document.
//
//go:embed openapi.json
var openapiSpec []byte

// handleHealth serves GET /healthz using the store health report.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	rep, err := store.HealthCheck(r.Context(), s.cfg.Store)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	code := http.StatusOK
	if !rep.OK {
		code = http.StatusServiceUnavailable
	}
	writeJSON(w, code, rep)
}

// handleReady serves GET /readyz. The server is ready when it can reach the
// store (health check returns without a hard error).
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if _, err := store.HealthCheck(r.Context(), s.cfg.Store); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ready": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ready": true})
}

// handleDiagnostics serves GET /diagnostics with a full diagnostics snapshot.
func (s *Server) handleDiagnostics(w http.ResponseWriter, r *http.Request) {
	d, err := store.DiagnosticsReport(r.Context(), s.cfg.Store)
	if err != nil {
		writeError(w, http.StatusInternalServerError, ErrCodeInternal, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, d)
}

// handleOpenAPI serves GET /openapi.json.
func (s *Server) handleOpenAPI(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(openapiSpec)
}

// buildHandler assembles the route table and middleware chain.
func (s *Server) buildHandler() http.Handler {
	mux := http.NewServeMux()

	// Operational endpoints (unauthenticated by default).
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.HandleFunc("GET /diagnostics", s.handleDiagnostics)
	mux.HandleFunc("GET /openapi.json", s.handleOpenAPI)

	// Data endpoints (protected when an Authenticator is configured).
	mux.HandleFunc("POST /upload", s.handleUpload)
	mux.HandleFunc("GET /search", s.handleSearch)
	mux.HandleFunc("POST /hybrid-search", s.handleHybridSearch)
	mux.HandleFunc("POST /rag", s.handleRAG)
	mux.HandleFunc("GET /graph/{entity}", s.handleGraphEntity)
	mux.HandleFunc("POST /graph/reason", s.handleGraphReason)
	mux.HandleFunc("GET /whoami", s.handleWhoami)

	var h http.Handler = mux

	// Authentication (applied only to non-exempt paths).
	if s.cfg.Authenticator != nil {
		h = s.requireAuth(h)
	}

	if s.cfg.AllowCORS {
		h = s.cors(h)
	}

	return h
}

// requireAuth applies authentication to every request except those whose
// path is exempt (see Config.ExemptPaths).
func (s *Server) requireAuth(next http.Handler) http.Handler {
	exempt := make(map[string]struct{}, len(s.cfg.ExemptPaths))
	for _, p := range s.cfg.ExemptPaths {
		exempt[p] = struct{}{}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isExempt(r.URL.Path, exempt) {
			next.ServeHTTP(w, r)
			return
		}
		RequireAuth(s.cfg.Authenticator, "Bearer", "ApiKey")(next).ServeHTTP(w, r)
	})
}

// isExempt reports whether path matches an exempt path exactly or as a
// sub-path of one.
func isExempt(path string, exempt map[string]struct{}) bool {
	for p := range exempt {
		if path == p || strings.HasPrefix(path, p+"/") {
			return true
		}
	}
	return false
}

// cors adds permissive CORS headers to every response.
func (s *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		h.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// readJSON decodes the request body as JSON, enforcing maxBytes.
func readJSON(w http.ResponseWriter, r *http.Request, maxBytes int64, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		if ne, ok := err.(*json.SyntaxError); ok {
			writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "invalid JSON: "+ne.Error())
		} else if strings.Contains(err.Error(), "http: request body too large") {
			writeError(w, http.StatusRequestEntityTooLarge, ErrCodeBadRequest, "request body too large")
		} else {
			writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "invalid request body: "+err.Error())
		}
		return false
	}
	return true
}

// parseIntQuery parses an optional integer query parameter.
func parseIntQuery(r *http.Request, name string, def int) int {
	v := r.URL.Query().Get(name)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

// parseFloatQuery parses an optional float query parameter.
func parseFloatQuery(r *http.Request, name string, def float64) float64 {
	v := r.URL.Query().Get(name)
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return f
}

// namespaceNamer is implemented by stores that expose their default
// namespace (all built-in stores do). It lets the API determine the
// namespace an upload without an explicit namespace will land in.
type namespaceNamer interface {
	Namespace() string
}

// containsStr reports whether s is present in list.
func containsStr(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// searchNamespaceFilter returns a metadata filter restricting search results
// to the request's allowed namespaces, or nil when the request is
// unrestricted.
func searchNamespaceFilter(r *http.Request) *index.TermInFilter {
	allowed := RequestNamespaces(r)
	if len(allowed) == 0 {
		return nil
	}
	return &index.TermInFilter{Key: core.MetadataKeyNamespace, Values: allowed}
}

// checkUploadNamespace enforces the request's namespace scope for an upload.
// Unrestricted requests always pass. For scoped requests the target namespace
// is the document's namespace, falling back to the store's default when the
// store exposes one; if it cannot be determined or is not allowed, a 403 is
// written and false is returned.
func (s *Server) checkUploadNamespace(w http.ResponseWriter, r *http.Request, doc *core.Document) bool {
	allowed := RequestNamespaces(r)
	if len(allowed) == 0 {
		return true
	}
	ns := doc.Namespace
	if ns == "" {
		if namer, ok := s.cfg.Store.(namespaceNamer); ok {
			ns = namer.Namespace()
		}
	}
	if ns == "" {
		writeError(w, http.StatusForbidden, ErrCodeForbidden, "uploads with a scoped API key must specify a namespace")
		return false
	}
	if !containsStr(allowed, ns) {
		writeError(w, http.StatusForbidden, ErrCodeForbidden, "namespace not allowed for this API key: "+ns)
		return false
	}
	return true
}

// entityAllowed reports whether a graph entity is visible to the request.
// Unrestricted requests see everything; scoped requests only see entities
// with at least one source chunk stamped with an allowed namespace (entities
// with no verifiable namespace are hidden — fail closed).
func (s *Server) entityAllowed(r *http.Request, e *graph.Entity) bool {
	allowed := RequestNamespaces(r)
	if len(allowed) == 0 {
		return true
	}
	if e == nil {
		return false
	}
	for _, id := range e.SourceChunks {
		chunk, ok := s.cfg.Store.GetChunk(id)
		if !ok || chunk == nil {
			continue
		}
		if ns := chunk.GetMetadataString(core.MetadataKeyNamespace); ns != "" && containsStr(allowed, ns) {
			return true
		}
	}
	return false
}

// handleUpload serves POST /upload: it builds a core.Document from the JSON
// body and ingests it via the store.
func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	var req uploadRequest
	if !readJSON(w, r, s.cfg.MaxUploadBytes, &req) {
		return
	}
	if strings.TrimSpace(req.Content) == "" {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "content is required")
		return
	}

	doc := core.NewDocument(strings.TrimSpace(req.ID), req.Title, req.Source)
	if doc.ID == "" {
		doc.ID = fmt.Sprintf("doc-%d", time.Now().UnixNano())
	}
	doc.Author = req.Author
	doc.Namespace = req.Namespace
	doc.Tags = req.Tags
	doc.Metadata = metadataFromJSON(req.Metadata)

	if !s.checkUploadNamespace(w, r, doc) {
		return
	}

	if err := s.cfg.Store.Upload(r.Context(), doc, req.Content); err != nil {
		writeError(w, http.StatusInternalServerError, ErrCodeInternal, "upload failed: "+err.Error())
		return
	}

	resp := uploadResponse{
		ID:        doc.ID,
		Title:     doc.Title,
		Namespace: doc.Namespace,
		Chunks:    doc.ChunkCount,
	}
	if !doc.CreatedAt.IsZero() {
		resp.CreatedAt = doc.CreatedAt.UTC().Format(time.RFC3339)
	}
	if !doc.UpdatedAt.IsZero() {
		resp.UpdatedAt = doc.UpdatedAt.UTC().Format(time.RFC3339)
	}
	writeJSON(w, http.StatusOK, resp)
}

// metadataFromJSON converts arbitrary JSON metadata values into typed
// core.Value instances.
func metadataFromJSON(m map[string]any) map[string]core.Value {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]core.Value, len(m))
	for k, v := range m {
		out[k] = valueFromAny(v)
	}
	return out
}

// valueFromAny maps a decoded JSON value to a core.Value.
func valueFromAny(v any) core.Value {
	switch val := v.(type) {
	case string:
		return core.String{Value: val}
	case float64:
		return core.Number{Value: val}
	case bool:
		return core.Boolean{Value: val}
	case nil:
		return core.Literal{Value: ""}
	default:
		b, _ := json.Marshal(v)
		return core.Literal{Value: string(b)}
	}
}

// handleSearch serves GET /search (vector similarity only).
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "query parameter q is required")
		return
	}
	k := parseIntQuery(r, "k", 10)
	if k <= 0 {
		k = 10
	}
	opts := index.DefaultSearchOptions(k)
	opts.MinScore = parseFloatQuery(r, "min_score", 0)
	opts.EfSearch = parseIntQuery(r, "ef_search", 0)
	if f := searchNamespaceFilter(r); f != nil {
		opts.Filters = append(opts.Filters, f)
	}

	results, err := s.cfg.Store.Search(r.Context(), q, opts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, ErrCodeInternal, "search failed: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, searchResponse{Query: q, Count: len(results), Results: resultsToDTO(results)})
}

// handleHybridSearch serves POST /hybrid-search (vector + BM25 fusion).
func (s *Server) handleHybridSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if !readJSON(w, r, s.cfg.MaxUploadBytes, &req) {
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "query is required")
		return
	}
	k := req.TopK
	if k <= 0 {
		k = 10
	}
	opts := index.DefaultSearchOptions(k)
	opts.MinScore = req.MinScore
	opts.Hybrid = true
	if req.BM25Weight != nil {
		opts.BM25Weight = *req.BM25Weight
	} else {
		opts.BM25Weight = 0.5
	}
	opts.EfSearch = req.EfSearch
	if f := searchNamespaceFilter(r); f != nil {
		opts.Filters = append(opts.Filters, f)
	}

	results, err := s.cfg.Store.SearchHybrid(r.Context(), req.Query, opts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, ErrCodeInternal, "hybrid search failed: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, searchResponse{Query: req.Query, Count: len(results), Results: resultsToDTO(results)})
}

// resultsToDTO converts search results into their API representation.
func resultsToDTO(results []index.SearchResult) []resultDTO {
	out := make([]resultDTO, 0, len(results))
	for _, r := range results {
		if r.Chunk == nil {
			continue
		}
		out = append(out, resultDTO{
			ID:         r.Chunk.ID,
			Document:   r.Chunk.DocumentRef,
			ChunkIndex: r.Chunk.ChunkIndex,
			Content:    r.Chunk.Content,
			Score:      r.Score,
			Metadata:   chunkMetadata(r.Chunk),
		})
	}
	return out
}

// handleRAG serves POST /rag, running the RAG pipeline if configured.
func (s *Server) handleRAG(w http.ResponseWriter, r *http.Request) {
	if s.cfg.Pipeline == nil {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "rag pipeline is not configured on this server")
		return
	}
	var req ragRequest
	if !readJSON(w, r, s.cfg.MaxUploadBytes, &req) {
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "query is required")
		return
	}

	var resp *pipeline.RAGResponse
	var err error
	p := s.cfg.Pipeline
	if f := searchNamespaceFilter(r); f != nil {
		p = p.Clone().WithSearchFilters(f)
	}
	if req.Hybrid {
		resp, err = p.QueryHybrid(r.Context(), req.Query)
	} else {
		resp, err = p.Query(r.Context(), req.Query)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, ErrCodeInternal, "rag query failed: "+err.Error())
		return
	}

	out := ragResponse{
		Query:   req.Query,
		Answer:  resp.Answer,
		Context: resp.Context,
		Tokens:  resp.Tokens,
		Sources: resultsToDTO(resp.Sources),
	}
	for _, c := range resp.Citations {
		out.Citations = append(out.Citations, citationDTO{
			Number:   c.Number,
			ChunkID:  c.ChunkID,
			Document: c.DocumentRef,
			Score:    c.Score,
			Snippet:  c.Snippet,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// handleGraphEntity serves GET /graph/{entity}.
func (s *Server) handleGraphEntity(w http.ResponseWriter, r *http.Request) {
	if s.cfg.Graph == nil {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "graph store is not configured on this server")
		return
	}
	id := r.PathValue("entity")
	if id == "" {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "entity id is required")
		return
	}
	// Try direct ID lookup, then fall back to a label match.
	entity, ok := s.cfg.Graph.GetEntity(id)
	if !ok {
		if matches := s.cfg.Graph.FindEntitiesByLabel(id); len(matches) == 1 {
			entity = matches[0]
		} else {
			writeError(w, http.StatusNotFound, ErrCodeNotFound, "entity not found: "+id)
			return
		}
	}

	// allowedEntity caches per-request visibility checks for scoped requests
	// (unscoped requests see everything).
	allowed := make(map[string]bool)
	allowedEntity := func(e *graph.Entity) bool {
		if e == nil {
			return false
		}
		if v, cached := allowed[e.ID]; cached {
			return v
		}
		v := s.entityAllowed(r, e)
		allowed[e.ID] = v
		return v
	}

	// Out-of-scope entities are reported as not found, so the endpoint does
	// not confirm their existence to scoped credentials.
	if !allowedEntity(entity) {
		writeError(w, http.StatusNotFound, ErrCodeNotFound, "entity not found: "+id)
		return
	}

	var relations []relationDTO
	for _, rel := range s.cfg.Graph.Relations() {
		if rel.From == entity.ID || rel.To == entity.ID {
			from, _ := s.cfg.Graph.GetEntity(rel.From)
			to, _ := s.cfg.Graph.GetEntity(rel.To)
			if !allowedEntity(from) || !allowedEntity(to) {
				continue
			}
			relations = append(relations, relationDTO{From: rel.From, To: rel.To, Type: rel.Type, Weight: rel.Weight})
		}
	}

	neighbors := make([]entityDTO, 0)
	for _, n := range s.cfg.Graph.Neighbors(entity.ID) {
		if n == nil || !allowedEntity(n) {
			continue
		}
		neighbors = append(neighbors, toEntityDTO(n))
	}

	writeJSON(w, http.StatusOK, entityResponse{
		Entity:    toEntityDTO(entity),
		Neighbors: neighbors,
		Relations: relations,
	})
}

// handleGraphReason serves POST /graph/reason, running NL reasoning or path
// exploration on the reasoning engine if configured.
func (s *Server) handleGraphReason(w http.ResponseWriter, r *http.Request) {
	if s.cfg.Reasoner == nil {
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "reasoning engine is not configured on this server")
		return
	}
	var req reasonRequest
	if !readJSON(w, r, s.cfg.MaxUploadBytes, &req) {
		return
	}

	out := reasonResponse{Inferences: []inferredRelationDTO{}, Paths: []graphPathDTO{}}
	hops := req.MaxHops
	if hops <= 0 {
		hops = 3
	}

	// allowedEntity caches per-request visibility checks for scoped requests
	// (unscoped requests see everything).
	allowed := make(map[string]bool)
	allowedEntity := func(e *graph.Entity) bool {
		if e == nil {
			return false
		}
		if v, cached := allowed[e.ID]; cached {
			return v
		}
		v := s.entityAllowed(r, e)
		allowed[e.ID] = v
		return v
	}
	// resolver resolves graph entities for scope checks: the configured graph
	// store, falling back to the reasoning engine's own graph. When neither
	// is available, scoped requests fail closed (everything denied).
	var resolver func(id string) (*graph.Entity, bool)
	switch {
	case s.cfg.Graph != nil:
		resolver = s.cfg.Graph.GetEntity
	case s.cfg.Reasoner != nil && s.cfg.Reasoner.Graph() != nil:
		resolver = s.cfg.Reasoner.Graph().GetEntity
	}
	allowedRelation := func(rel *graph.Relation) bool {
		if rel == nil || resolver == nil {
			return false
		}
		from, _ := resolver(rel.From)
		to, _ := resolver(rel.To)
		return allowedEntity(from) && allowedEntity(to)
	}

	switch {
	case strings.TrimSpace(req.Query) != "":
		for _, ir := range s.cfg.Reasoner.Reason(req.Query, hops) {
			if ir == nil {
				continue
			}
			var from, to *graph.Entity
			if resolver != nil {
				from, _ = resolver(ir.From)
				to, _ = resolver(ir.To)
			}
			if !allowedEntity(from) || !allowedEntity(to) {
				continue
			}
			pathOK := true
			for _, rel := range ir.Path {
				if !allowedRelation(rel) {
					pathOK = false
					break
				}
			}
			if !pathOK {
				continue
			}
			out.Inferences = append(out.Inferences, inferredRelationDTO{
				From:       ir.From,
				To:         ir.To,
				Type:       ir.Type,
				Confidence: ir.Confidence,
				Rule:       ir.Rule,
				Hops:       len(ir.Path),
			})
		}
	case strings.TrimSpace(req.From) != "" && strings.TrimSpace(req.To) != "":
		for _, p := range s.cfg.Reasoner.ExplorePaths(req.From, req.To) {
			if p == nil {
				continue
			}
			pathOK := true
			for _, e := range p.Entities {
				if !allowedEntity(e) {
					pathOK = false
					break
				}
			}
			if !pathOK {
				continue
			}
			gp := graphPathDTO{Entities: make([]string, 0, len(p.Entities)), Relations: make([]string, 0, len(p.Relations))}
			for _, e := range p.Entities {
				if e != nil {
					gp.Entities = append(gp.Entities, e.ID)
				}
			}
			for _, rel := range p.Relations {
				if rel != nil {
					gp.Relations = append(gp.Relations, fmt.Sprintf("%s->%s:%s", rel.From, rel.To, rel.Type))
				}
			}
			out.Paths = append(out.Paths, gp)
		}
	default:
		writeError(w, http.StatusBadRequest, ErrCodeBadRequest, "either query or both from and to are required")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// toEntityDTO converts a graph entity into its API representation.
func toEntityDTO(e *graph.Entity) entityDTO {
	if e == nil {
		return entityDTO{}
	}
	return entityDTO{
		ID:           e.ID,
		Label:        e.Label,
		Type:         string(e.Type),
		Properties:   e.Properties,
		SourceChunks: e.SourceChunks,
	}
}

// handleWhoami reports the subject this request authenticated as.
//
// The server has always known it -- RequireAuth resolves it and puts it in the
// request context -- and never told the caller. That gap is why a client can
// hold a credential and still be unable to record who it is: it knows what it
// sent, not what the server decided that meant, and those are different
// claims. Only the second is worth writing into an audit trail.
//
// Deliberately behind the same authentication as the data endpoints. An
// unauthenticated caller has no subject to report, and answering them at all
// would turn this into an oracle for which key names which person.
func (s *Server) handleWhoami(w http.ResponseWriter, r *http.Request) {
	subject := Subject(r)
	if subject == "" {
		writeJSON(w, http.StatusOK, map[string]any{
			"authenticated": false,
			"detail": "this server has no authenticator configured, so it vouches for nobody; " +
				"a caller's asserted identity here is not verified by anything",
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"authenticated": true,
		"subject":       subject,
		"namespaces":    RequestNamespaces(r),
	})
}
