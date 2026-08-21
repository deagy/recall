// Package store provides the interface and implementations for the knowledge store.
package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/index"
)

// SQLiteStore is a persistent store backed by SQLite with FTS5 for keyword search.
type SQLiteStore struct {
	mu       sync.RWMutex
	config   Config
	embedder embedder.Embedder
	chunker  chunker.Chunker
	db       *sql.DB
	chunks   map[string]*core.Chunk // in-memory copy for HNSW
	hnsw     *index.HNSW            // HNSW index for ANN search

	autoCheckpointCancel context.CancelFunc // stops the background WAL checkpoint loop
}

// NewSQLiteStore creates a new SQLite-backed store.
// dbPath can be a file path or ":memory:" for in-memory testing.
func NewSQLiteStore(cfg Config, dbPath string) (*SQLiteStore, error) {
	if cfg.Namespace == "" {
		cfg.Namespace = "default"
	}
	if cfg.Embedder == nil {
		cfg.Embedder = embedder.NewMockEmbedder(384)
	}
	if cfg.ChunkerFactory == nil {
		cfg.ChunkerFactory = chunker.NewFixed
	}

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	// Restrict the pool to a single connection: with ":memory:" databases every
	// pooled connection is a separate database, so the schema created on one
	// connection would be invisible to the others. A single serialized
	// connection is also the safe pattern for embedded SQLite writers.
	db.SetMaxOpenConns(1)

	// Enable WAL mode for better concurrency
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("enabling WAL: %w", err)
	}

	s := &SQLiteStore{
		config:   cfg,
		embedder: cfg.Embedder,
		chunker:  cfg.ChunkerFactory(chunker.DefaultConfig()),
		db:       db,
		chunks:   make(map[string]*core.Chunk),
	}

	if err := s.createSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating schema: %w", err)
	}

	// Automatically apply any schema migrations supplied in the config.
	// The base schema is always created first (createSchema), so migrations
	// only need to evolve the schema for newer versions.
	if len(cfg.Migrations) > 0 {
		if err := NewMigrator(db, cfg.Migrations).Migrate(context.Background()); err != nil {
			db.Close()
			return nil, fmt.Errorf("applying migrations: %w", err)
		}
	}

	return s, nil
}

// Migrate applies any pending schema migrations from the store's config. It is
// idempotent: already-applied migrations are skipped. Call it with your own
// context when you need to control cancellation of long migrations.
func (s *SQLiteStore) Migrate(ctx context.Context) error {
	if len(s.config.Migrations) == 0 {
		return nil
	}
	// Serialize migration runs so two concurrent callers cannot both attempt
	// to apply the same version.
	s.mu.Lock()
	defer s.mu.Unlock()
	return NewMigrator(s.db, s.config.Migrations).Migrate(ctx)
}

// SchemaVersion returns the current schema version (PRAGMA user_version).
func (s *SQLiteStore) SchemaVersion(ctx context.Context) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var v int
	err := s.db.QueryRowContext(ctx, `PRAGMA user_version`).Scan(&v)
	return v, err
}

func (s *SQLiteStore) createSchema() error {
	schema := `
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    document_ref TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    namespace TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
`
	_, err := s.db.Exec(schema)
	return err
}

// Upload processes a document: chunks it, embeds the chunks, and stores them in SQLite.
func (s *SQLiteStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	if doc == nil {
		return core.ErrInvalidDocument
	}
	if content == "" {
		return core.ErrInvalidChunk
	}

	chunks, err := s.chunker.Chunk(doc, content)
	if err != nil {
		return fmt.Errorf("chunking: %w", err)
	}
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks produced")
	}

	contents := make([]string, len(chunks))
	for i, c := range chunks {
		contents[i] = c.Content
	}
	embeddings, err := s.embedder.EmbedBatch(ctx, contents)
	if err != nil {
		return fmt.Errorf("embedding: %w", err)
	}
	for i, chunk := range chunks {
		chunk.Embedding = embeddings[i]
	}

	ns := doc.Namespace
	if ns == "" {
		ns = s.config.Namespace
	}
	stampChunkNamespace(chunks, ns)

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	// RFC3339 UTC. A bare "Z" in a layout string is a literal, not a
	// timezone token, so formatting local time with "2006-01-02T15:04:05Z"
	// would stamp local time with a fake UTC marker.
	now := time.Now().UTC().Format(time.RFC3339)
	for _, chunk := range chunks {
		metaJSON, err := serializeMetadata(chunk.Metadata)
		if err != nil {
			return fmt.Errorf("serializing metadata: %w", err)
		}

		_, err = tx.ExecContext(ctx,
			`INSERT OR REPLACE INTO chunks (id, content, document_ref, chunk_index, namespace, metadata, created_at, updated_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			chunk.ID, chunk.Content, chunk.DocumentRef, chunk.ChunkIndex,
			ns, metaJSON, now, now)
		if err != nil {
			return fmt.Errorf("inserting chunk: %w", err)
		}

		embBytes := packEmbedding(chunk.Embedding)
		_, err = tx.ExecContext(ctx,
			`INSERT OR REPLACE INTO embeddings (chunk_id, namespace, embedding) VALUES (?, ?, ?)`,
			chunk.ID, ns, embBytes)
		if err != nil {
			return fmt.Errorf("inserting embedding: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}

	// Update in-memory copy for HNSW under the store lock so concurrent
	// searches never observe a partially updated mirror.
	s.mu.Lock()
	for _, chunk := range chunks {
		s.chunks[chunk.ID] = chunk
	}

	// Build HNSW if threshold reached, or keep the existing mirror in sync.
	if s.hnsw == nil {
		if len(s.chunks) > index.HNSWThreshold {
			s.buildHNSW()
		}
	} else {
		for _, chunk := range chunks {
			if !s.hnsw.Contains(chunk.ID) {
				s.hnsw.Add(chunk.ID, chunk.Embedding)
			}
		}
	}
	s.mu.Unlock()

	doc.ChunkCount = len(chunks)
	return nil
}

// buildHNSW constructs the HNSW graph from in-memory chunks.
// Callers must hold s.mu (write lock).
func (s *SQLiteStore) buildHNSW() {
	s.hnsw = index.NewHNSW(s.embedder.Dimension(), index.DefaultHNSWConfig())
	for _, chunk := range s.chunks {
		s.hnsw.Add(chunk.ID, chunk.Embedding)
	}
}

// searchHNSW performs ANN search using the HNSW index.
func (s *SQLiteStore) searchHNSW(ctx context.Context, embed []float32, opts index.SearchOptions) ([]index.SearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check context before search
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	ids := s.hnsw.Search(embed, opts.TopK)

	var results []index.SearchResult
	for _, id := range ids {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		chunk, ok := s.chunks[id]
		if !ok {
			continue
		}
		sim := cosineSimilarity(embed, chunk.Embedding)
		if !matchesFilters(sim, chunk, opts.Filters) {
			continue
		}
		results = append(results, index.SearchResult{Chunk: chunk, Score: sim})
	}

	sortResultsByScore(results)
	if opts.TopK > 0 && len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	return results, nil
}

// Search finds the most relevant chunks for a query string using vector similarity.
func (s *SQLiteStore) Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if query == "" {
		return nil, core.ErrEmptyQuery
	}

	queryEmbed, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	// Use HNSW for ANN search if available and dataset is large enough.
	// The gate is read under the store lock; searchHNSW acquires it again.
	s.mu.RLock()
	useHNSW := s.hnsw != nil && len(s.chunks) > index.HNSWThreshold
	s.mu.RUnlock()
	if useHNSW {
		return s.searchHNSW(ctx, queryEmbed, opts)
	}

	// Brute-force vector search. Search spans every namespace present in
	// this store (matching MemoryStore and the HNSW path above, which both
	// search the whole store), so documents routed to a custom
	// Document.Namespace remain retrievable.
	rows, err := s.db.QueryContext(ctx,
		`SELECT e.chunk_id, c.content, c.document_ref, c.chunk_index, c.metadata, e.namespace, e.embedding
		 FROM embeddings e
		 JOIN chunks c ON e.chunk_id = c.id`)
	if err != nil {
		return nil, fmt.Errorf("querying embeddings: %w", err)
	}
	defer rows.Close()

	var results []index.SearchResult
	for rows.Next() {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		var chunkID, content, docRef, metaJSON, ns string
		var chunkIdx int
		var embBytes []byte
		if err := rows.Scan(&chunkID, &content, &docRef, &chunkIdx, &metaJSON, &ns, &embBytes); err != nil {
			return nil, fmt.Errorf("scanning row: %w", err)
		}

		embedding := unpackEmbedding(embBytes)
		sim := cosineSimilarity(queryEmbed, embedding)

		chunk := &core.Chunk{
			ID:          chunkID,
			Content:     content,
			DocumentRef: docRef,
			ChunkIndex:  chunkIdx,
			Embedding:   embedding,
			Metadata:    deserializeMetadata(metaJSON),
		}

		if !matchesFilters(sim, chunk, opts.Filters) {
			continue
		}

		results = append(results, index.SearchResult{Chunk: chunk, Score: sim})
	}

	sortResultsByScore(results)
	if opts.TopK > 0 && len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	return results, nil
}

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores via FTS5.
func (s *SQLiteStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if query == "" {
		return nil, core.ErrEmptyQuery
	}

	// Vector search
	embed, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	if embed == nil {
		return nil, fmt.Errorf("embedding query failed")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// FTS5 keyword search
	ftsResults := s.searchFTS5(query, opts)

	// Build fused results
	return s.fuseFTS5Results(ctx, embed, ftsResults, opts)
}

// searchFTS5 performs keyword search using FTS5.
func (s *SQLiteStore) searchFTS5(query string, opts index.SearchOptions) map[string]float64 {
	// Escape single quotes for FTS5 query
	escaped := strings.ReplaceAll(query, "'", "''")
	limit := opts.TopK
	if limit <= 0 {
		limit = 10
	}
	sql := `
		SELECT c.id, rank
		FROM chunks c
		INNER JOIN chunks_fts ON chunks_fts.rowid = c.rowid
		WHERE chunks_fts MATCH ?
		ORDER BY rank
		LIMIT ?
	`

	rows, err := s.db.Query(sql, escaped, limit)
	if err != nil {
		// FTS5 unavailable (e.g. a database created before the FTS
		// schema existed): degrade to vector-only scoring instead of
		// failing the whole hybrid search.
		return make(map[string]float64)
	}
	defer rows.Close()

	scores := make(map[string]float64)
	for rows.Next() {
		var id string
		var rank float64
		if err := rows.Scan(&id, &rank); err != nil {
			continue
		}
		// FTS5 rank is a bm25 score where lower (more negative) is better;
		// flip the sign so that higher means better, matching vector scores.
		scores[id] = -rank
	}
	return scores
}

// fuseFTS5Results fuses vector and FTS5 scores into final results. Chunks
// matched only by the FTS5 query (keyword-only hits) are included as well.
// Callers must hold s.mu (at least for read).
func (s *SQLiteStore) fuseFTS5Results(ctx context.Context, query []float32, ftsResults map[string]float64, opts index.SearchOptions) ([]index.SearchResult, error) {
	// Load all chunks with embeddings.
	rows, err := s.db.Query(`
		SELECT c.id, c.content, c.document_ref, c.chunk_index, c.metadata, e.embedding
		FROM chunks c
		LEFT JOIN embeddings e ON c.id = e.chunk_id
	`)
	if err != nil {
		return nil, fmt.Errorf("querying chunks: %w", err)
	}
	defer rows.Close()

	chunks := make(map[string]*core.Chunk)
	vecScoreMap := make(map[string]float64)

	for rows.Next() {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		var chunkID, content, docRef string
		var chunkIdx int
		var metaJSON, embBytes []byte
		if err := rows.Scan(&chunkID, &content, &docRef, &chunkIdx, &metaJSON, &embBytes); err != nil {
			continue
		}

		embedding := unpackEmbedding(embBytes)
		chunks[chunkID] = &core.Chunk{
			ID:          chunkID,
			Content:     content,
			DocumentRef: docRef,
			ChunkIndex:  chunkIdx,
			Metadata:    deserializeMetadata(string(metaJSON)),
			Embedding:   embedding,
		}
		vecScoreMap[chunkID] = cosineSimilarity(query, embedding)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating chunks: %w", err)
	}

	// A custom fusion is computed once over the full score maps, so RRF and
	// friends see real ranks instead of a singleton vector map per chunk.
	var fusedMap map[string]float64
	if opts.Fusion != nil {
		fusedMap = opts.Fusion.Fuse(vecScoreMap, ftsResults)
	}

	var results []index.SearchResult
	for chunkID, chunk := range chunks {
		var fusedScore float64
		if opts.Fusion != nil {
			fusedScore = fusedMap[chunkID]
		} else {
			// Weighted sum per SearchOptions.BM25Weight: 0 = pure vector,
			// 1 = pure BM25.
			fusedScore = (1-opts.BM25Weight)*vecScoreMap[chunkID] + opts.BM25Weight*ftsResults[chunkID]
		}
		if fusedScore <= 0 {
			continue
		}
		if !matchesFilters(fusedScore, chunk, opts.Filters) {
			continue
		}
		results = append(results, index.SearchResult{Chunk: chunk, Score: fusedScore})
	}

	sortResultsByScore(results)
	if opts.TopK > 0 && len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	return results, nil
}

// GetChunk returns a chunk by its ID.
func (s *SQLiteStore) GetChunk(id string) (*core.Chunk, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var content, docRef string
	var chunkIdx int
	var metaJSON, ns string
	var embBytes []byte
	err := s.db.QueryRow(`
		SELECT c.content, c.document_ref, c.chunk_index, c.metadata, c.namespace, e.embedding
		FROM chunks c
		LEFT JOIN embeddings e ON c.id = e.chunk_id
		WHERE c.id = ?`, id).Scan(&content, &docRef, &chunkIdx, &metaJSON, &ns, &embBytes)

	if err == sql.ErrNoRows {
		return nil, false
	}
	if err != nil {
		return nil, false
	}

	chunk := &core.Chunk{
		ID:          id,
		Content:     content,
		DocumentRef: docRef,
		ChunkIndex:  chunkIdx,
		Metadata:    deserializeMetadata(metaJSON),
	}
	if len(embBytes) > 0 {
		chunk.Embedding = unpackEmbedding(embBytes)
	}
	return chunk, true
}

// DeleteChunk removes a chunk from the store.
func (s *SQLiteStore) DeleteChunk(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(ctx,
		`DELETE FROM chunks WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("deleting chunk: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing: %w", err)
	}

	// Prune the in-memory mirror so deleted chunks are not served by HNSW search.
	delete(s.chunks, id)
	return nil
}

// DeleteDocument removes all chunks belonging to a document.
func (s *SQLiteStore) DeleteDocument(ctx context.Context, docID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(ctx,
		`DELETE FROM chunks WHERE document_ref = ?`, docID)
	if err != nil {
		return fmt.Errorf("deleting document chunks: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing: %w", err)
	}

	// Prune the in-memory mirror so deleted chunks are not served by HNSW search.
	for chunkID, chunk := range s.chunks {
		if chunk.DocumentRef == docID {
			delete(s.chunks, chunkID)
		}
	}
	return nil
}

// Count returns the total number of chunks across all namespaces.
func (s *SQLiteStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM chunks`).Scan(&count)
	if err != nil {
		return 0
	}
	return count
}

// Namespaces returns the list of namespaces in the store.
func (s *SQLiteStore) Namespaces() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(`SELECT DISTINCT namespace FROM chunks ORDER BY namespace`)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var ns []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil
		}
		ns = append(ns, name)
	}
	return ns
}

// Namespace returns the store's default namespace (used for documents that
// do not override it).
func (s *SQLiteStore) Namespace() string {
	return s.config.Namespace
}

// Close cleans up resources.
func (s *SQLiteStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.autoCheckpointCancel != nil {
		s.autoCheckpointCancel()
		s.autoCheckpointCancel = nil
	}
	return s.db.Close()
}

// serializeMetadata converts chunk metadata to a JSON string.
func serializeMetadata(meta map[string]core.Value) (interface{}, error) {
	if meta == nil {
		return nil, nil
	}
	// Serialize typed values as plain JSON primitives so they round-trip:
	// marshaling the typed structs directly would produce objects
	// ({"Value": ...}) that cannot be decoded back into core.Value.
	m := make(map[string]interface{}, len(meta))
	for k, v := range meta {
		m[k] = valueToJSON(v)
	}
	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	return string(b), nil
}

// valueToJSON converts a typed Value to a plain JSON primitive.
func valueToJSON(v core.Value) interface{} {
	switch val := v.(type) {
	case core.String:
		return val.Value
	case core.Number:
		return val.Value
	case core.Boolean:
		return val.Value
	case core.URI:
		// URIs round-trip as plain strings (String on read).
		return val.Value
	case core.Literal:
		return val.Value
	case nil:
		return nil
	default:
		// Unknown Value implementation: fall back to its string form.
		return val.String()
	}
}

// deserializeMetadata parses a JSON metadata string back to a map.
func deserializeMetadata(s string) map[string]core.Value {
	if s == "" || s == "null" {
		return nil
	}
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(s), &raw); err != nil {
		return nil
	}
	result := make(map[string]core.Value, len(raw))
	for k, v := range raw {
		switch val := v.(type) {
		case string:
			result[k] = core.String{Value: val}
		case float64:
			result[k] = core.Number{Value: val}
		case bool:
			result[k] = core.Boolean{Value: val}
		default:
			// Legacy rows serialized typed values as objects
			// ({"Value": ...}); unwrap them so older databases keep working.
			if obj, ok := val.(map[string]interface{}); ok && len(obj) == 1 {
				switch inner := obj["Value"].(type) {
				case string:
					result[k] = core.String{Value: inner}
					continue
				case float64:
					result[k] = core.Number{Value: inner}
					continue
				case bool:
					result[k] = core.Boolean{Value: inner}
					continue
				}
			}
			result[k] = core.String{Value: fmt.Sprintf("%v", val)}
		}
	}
	return result
}

// packEmbedding serializes a float32 slice to bytes.
func packEmbedding(emb []float32) []byte {
	buf := make([]byte, len(emb)*4)
	for i, v := range emb {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// unpackEmbedding deserializes bytes back to a float32 slice.
func unpackEmbedding(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 4
	emb := make([]float32, n)
	for i := 0; i < n; i++ {
		emb[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return emb
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// matchesFilters checks if a chunk matches the given filters.
func matchesFilters(score float64, chunk *core.Chunk, filters []index.Filter) bool {
	if len(filters) == 0 {
		return true
	}
	for _, f := range filters {
		if !f.Match(chunk) {
			return false
		}
	}
	return true
}

// sortResultsByScore sorts search results by score descending.
func sortResultsByScore(results []index.SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}
