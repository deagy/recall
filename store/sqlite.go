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
	"sync"
	"time"

	_ "modernc.org/sqlite"

	"github.com/deagy/recall/bm25"
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
	bm25s    map[string]*bm25.BM25 // fallback BM25 for namespaces without FTS5
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
		bm25s:    make(map[string]*bm25.BM25),
	}

	if err := s.createSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating schema: %w", err)
	}

	return s, nil
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
`
	_, err := s.db.Exec(schema)
	return err
}

// Upload processes a document: chunks it, embeds the chunks, and stores them in SQLite.
func (s *SQLiteStore) Upload(ctx context.Context, doc *core.Document, content string) error {
	if doc == nil {
		return core.ErrInvalidChunk
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

	ns := s.config.Namespace

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	now := time.Now()
	for _, chunk := range chunks {
		metaJSON, err := serializeMetadata(chunk.Metadata)
		if err != nil {
			return fmt.Errorf("serializing metadata: %w", err)
		}

		_, err = tx.ExecContext(ctx,
			`INSERT OR REPLACE INTO chunks (id, content, document_ref, chunk_index, namespace, metadata, created_at, updated_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			chunk.ID, chunk.Content, chunk.DocumentRef, chunk.ChunkIndex,
			ns, metaJSON, now.Format("2006-01-02T15:04:05Z"), now.Format("2006-01-02T15:04:05Z"))
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

		s.mu.RLock()
		bm25Idx, ok := s.bm25s[ns]
		s.mu.RUnlock()
		if !ok {
			s.mu.Lock()
			bm25Idx = bm25.New(bm25.DefaultConfig())
			s.bm25s[ns] = bm25Idx
			s.mu.Unlock()
		}
		bm25Idx.AddDocument(chunk.ID, chunk.Content)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}

	doc.ChunkCount = len(chunks)
	return nil
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

	rows, err := s.db.QueryContext(ctx,
		`SELECT e.chunk_id, c.content, c.document_ref, c.chunk_index, c.metadata, e.namespace, e.embedding
		 FROM embeddings e
		 JOIN chunks c ON e.chunk_id = c.id
		 WHERE e.namespace = ?`,
		s.config.Namespace)
	if err != nil {
		return nil, fmt.Errorf("querying embeddings: %w", err)
	}
	defer rows.Close()

	var results []index.SearchResult
	for rows.Next() {
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

// SearchHybrid performs hybrid search combining vector similarity and BM25 keyword scores.
func (s *SQLiteStore) SearchHybrid(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error) {
	if query == "" {
		return nil, core.ErrEmptyQuery
	}

	vecResults, err := s.Search(ctx, query, opts)
	if err != nil {
		return nil, err
	}

	vecScoreMap := make(map[string]float64)
	for _, r := range vecResults {
		vecScoreMap[r.Chunk.ID] = r.Score
	}

	// Use BM25 fallback (FTS5 not available with modernc.org/sqlite)
	bm25Scores := s.searchBm25Fallback(query, opts)

	return fuseScores(vecScoreMap, bm25Scores, vecResults, opts)
}

// searchBm25Fallback uses the in-memory BM25 index as fallback.
func (s *SQLiteStore) searchBm25Fallback(query string, opts index.SearchOptions) map[string]float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	bm25Idx, ok := s.bm25s[s.config.Namespace]
	if !ok {
		return nil
	}

	results := bm25Idx.Search(query)
	scores := make(map[string]float64)
	for _, r := range results {
		scores[r.DocID] = r.Score
	}
	return scores
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
func (s *SQLiteStore) DeleteChunk(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(context.Background(), nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(context.Background(),
		`DELETE FROM chunks WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("deleting chunk: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing: %w", err)
	}
	return nil
}

// DeleteDocument removes all chunks belonging to a document.
func (s *SQLiteStore) DeleteDocument(docID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(context.Background(), nil)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.ExecContext(context.Background(),
		`DELETE FROM chunks WHERE document_ref = ?`, docID)
	if err != nil {
		return fmt.Errorf("deleting document chunks: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing: %w", err)
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

// Close cleans up resources.
func (s *SQLiteStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.db.Close()
}

// serializeMetadata converts chunk metadata to a JSON string.
func serializeMetadata(meta map[string]core.Value) (interface{}, error) {
	if meta == nil {
		return nil, nil
	}
	// Use a map of interface{} for JSON serialization
	m := make(map[string]interface{}, len(meta))
	for k, v := range meta {
		m[k] = v
	}
	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	return string(b), nil
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

// fuseScores fuses vector and BM25 scores into final results.
func fuseScores(vecScoreMap, bm25Scores map[string]float64, vecResults []index.SearchResult, opts index.SearchOptions) ([]index.SearchResult, error) {
	// Collect all unique IDs
	allIDs := make(map[string]bool)
	for id := range vecScoreMap {
		allIDs[id] = true
	}
	for id := range bm25Scores {
		allIDs[id] = true
	}

	type fusedResult struct {
		chunk *core.Chunk
		score float64
	}
	var results []fusedResult

	for id := range allIDs {
		vecScore := vecScoreMap[id]
		bm25Score := bm25Scores[id]

		var fusedScore float64
		if opts.Fusion != nil {
			fusionInput := []map[string]float64{vecScoreMap, bm25Scores}
			fusedMap := opts.Fusion.Fuse(fusionInput...)
			fusedScore = fusedMap[id]
		} else {
			alpha := opts.BM25Weight
			if alpha == 0 {
				alpha = 0.5 // default
			}
			fusedScore = alpha*vecScore + (1-alpha)*bm25Score
		}

		if fusedScore > 0 {
			var chunk *core.Chunk
			for _, r := range vecResults {
				if r.Chunk.ID == id {
					chunk = r.Chunk
					break
				}
			}
			if chunk == nil {
				continue
			}
			results = append(results, fusedResult{chunk: chunk, score: fusedScore})
		}
	}

	searchResults := make([]index.SearchResult, len(results))
	for i, r := range results {
		searchResults[i] = index.SearchResult{Chunk: r.chunk, Score: r.score}
	}
	return searchResults, nil
}

// sortResultsByScore sorts search results by score descending.
func sortResultsByScore(results []index.SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}
