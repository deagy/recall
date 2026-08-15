package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"

	_ "modernc.org/sqlite"

	"github.com/deagy/recall/graph"
)

// GraphPersistence defines the interface for graph persistence operations.
type GraphPersistence interface {
	AddEntity(e *graph.Entity) error
	AddRelation(r *graph.Relation) error
	LoadFromDB() error
	Clear() error
	Close() error
}

// SQLiteGraphStore persists a knowledge graph to SQLite.
type SQLiteGraphStore struct {
	mu   sync.RWMutex
	db   *sql.DB
	graph *graph.KnowledgeGraph
}

// NewSQLiteGraphStore creates a new SQLite-backed graph store.
func NewSQLiteGraphStore(dbPath string) (*SQLiteGraphStore, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	s := &SQLiteGraphStore{
		db:    db,
		graph: graph.NewKnowledgeGraph(),
	}

	if err := s.initTables(); err != nil {
		db.Close()
		return nil, fmt.Errorf("initializing tables: %w", err)
	}

	return s, nil
}

// initTables creates the required tables if they don't exist.
func (s *SQLiteGraphStore) initTables() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS entities (
			id TEXT PRIMARY KEY,
			label TEXT NOT NULL,
			entity_type TEXT NOT NULL,
			properties TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS relations (
			from_id TEXT NOT NULL,
			to_id TEXT NOT NULL,
			rel_type TEXT NOT NULL,
			weight REAL NOT NULL,
			properties TEXT,
			PRIMARY KEY (from_id, to_id, rel_type)
		)`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("exec %q: %w", stmt[:50], err)
		}
	}
	return nil
}

// Graph returns the in-memory knowledge graph.
func (s *SQLiteGraphStore) Graph() *graph.KnowledgeGraph {
	return s.graph
}

// AddEntity adds an entity to the graph and persists it.
func (s *SQLiteGraphStore) AddEntity(e *graph.Entity) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.graph.AddEntity(e) {
		props, _ := json.Marshal(e.Properties)
		_, err := s.db.Exec(
			`INSERT OR REPLACE INTO entities (id, label, entity_type, properties) VALUES (?, ?, ?, ?)`,
			e.ID, e.Label, string(e.Type), string(props),
		)
		return err
	}
	return nil
}

// AddRelation adds a relation to the graph and persists it.
func (s *SQLiteGraphStore) AddRelation(r *graph.Relation) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.graph.AddRelation(r) {
		props, _ := json.Marshal(r.Properties)
		_, err := s.db.Exec(
			`INSERT OR REPLACE INTO relations (from_id, to_id, rel_type, weight, properties) VALUES (?, ?, ?, ?, ?)`,
			r.From, r.To, r.Type, r.Weight, string(props),
		)
		return err
	}
	return nil
}

// LoadFromDB loads all entities and relations from the database into the in-memory graph.
func (s *SQLiteGraphStore) LoadFromDB() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Load entities
	rows, err := s.db.Query(`SELECT id, label, entity_type, properties FROM entities`)
	if err != nil {
		return fmt.Errorf("querying entities: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id, name, entityType, propsJSON string
		if err := rows.Scan(&id, &name, &entityType, &propsJSON); err != nil {
			return fmt.Errorf("scanning entity: %w", err)
		}
		props := make(map[string]string)
		if propsJSON != "" {
			json.Unmarshal([]byte(propsJSON), &props)
		}
		s.graph.AddEntity(&graph.Entity{
			ID:         id,
			Label:      name,
			Type:       graph.EntityType(entityType),
			Properties: props,
		})
	}

	// Load relations
	rows, err = s.db.Query(`SELECT from_id, to_id, rel_type, weight, properties FROM relations`)
	if err != nil {
		return fmt.Errorf("querying relations: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var fromID, toID, relType string
		var weight float64
		var propsJSON string
		if err := rows.Scan(&fromID, &toID, &relType, &weight, &propsJSON); err != nil {
			return fmt.Errorf("scanning relation: %w", err)
		}
		props := make(map[string]string)
		if propsJSON != "" {
			json.Unmarshal([]byte(propsJSON), &props)
		}
		s.graph.AddRelation(&graph.Relation{
			From:         fromID,
			To:           toID,
			Type:         relType,
			Weight:       weight,
			Properties:   props,
		})
	}

	return nil
}

// Clear clears all data from the database.
func (s *SQLiteGraphStore) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(`DELETE FROM relations; DELETE FROM entities`)
	if err != nil {
		return err
	}
	s.graph = graph.NewKnowledgeGraph()
	return nil
}

// Ensure SQLiteGraphStore implements GraphPersistence.
var _ GraphPersistence = (*SQLiteGraphStore)(nil)

// Close closes the database connection.
func (s *SQLiteGraphStore) Close() error {
	return s.db.Close()
}