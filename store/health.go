package store

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// Health status values reported by HealthCheck.
const (
	// StatusHealthy means the store is fully operational.
	StatusHealthy = "healthy"
	// StatusDegraded means the store is operational but has problems.
	StatusDegraded = "degraded"
	// StatusDown means the store is not reachable.
	StatusDown = "down"
)

// HealthReport is the result of a store health check.
type HealthReport struct {
	// OK is true when the store is fully healthy and reachable.
	OK bool `json:"ok"`
	// Status is "healthy", "degraded", or "down".
	Status string `json:"status"`
	// Backend is the store backend ("memory", "sqlite", or "other").
	Backend string `json:"backend"`
	// Connected is true when the store accepted a probe query.
	Connected bool `json:"connected"`
	// Count is the total number of chunks across namespaces.
	Count int `json:"count"`
	// Namespaces lists the namespaces present in the store.
	Namespaces []string `json:"namespaces,omitempty"`
	// Integrity is the database integrity report (SQLite only).
	Integrity *IntegrityReport `json:"integrity,omitempty"`
	// Issues lists human-readable problems found during the check.
	Issues []string `json:"issues,omitempty"`
	// CheckedAt is when the check ran.
	CheckedAt time.Time `json:"checked_at"`
}

// HealthCheck probes the store for connectivity, size, and (for SQLite)
// structural integrity, returning a HealthReport. It is read-only and safe to
// call while the store is in use. It reports the result in the HealthReport
// rather than returning an error for non-fatal problems; a non-nil error is
// returned only for unexpected failures.
func HealthCheck(ctx context.Context, s Store) (*HealthReport, error) {
	rep := &HealthReport{CheckedAt: time.Now().UTC()}

	if sq, ok := s.(*SQLiteStore); ok {
		rep.Backend = "sqlite"
		var one int
		if err := sq.db.QueryRowContext(ctx, `SELECT 1`).Scan(&one); err != nil {
			rep.Status = StatusDown
			rep.Issues = append(rep.Issues, "connectivity: "+err.Error())
			return rep, nil
		}
		rep.Connected = true
		rep.Count = sq.Count()
		rep.Namespaces = sq.Namespaces()

		integrity, err := sq.IntegrityCheck(ctx)
		if err != nil {
			rep.Status = StatusDegraded
			rep.Issues = append(rep.Issues, "integrity check failed: "+err.Error())
		} else {
			rep.Integrity = integrity
			if !integrity.OK {
				rep.Status = StatusDegraded
				rep.Issues = append(rep.Issues, "integrity problems detected")
			}
		}
	} else {
		rep.Backend = storeBackend(s)
		rep.Connected = true
		rep.Count = s.Count()
		rep.Namespaces = s.Namespaces()
		rep.Status = StatusHealthy
	}

	if rep.Status == "" {
		rep.Status = StatusHealthy
	}
	rep.OK = rep.Connected && rep.Status == StatusHealthy
	return rep, nil
}

// storeBackend names the backend of a non-SQLite store.
func storeBackend(s Store) string {
	if _, ok := s.(*MemoryStore); ok {
		return "memory"
	}
	return "other"
}

// Diagnostics is a detailed snapshot of a store for operators.
type Diagnostics struct {
	// Health is the health check result.
	Health HealthReport `json:"health"`
	// GeneratedAt is when the snapshot was taken.
	GeneratedAt time.Time `json:"generated_at"`
}

// DiagnosticsReport builds a Diagnostics snapshot for the store.
func DiagnosticsReport(ctx context.Context, s Store) (*Diagnostics, error) {
	health, err := HealthCheck(ctx, s)
	if err != nil {
		return nil, err
	}
	return &Diagnostics{Health: *health, GeneratedAt: time.Now().UTC()}, nil
}

// HealthHandler returns an http.Handler exposing store health and
// diagnostics:
//
//	GET /healthz      -> 200 when healthy, 503 otherwise (JSON body)
//	GET /diagnostics  -> 200 with a JSON diagnostics snapshot
//
// Other paths return 404.
func HealthHandler(s Store) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			rep, err := HealthCheck(r.Context(), s)
			code := http.StatusOK
			if err != nil || !rep.OK {
				code = http.StatusServiceUnavailable
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(code)
			_ = json.NewEncoder(w).Encode(rep)
		case "/diagnostics":
			d, err := DiagnosticsReport(r.Context(), s)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(d)
		default:
			http.NotFound(w, r)
		}
	})
}
