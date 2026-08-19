// Command production demonstrates deploying recall as a service the way a
// production system would:
//
//  1. assemble the full API server in-process with app.BuildAPIServer —
//     the exact same wiring the standalone recall-server binary uses
//     (SQLite store, RAG pipeline, knowledge graph, reasoning engine),
//  2. serve it over HTTP on an ephemeral local port (no fixed port, so the
//     example never conflicts with a running server),
//  3. drive it exclusively through the typed client package, as a real
//     application or the recall CLI would, and
//  4. shut everything down gracefully.
//
// The example is deterministic and offline (mock embedder, no auth). To run
// a real deployment, replace the in-process serve step with the standalone
// binary:
//
//	recall-server -config /etc/recall/recall.yaml
//
// Run it with:
//
//	go run ./example/production
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/deagy/recall/app"
	"github.com/deagy/recall/client"
	"github.com/deagy/recall/config"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	workDir, err := os.MkdirTemp("", "recall-production-*")
	if err != nil {
		log.Fatalf("creating work dir: %v", err)
	}
	defer os.RemoveAll(workDir)

	// 1. Assemble the service + API server from a programmatic config.
	cfg := &config.Config{}
	cfg.WithDefaults()
	cfg.Store.Backend = config.BackendSQLite
	cfg.Store.Path = filepath.Join(workDir, "recall.db")
	// cfg.Store.Embedder is "mock" by default: deterministic and offline.
	// For real deployments set Type to "openai"/"cohere"/"ollama" and point
	// APIKeyEnv at the environment variable holding the key.

	srv, cleanup, err := app.BuildAPIServer(cfg)
	if err != nil {
		log.Fatalf("building api server: %v", err)
	}
	defer cleanup()

	// 2. Serve the handler on an ephemeral port.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("listening: %v", err)
	}
	hsrv := &http.Server{
		Handler:     srv.Handler(),
		ReadTimeout: 30 * time.Second,
	}
	go func() {
		if err := hsrv.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Printf("server: %v", err)
		}
	}()
	defer func() {
		sctx, scancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer scancel()
		_ = hsrv.Shutdown(sctx)
	}()
	fmt.Printf("api server listening on http://%s\n\n", ln.Addr())

	// 3. Drive the service through the typed client, like a real app.
	c, err := client.New(client.Config{
		BaseURL: "http://" + ln.Addr().String(),
		Timeout: 10 * time.Second,
	})
	if err != nil {
		log.Fatalf("creating client: %v", err)
	}

	health, err := c.Health(ctx)
	if err != nil {
		log.Fatalf("health check: %v", err)
	}
	printJSON("health", health)

	upload, err := c.Upload(ctx, client.UploadRequest{
		ID:     "go-guide",
		Title:  "Go Programming Guide",
		Source: "example://production",
		Tags:   []string{"language", "go"},
		Metadata: map[string]any{
			"language": "go",
			"kind":     "guide",
		},
		Content: "Go is a statically typed, compiled programming language designed at Google. It emphasizes simplicity and built-in concurrency through goroutines and channels.",
	})
	if err != nil {
		log.Fatalf("upload: %v", err)
	}
	printJSON("upload", upload)

	search, err := c.Search(ctx, "compiled language with concurrency", client.SearchOptions{TopK: 2})
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	printJSON("search", search)

	rag, err := c.RAG(ctx, "Which company designed Go?", true)
	if err != nil {
		log.Fatalf("rag: %v", err)
	}
	printJSON("rag", rag)

	diag, err := c.Diagnostics(ctx)
	if err != nil {
		log.Fatalf("diagnostics: %v", err)
	}
	printJSON("diagnostics", diag)

	// 4. Graceful shutdown happens via the deferred Shutdown/cleanup calls.
	fmt.Println("\nproduction example complete (server stopped gracefully)")
}

func printJSON(label string, v any) {
	b, err := json.MarshalIndent(v, "  ", "  ")
	if err != nil {
		log.Fatalf("marshaling %s: %v", label, err)
	}
	fmt.Printf("--- %s ---\n%s\n\n", label, b)
}
