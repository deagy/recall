// Command recall-server runs the Recall REST API as a standalone service.
//
// Usage:
//
//	recall-server [-config /etc/recall/recall.yaml]
//
// Without -config it starts with defaults: an in-memory store with a
// deterministic mock embedder, listening on 127.0.0.1:8080 — handy for a
// local smoke test. In production, pass a config file (JSON or YAML);
// environment variables (RECALL__SECTION__KEY) override file values. The
// server supports graceful shutdown on SIGINT/SIGTERM.
package main

import (
	"context"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/deagy/recall/app"
	"github.com/deagy/recall/config"
)

func main() {
	configPath := flag.String("config", "", "path to a JSON or YAML config file (default: built-in dev defaults)")
	probeURL := flag.String("probe-url", "http://127.0.0.1:8080/healthz", "URL probed in health-probe mode")
	healthProbe := flag.Bool("health-probe", false, "probe the local health endpoint and exit (0 = healthy, 1 = unhealthy); for container HEALTHCHECK on images without curl")
	flag.Parse()

	if *healthProbe {
		os.Exit(runHealthProbe(*probeURL))
	}

	var cfg *config.Config
	if *configPath != "" {
		var err error
		cfg, err = config.Load(*configPath)
		if err != nil {
			log.Fatalf("loading config: %v", err)
		}
		cfg.ApplyEnv("")
		if err := cfg.Validate(); err != nil {
			log.Fatalf("config: %v", err)
		}
		log.Printf("loaded config from %s", *configPath)
	} else {
		cfg = &config.Config{}
		cfg.WithDefaults()
		log.Printf("no -config flag; using dev defaults (%s, in-memory store, mock embedder)", cfg.Server.Host)
	}

	srv, cleanup, err := app.BuildAPIServer(cfg)
	if err != nil {
		log.Fatalf("building server: %v", err)
	}
	defer cleanup()

	// Graceful shutdown on SIGINT/SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()

	select {
	case err := <-errCh:
		if err != nil {
			log.Fatalf("server: %v", err)
		}
	case <-ctx.Done():
		log.Printf("shutting down...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Printf("graceful shutdown: %v", err)
		}
	}
	log.Printf("stopped")
}

// runHealthProbe performs a GET against url and returns 0 for a 2xx
// response, 1 otherwise. It is used by the container HEALTHCHECK on
// distroless images that ship no curl or wget.
func runHealthProbe(url string) int {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 1
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return 0
	}
	return 1
}
