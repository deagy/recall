package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/deagy/recall/client"
	"github.com/deagy/recall/config"
)

// globalOptions carries the parsed global state shared by all commands.
type globalOptions struct {
	// Flags (bound to persistent flags).
	configPath string
	server     string
	apiKey     string
	output     string
	timeout    time.Duration
	namespace  string

	// Resolved state (filled by init in PersistentPreRunE).
	cfg       *config.Config
	cli       *client.Client // non-nil in server mode
	serverURL string         // effective server URL in server mode
}

// newRootCmd builds the root command with all subcommands.
func newRootCmd(version string) *cobra.Command {
	o := &globalOptions{}
	root := &cobra.Command{
		Use:   "recall",
		Short: "Recall RAG toolkit — ingest, search, and reason over your knowledge",
		Long: `recall is the command-line interface for the Recall RAG toolkit.

By default, commands run in-process against the store from your
configuration (a SQLite file with "store.backend: sqlite", or an in-memory
store). To operate a running recall-server instead, pass --server URL or
set cli.url in the config file; the data commands (upload, search,
hybrid-search, rag, graph, reason, store info) then act as an HTTP client.

Configuration is resolved in order: --config flag, $HOME/.recall.yaml
(also .yml or .json), built-in defaults. Environment variables of the form
RECALL__SECTION__KEY (e.g. RECALL__CLI__API_KEY) override file values.`,
		Version:       version,
		SilenceErrors: true,
		SilenceUsage:  true,
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			return o.init(cmd)
		},
		Example: `  # Ingest a directory of markdown notes into a local SQLite store
  recall upload ./docs

  # Vector search (local mode)
  recall search "how does the index work" --top-k 5

  # Hybrid search against a running server
  recall --server http://localhost:8080 hybrid-search "indexing" --bm25-weight 0.7

  # Run a RAG query and print the assembled prompt
  recall rag "Why is chunking important?" --top-k 5

  # Inspect the store, then back it up
  recall store info
  recall store backup recall.db.backup

  # Compare two evaluation reports (CI gate)
  recall eval compare baseline.json current.json`,
	}

	pf := root.PersistentFlags()
	pf.StringVar(&o.configPath, "config", "", "path to a JSON or YAML config file (default: $HOME/.recall.yaml, then built-in defaults)")
	pf.StringVar(&o.server, "server", "", "URL of a running recall-server; switches data commands to server mode (overrides cli.url)")
	pf.StringVar(&o.apiKey, "api-key", "", "API key for server mode (overrides RECALL__CLI__API_KEY and cli.api_key)")
	pf.StringVarP(&o.output, "output", "o", "", "output format: table, json, or yaml (default from config, then table)")
	pf.DurationVar(&o.timeout, "timeout", 0, "timeout for server requests (default from config, then 30s)")
	pf.StringVar(&o.namespace, "namespace", "", "namespace for local operations (default from store.namespace in config)")

	root.AddCommand(
		newUploadCmd(o),
		newSearchCmd(o),
		newHybridSearchCmd(o),
		newRAGCmd(o),
		newGraphCmd(o),
		newReasonCmd(o),
		newStoreCmd(o),
		newClusterCmd(o),
		newEvalCmd(o),
	)
	return root
}

// init loads the configuration, validates it, and resolves the execution
// mode (local vs. server).
func (o *globalOptions) init(cmd *cobra.Command) error {
	cfg, err := o.loadConfig()
	if err != nil {
		return err
	}
	cfg.ApplyEnv("")
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	o.cfg = cfg

	// Effective output format: flag > config > "table".
	out := o.output
	if out == "" {
		out = cfg.CLI.Output
	}
	if out == "" {
		out = config.OutputTable
	}
	switch out {
	case config.OutputTable, config.OutputJSON, config.OutputYAML:
		o.output = out
	default:
		return fmt.Errorf("invalid output format %q (want table, json, or yaml)", out)
	}

	// Server mode: an explicit --server flag wins, then cli.url from the
	// config file or environment.
	serverURL := cfg.CLI.URL
	if cmd.Flags().Changed("server") {
		serverURL = o.server
	}
	if serverURL == "" {
		return nil // local mode
	}
	o.serverURL = serverURL

	apiKey := cfg.CLI.APIKey
	if cmd.Flags().Changed("api-key") {
		apiKey = o.apiKey
	}
	timeout := o.timeout
	if timeout <= 0 {
		timeout = cfg.CLI.Timeout.AsDuration()
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	cli, err := client.New(client.Config{BaseURL: serverURL, APIKey: apiKey, Timeout: timeout})
	if err != nil {
		return err
	}
	o.cli = cli
	return nil
}

// loadConfig resolves the configuration: explicit --config path, then
// $HOME/.recall.{yaml,yml,json}, then built-in defaults.
func (o *globalOptions) loadConfig() (*config.Config, error) {
	if o.configPath != "" {
		return config.Load(o.configPath)
	}
	if home, err := os.UserHomeDir(); err == nil {
		for _, name := range []string{".recall.yaml", ".recall.yml", ".recall.json"} {
			path := filepath.Join(home, name)
			if _, err := os.Stat(path); err == nil {
				return config.Load(path)
			}
		}
	}
	cfg := &config.Config{}
	cfg.WithDefaults()
	return cfg, nil
}

// local reports whether the command runs in local (in-process) mode.
func (o *globalOptions) local() bool { return o.cli == nil }

// requireLocal rejects commands that only work in-process.
func (o *globalOptions) requireLocal(name string) error {
	if o.cli != nil {
		return fmt.Errorf("%s is a local-only command; rerun without --server (and clear cli.url) to operate on the local store", name)
	}
	return nil
}

// effectiveNamespace returns the namespace for local operations: the
// --namespace flag when set, then the configured store namespace, then
// "default".
func (o *globalOptions) effectiveNamespace() string {
	if o.namespace != "" {
		return o.namespace
	}
	if o.cfg.Store.Namespace != "" {
		return o.cfg.Store.Namespace
	}
	return "default"
}
