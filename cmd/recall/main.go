// Command recall is the Recall command-line interface.
//
// It operates in two modes:
//
//   - local (default): commands run in-process against the store defined in
//     the configuration — a SQLite file for persistence, or an in-memory
//     store. This is the mode for ingestion, search, RAG, graph, reasoning,
//     and store maintenance (info, migrate, backup, restore).
//
//   - server: when a server URL is given (--server flag or cli.url in the
//     config file), the data commands (upload, search, hybrid-search, rag,
//     graph, reason, store info) act as an HTTP client for a running
//     recall-server.
//
// Configuration is resolved in order: --config flag, $HOME/.recall.yaml
// (or .yml/.json), built-in defaults. Environment variables of the form
// RECALL__SECTION__KEY (e.g. RECALL__CLI__API_KEY) override file values.
package main

import (
	"errors"
	"fmt"
	"os"
)

// version is set at build time with -ldflags "-X main.version=v1.2.3".
var version = "dev"

func main() {
	os.Exit(execute())
}

// execute runs the root command and maps errors to exit codes:
// 0 success, 1 generic error, or a command-specific code (e.g. 2 for
// evaluation regressions in `recall eval compare`).
func execute() int {
	root := newRootCmd(version)
	err := root.Execute()
	if err == nil {
		return 0
	}
	var ee *exitError
	if errors.As(err, &ee) {
		fmt.Fprintln(os.Stderr, "recall:", ee.Message)
		return ee.Code
	}
	fmt.Fprintln(os.Stderr, "recall:", err)
	return 1
}

// exitError signals a controlled non-zero exit distinct from a plain error
// (e.g. evaluation regressions, a down cluster node).
type exitError struct {
	Code    int
	Message string
}

func (e *exitError) Error() string { return e.Message }
