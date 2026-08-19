package main

import (
	"errors"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/deagy/recall/distributed"
)

func TestClusterStatus(t *testing.T) {
	cfgPath, _ := writeSQLiteConfig(t)

	cluster := distributed.NewCluster(distributed.DefaultClusterConfig())
	if err := cluster.AddNode(&distributed.Node{ID: "n1", Address: "http://n1", Status: "online"}); err != nil {
		t.Fatalf("AddNode: %v", err)
	}
	ts := httptest.NewServer(distributed.HealthHandler(cluster, nil))
	defer ts.Close()

	out := mustRunCLI(t, "--config", cfgPath, "cluster", "status", "--node", ts.URL)
	if !strings.Contains(out, "cluster OK") {
		t.Fatalf("cluster status output: %s", out)
	}

	// JSON output parses.
	out = mustRunCLI(t, "--config", cfgPath, "cluster", "status", "--node", ts.URL, "-o", "json")
	var cs clusterStatusOutput
	decodeJSON(t, out, &cs)
	if !cs.OK || len(cs.Nodes) != 1 {
		t.Errorf("cluster status = %+v", cs)
	}

	// An unreachable node fails the command with exit code 1.
	dead := httptest.NewServer(distributed.HealthHandler(cluster, nil))
	deadURL := dead.URL
	dead.Close()
	out, err := runCLI(t, "--config", cfgPath, "cluster", "status", "--node", deadURL)
	var ee *exitError
	if !errors.As(err, &ee) || ee.Code != 1 {
		t.Fatalf("expected exit code 1, got err=%v", err)
	}
	if !strings.Contains(out, "NOT OK") {
		t.Errorf("down cluster output: %s", out)
	}

	// No nodes configured is a usage error.
	if _, err := runCLI(t, "--config", cfgPath, "cluster", "status"); err == nil || !strings.Contains(err.Error(), "no cluster nodes") {
		t.Errorf("expected no-cluster-nodes error, got %v", err)
	}
}

func TestExecuteExitCodes(t *testing.T) {
	oldArgs := os.Args
	t.Cleanup(func() { os.Args = oldArgs })

	cfgPath, _ := writeSQLiteConfig(t)

	os.Args = []string{"recall", "--config", cfgPath, "store", "info"}
	if code := execute(); code != 0 {
		t.Errorf("success exit code = %d, want 0", code)
	}

	os.Args = []string{"recall", "--config", filepath.Join(t.TempDir(), "missing.json"), "store", "info"}
	if code := execute(); code != 1 {
		t.Errorf("generic error exit code = %d, want 1", code)
	}

	// A controlled exitError keeps its code (cluster down -> 1).
	dead := httptest.NewServer(distributed.HealthHandler(distributed.NewCluster(distributed.DefaultClusterConfig()), nil))
	deadURL := dead.URL
	dead.Close()
	os.Args = []string{"recall", "--config", cfgPath, "cluster", "status", "--node", deadURL}
	if code := execute(); code != 1 {
		t.Errorf("cluster-down exit code = %d, want 1", code)
	}
}
