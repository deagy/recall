package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRunHealthProbe(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthy.Close()
	if got := runHealthProbe(healthy.URL); got != 0 {
		t.Errorf("healthy probe exit = %d, want 0", got)
	}

	unhealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer unhealthy.Close()
	if got := runHealthProbe(unhealthy.URL); got != 1 {
		t.Errorf("unhealthy probe exit = %d, want 1", got)
	}

	// A dead server (closed immediately) must report unhealthy.
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close()
	if got := runHealthProbe(deadURL); got != 1 {
		t.Errorf("dead probe exit = %d, want 1", got)
	}
}
