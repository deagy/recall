package main

import (
	"testing"

	"github.com/deagy/recall/api"
	"github.com/deagy/recall/config"
)

func TestBuildAuthenticator(t *testing.T) {
	// Plain API keys are served by the scoped authenticator with no scope.
	auth, err := buildAuthenticator(config.AuthConfig{APIKeys: []string{"k1"}})
	if err != nil {
		t.Fatalf("plain keys: %v", err)
	}
	sc, ok := auth.(*api.ScopedAPIKeyAuth)
	if !ok {
		t.Fatalf("expected ScopedAPIKeyAuth, got %T", auth)
	}
	if ns := sc.Namespaces("k1"); ns != nil {
		t.Errorf("plain key should be unrestricted, got %v", ns)
	}

	// Scoped keys keep their scope end to end.
	scoped := config.ScopedKeyConfig{Key: "team", Namespaces: []string{"ns-a"}}
	auth, err = buildAuthenticator(config.AuthConfig{ScopedKeys: []config.ScopedKeyConfig{scoped}})
	if err != nil {
		t.Fatalf("scoped keys: %v", err)
	}
	sc, ok = auth.(*api.ScopedAPIKeyAuth)
	if !ok {
		t.Fatalf("expected ScopedAPIKeyAuth, got %T", auth)
	}
	if ns := sc.Namespaces("team"); len(ns) != 1 || ns[0] != "ns-a" {
		t.Errorf("scoped key lost its scope: %v", ns)
	}

	// Scoped keys + JWT compose, and the composite still exposes the scope.
	auth, err = buildAuthenticator(config.AuthConfig{
		ScopedKeys: []config.ScopedKeyConfig{scoped},
		JWTSecret:  "s3cret",
	})
	if err != nil {
		t.Fatalf("scoped + jwt: %v", err)
	}
	comp, ok := auth.(*api.Composite)
	if !ok {
		t.Fatalf("expected Composite, got %T", auth)
	}
	if ns := comp.Namespaces("team"); len(ns) != 1 || ns[0] != "ns-a" {
		t.Errorf("composite lost the key scope: %v", ns)
	}

	// JWT alone stays a plain JWTAuth.
	auth, err = buildAuthenticator(config.AuthConfig{JWTSecret: "s3cret"})
	if err != nil {
		t.Fatalf("jwt only: %v", err)
	}
	if _, ok := auth.(*api.JWTAuth); !ok {
		t.Errorf("expected JWTAuth, got %T", auth)
	}

	// No credentials is an error.
	if _, err := buildAuthenticator(config.AuthConfig{}); err == nil {
		t.Error("expected error when no credentials are configured")
	}
}
