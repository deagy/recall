package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

// Two credentials on one store must not see each other's content.
//
// ScopedAPIKeyAuth asserts this and 37 unit tests exercise the authenticator.
// None of them runs a server: they check that a key resolves to a subject and
// that a subject maps to namespaces, which is the mechanism rather than the
// property. The property is that a search made with one credential does not
// return the other's documents, and only a request through the whole stack
// shows that.
//
// It matters here because a shared store is what a team means by "shared",
// and the difference between one operator and several is precisely whether
// this holds.

func isolationServer(t *testing.T) (*Server, *store.MemoryStore) {
	t.Helper()
	memory, err := store.NewMemoryStore(store.Config{
		Namespace: "alice-team",
		Embedder:  embedder.NewMockEmbedder(16),
	})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = memory.Close() })

	graphStore := store.NewMemoryGraphStore()
	server, err := NewServer(Config{
		Store:    memory,
		Pipeline: pipeline.NewRAGPipeline(memory, nil),
		Graph:    graphStore,
		Reasoner: reasoning.NewEngine(graphStore.Graph(), reasoning.DefaultConfig()),
		Authenticator: NewScopedAPIKeyAuth(
			KeySpec{Key: "alice-key", Namespaces: []string{"alice-team"}},
			KeySpec{Key: "bob-key", Namespaces: []string{"bob-team"}},
		),
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return server, memory
}

func searchAs(t *testing.T, server *Server, key, query string) (int, string) {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, "/search?q="+query, nil)
	request.Header.Set("X-API-Key", key)
	recorder := httptest.NewRecorder()
	server.Handler().ServeHTTP(recorder, request)
	return recorder.Code, recorder.Body.String()
}

func TestOneCredentialDoesNotSeeAnothersNamespace(t *testing.T) {
	server, memory := isolationServer(t)

	// Content that exists only in alice's namespace.
	doc := core.NewDocument("alice-doc", "Alice's private note", "alice-doc.txt")
	if err := memory.Upload(context.Background(), doc,
		strings.Repeat("a distinctive marker phrase about quarterly planning. ", 30)); err != nil {
		t.Fatalf("seeding alice's namespace: %v", err)
	}

	aliceCode, aliceBody := searchAs(t, server, "alice-key", "quarterly+planning")
	if aliceCode != http.StatusOK {
		t.Fatalf("alice's own search returned %d: %s", aliceCode, aliceBody)
	}
	if !strings.Contains(aliceBody, "alice-doc") {
		t.Fatalf("alice cannot find her own document; the test proves nothing about isolation.\n%s",
			aliceBody)
	}

	bobCode, bobBody := searchAs(t, server, "bob-key", "quarterly+planning")
	if bobCode == http.StatusOK && strings.Contains(bobBody, "alice-doc") {
		t.Fatalf("bob's credential returned a document from alice's namespace:\n%s", bobBody)
	}
}

// An unknown credential gets nothing at all.
func TestAnUnknownCredentialIsRefused(t *testing.T) {
	server, _ := isolationServer(t)

	code, body := searchAs(t, server, "not-a-key", "anything")
	if code == http.StatusOK {
		t.Fatalf("an unknown credential was served: %d %s", code, body)
	}
}

// Each credential reports its own namespaces, not the other's.
//
// /whoami is where a client learns what it is scoped to, so a client that
// records provenance depends on this being per-credential rather than global.
func TestWhoamiReportsPerCredentialNamespaces(t *testing.T) {
	server, _ := isolationServer(t)

	namespacesFor := func(key string) []string {
		t.Helper()
		request := httptest.NewRequest(http.MethodGet, "/whoami", nil)
		request.Header.Set("X-API-Key", key)
		recorder := httptest.NewRecorder()
		server.Handler().ServeHTTP(recorder, request)
		var body struct {
			Namespaces []string `json:"namespaces"`
		}
		_ = json.Unmarshal(recorder.Body.Bytes(), &body)
		return body.Namespaces
	}

	alice, bob := namespacesFor("alice-key"), namespacesFor("bob-key")
	if len(alice) == 0 || len(bob) == 0 {
		t.Fatalf("a scoped credential reported no namespaces: alice=%v bob=%v", alice, bob)
	}
	if strings.Join(alice, ",") == strings.Join(bob, ",") {
		t.Fatalf("both credentials reported the same namespaces %v; the scope is not per-credential", alice)
	}
}
