package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
	"github.com/deagy/recall/testutil"
)

// A caller must be able to learn which subject the server authenticated it as.
//
// The server has always resolved one -- RequireAuth puts it in the request
// context -- and until now kept it to itself. A client could therefore hold a
// credential and still not know what the server decided that credential names.
// Those are different claims: the first is something the caller chose, the
// second is not, and only the second is worth recording as who did something.

func whoamiServer(t *testing.T, auth Authenticator) *Server {
	t.Helper()
	st, err := store.NewMemoryStore(store.Config{
		Namespace: "default",
		Embedder:  testutil.NewMockEmbedder(16),
	})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	graphStore := store.NewMemoryGraphStore()
	server, err := NewServer(Config{
		Store:         st,
		Pipeline:      pipeline.NewRAGPipeline(st, nil),
		Graph:         graphStore,
		Reasoner:      reasoning.NewEngine(graphStore.Graph(), reasoning.DefaultConfig()),
		Authenticator: auth,
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return server
}

func whoami(t *testing.T, server *Server, key string) (int, map[string]any) {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, "/whoami", nil)
	if key != "" {
		request.Header.Set("X-API-Key", key)
	}
	recorder := httptest.NewRecorder()
	server.Handler().ServeHTTP(recorder, request)
	var body map[string]any
	_ = json.Unmarshal(recorder.Body.Bytes(), &body)
	return recorder.Code, body
}

func TestWhoamiReportsTheAuthenticatedSubject(t *testing.T) {
	server := whoamiServer(t, NewScopedAPIKeyAuth(
		KeySpec{Key: "alice-key", Namespaces: []string{"team"}},
	))

	code, body := whoami(t, server, "alice-key")
	if code != http.StatusOK {
		t.Fatalf("whoami returned %d, want 200: %v", code, body)
	}
	if body["authenticated"] != true {
		t.Fatalf("whoami reported %v, want authenticated", body)
	}
	if body["subject"] != "alice-key" {
		t.Fatalf("subject was %v, want the authenticated key", body["subject"])
	}
}

// Two credentials must report two subjects.
//
// One key returning the value the test seeded proves nothing about whether the
// endpoint reads the credential at all. Two keys returning two different
// answers does.
func TestWhoamiDistinguishesTwoCredentials(t *testing.T) {
	server := whoamiServer(t, NewScopedAPIKeyAuth(
		KeySpec{Key: "alice-key", Namespaces: []string{"team"}},
		KeySpec{Key: "bob-key", Namespaces: []string{"team"}},
	))

	_, aliceBody := whoami(t, server, "alice-key")
	_, bobBody := whoami(t, server, "bob-key")

	if aliceBody["subject"] == bobBody["subject"] {
		t.Fatalf("two credentials reported the same subject %v; the endpoint is not reading the credential",
			aliceBody["subject"])
	}
}

// An unauthenticated request must not be told a subject.
//
// Answering would make this an oracle for which key names which person, and
// would let a caller holding no credential record an identity it does not have.
func TestWhoamiRefusesAnUnauthenticatedCaller(t *testing.T) {
	server := whoamiServer(t, NewScopedAPIKeyAuth(
		KeySpec{Key: "alice-key", Namespaces: []string{"team"}},
	))

	code, body := whoami(t, server, "")
	if code == http.StatusOK {
		t.Fatalf("whoami answered an unauthenticated caller with %d: %v", code, body)
	}
}

// A server with no authenticator must say it vouches for nobody.
//
// Silence would be worse than the old behaviour: a client that gets no subject
// cannot tell "this server does not authenticate" from "this server has not
// been asked", and would be free to record whatever it liked.
func TestWhoamiSaysSoWhenNothingAuthenticates(t *testing.T) {
	server := whoamiServer(t, nil)

	code, body := whoami(t, server, "")
	if code != http.StatusOK {
		t.Fatalf("whoami returned %d on an unauthenticated server, want 200: %v", code, body)
	}
	if body["authenticated"] != false {
		t.Fatalf("whoami reported %v; a server with no authenticator must say it vouches for nobody", body)
	}
	if _, present := body["subject"]; present {
		t.Fatalf("whoami returned a subject with no authenticator configured: %v", body)
	}
}
