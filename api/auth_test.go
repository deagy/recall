package api

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// signJWT creates an HS256-signed token with the given claims.
func signJWT(t *testing.T, secret string, claims map[string]any) string {
	t.Helper()
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshaling claims: %v", err)
	}
	payloadB64 := base64.RawURLEncoding.EncodeToString(payload)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(header + "." + payloadB64))
	sig := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	return header + "." + payloadB64 + "." + sig
}

func newRequest(t *testing.T, method, target string, header http.Header) *http.Request {
	t.Helper()
	req := httptest.NewRequest(method, target, nil)
	if header != nil {
		// Copy via Add so keys are MIME-canonicalized, exactly like the
		// real net/http server does for incoming headers.
		req.Header = make(http.Header, len(header))
		for k, vs := range header {
			for _, v := range vs {
				req.Header.Add(k, v)
			}
		}
	}
	return req
}

func TestAPIKeyAuth_XAPIKeyHeader(t *testing.T) {
	a := NewAPIKeyAuth("key-1", "key-2")
	req := newRequest(t, "GET", "/", http.Header{"X-API-Key": []string{"key-2"}})
	subject, ok := a.Authenticate(req)
	if !ok || subject != "key-2" {
		t.Fatalf("Authenticate = (%q, %v), want (key-2, true)", subject, ok)
	}
}

func TestAPIKeyAuth_BearerHeader(t *testing.T) {
	a := NewAPIKeyAuth("key-1")
	req := newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Bearer key-1"}})
	subject, ok := a.Authenticate(req)
	if !ok || subject != "key-1" {
		t.Fatalf("Authenticate = (%q, %v), want (key-1, true)", subject, ok)
	}
}

func TestAPIKeyAuth_Rejects(t *testing.T) {
	a := NewAPIKeyAuth("key-1")
	cases := []*http.Request{
		newRequest(t, "GET", "/", nil),
		newRequest(t, "GET", "/", http.Header{"X-API-Key": []string{"wrong"}}),
		newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Bearer wrong"}}),
		newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Basic key-1"}}),
	}
	for i, req := range cases {
		if subject, ok := a.Authenticate(req); ok {
			t.Errorf("case %d: expected rejection, got subject %q", i, subject)
		}
	}
	if subject, ok := (*APIKeyAuth)(nil).Authenticate(newRequest(t, "GET", "/", nil)); ok {
		t.Errorf("nil authenticator should reject, got %q", subject)
	}
}

func TestJWTAuth_Valid(t *testing.T) {
	now := time.Now().Unix()
	auth, err := NewJWTAuth(JWTConfig{Secret: "s3cr3t", Issuer: "recall", Audience: "api"})
	if err != nil {
		t.Fatalf("NewJWTAuth: %v", err)
	}
	token := signJWT(t, "s3cr3t", map[string]any{
		"iss": "recall", "sub": "alice", "aud": "api",
		"exp": now + 300, "nbf": now - 10, "iat": now - 10,
	})
	req := newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Bearer " + token}})
	subject, ok := auth.Authenticate(req)
	if !ok || subject != "alice" {
		t.Fatalf("Authenticate = (%q, %v), want (alice, true)", subject, ok)
	}
}

func TestJWTAuth_AudienceArray(t *testing.T) {
	now := time.Now().Unix()
	auth, err := NewJWTAuth(JWTConfig{Secret: "s3cr3t", Audience: "api"})
	if err != nil {
		t.Fatalf("NewJWTAuth: %v", err)
	}
	token := signJWT(t, "s3cr3t", map[string]any{
		"sub": "bob", "aud": []string{"web", "api"}, "exp": now + 300,
	})
	req := newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Bearer " + token}})
	if subject, ok := auth.Authenticate(req); !ok || subject != "bob" {
		t.Fatalf("Authenticate = (%q, %v), want (bob, true)", subject, ok)
	}
}
func TestJWTAuth_Rejects(t *testing.T) {
	now := time.Now().Unix()
	auth, err := NewJWTAuth(JWTConfig{Secret: "s3cr3t", Issuer: "recall", Audience: "api"})
	if err != nil {
		t.Fatalf("NewJWTAuth: %v", err)
	}

	valid := map[string]any{"iss": "recall", "sub": "alice", "aud": "api", "exp": now + 300}
	cases := map[string]string{
		"malformed":     "not-a-jwt",
		"bad signature": signJWT(t, "other-secret", valid),
		"expired":       signJWT(t, "s3cr3t", map[string]any{"iss": "recall", "sub": "alice", "aud": "api", "exp": now - 10}),
		"not yet":       signJWT(t, "s3cr3t", map[string]any{"iss": "recall", "sub": "alice", "aud": "api", "exp": now + 300, "nbf": now + 300}),
		"bad issuer":    signJWT(t, "s3cr3t", map[string]any{"iss": "other", "sub": "alice", "aud": "api", "exp": now + 300}),
		"bad audience":  signJWT(t, "s3cr3t", map[string]any{"iss": "recall", "sub": "alice", "aud": "other", "exp": now + 300}),
		"no subject":    signJWT(t, "s3cr3t", map[string]any{"iss": "recall", "aud": "api", "exp": now + 300}),
	}
	for name, token := range cases {
		header := http.Header{}
		if token != "" {
			header.Set("Authorization", "Bearer "+token)
		}
		if subject, ok := auth.Authenticate(newRequest(t, "GET", "/", header)); ok {
			t.Errorf("%s: expected rejection, got subject %q", name, subject)
		}
	}

	// No token at all.
	if _, ok := auth.Authenticate(newRequest(t, "GET", "/", nil)); ok {
		t.Error("request without token should be rejected")
	}

	// Non-HS256 algorithm must be rejected even with a valid-looking shape.
	raw := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none"}`)) + "." +
		base64.RawURLEncoding.EncodeToString([]byte(`{"sub":"x"}`)) + "." +
		base64.RawURLEncoding.EncodeToString([]byte("sig"))
	if _, ok := auth.Authenticate(newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Bearer " + raw}})); ok {
		t.Error("alg=none token should be rejected")
	}
}

func TestJWTAuth_RequiresSecret(t *testing.T) {
	if _, err := NewJWTAuth(JWTConfig{}); err == nil {
		t.Fatal("expected error for empty secret")
	}
}

func TestCompositeAuth(t *testing.T) {
	keys := NewAPIKeyAuth("k1")
	jwt, err := NewJWTAuth(JWTConfig{Secret: "sec"})
	if err != nil {
		t.Fatalf("NewJWTAuth: %v", err)
	}
	c := NewComposite(nil, keys, jwt, nil)

	req := newRequest(t, "GET", "/", http.Header{"X-API-Key": []string{"k1"}})
	if subject, ok := c.Authenticate(req); !ok || subject != "k1" {
		t.Fatalf("api key via composite = (%q, %v)", subject, ok)
	}

	now := time.Now().Unix()
	token := signJWT(t, "sec", map[string]any{"sub": "carol", "exp": now + 300})
	req = newRequest(t, "GET", "/", http.Header{"Authorization": []string{"Bearer " + token}})
	if subject, ok := c.Authenticate(req); !ok || subject != "carol" {
		t.Fatalf("jwt via composite = (%q, %v)", subject, ok)
	}

	if _, ok := c.Authenticate(newRequest(t, "GET", "/", nil)); ok {
		t.Fatal("composite should reject unauthenticated request")
	}
}

func TestRequireAuth_Middleware(t *testing.T) {
	a := NewAPIKeyAuth("good")
	var gotSubject string
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotSubject = Subject(r)
		w.WriteHeader(http.StatusOK)
	})
	h := RequireAuth(a, "Bearer")(inner)

	// Unauthenticated request.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, "GET", "/", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status = %d, want 401", rec.Code)
	}
	if rec.Header().Get("WWW-Authenticate") == "" {
		t.Error("expected WWW-Authenticate header on 401")
	}
	var env Error
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil || env.Code != ErrCodeUnauthorized {
		t.Errorf("401 body = %q (err=%v), want unauthorized envelope", rec.Body.String(), err)
	}

	// Authenticated request.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, "GET", "/", http.Header{"X-API-Key": []string{"good"}}))
	if rec.Code != http.StatusOK || gotSubject != "good" {
		t.Fatalf("authenticated = (%d, subject %q), want (200, good)", rec.Code, gotSubject)
	}

	// Subject on a nil request is safe.
	if Subject(nil) != "" {
		t.Error("Subject(nil) should be empty")
	}
}
