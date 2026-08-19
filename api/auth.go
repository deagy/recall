package api

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"
)

// Authenticator validates an incoming request and returns the authenticated
// subject (e.g. the API key identity or JWT "sub" claim) when valid.
type Authenticator interface {
	// Authenticate returns the subject and true when the request carries
	// valid credentials, or "" and false otherwise.
	Authenticate(r *http.Request) (string, bool)
}

type contextKey int

const (
	subjectKey contextKey = iota
	namespacesKey
)

// Subject returns the authenticated subject set by the RequireAuth
// middleware, or "" when the request was not authenticated.
func Subject(r *http.Request) string {
	if r == nil {
		return ""
	}
	s, _ := r.Context().Value(subjectKey).(string)
	return s
}

// RequestNamespaces returns the namespace scope imposed on the request by
// its authenticator (see NamespaceScoper), or nil when the request is
// unrestricted. Handlers use it to enforce namespace-scoped credentials.
func RequestNamespaces(r *http.Request) []string {
	if r == nil {
		return nil
	}
	ns, _ := r.Context().Value(namespacesKey).([]string)
	return ns
}

// NamespaceScoper is an optional interface for Authenticators that impose
// per-subject namespace restrictions (e.g. ScopedAPIKeyAuth). It returns the
// namespaces a subject may access; nil or empty means unrestricted. When the
// authenticator passed to RequireAuth implements it, the scope is injected
// into the request context (see RequestNamespaces).
type NamespaceScoper interface {
	Authenticator
	// Namespaces returns the allowed namespaces for the authenticated
	// subject, or nil when the subject is unrestricted (or unknown).
	Namespaces(subject string) []string
}

// RequireAuth wraps an http.Handler with authentication. Requests that fail
// authentication receive a 401 with a JSON error envelope (a
// WWW-Authenticate hint is included when hints are given). Successful
// requests carry the subject in the request context (see Subject).
func RequireAuth(a Authenticator, hints ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			subject, ok := a.Authenticate(r)
			if !ok {
				if len(hints) > 0 {
					w.Header().Set("WWW-Authenticate", strings.Join(hints, ", "))
				}
				writeError(w, http.StatusUnauthorized, ErrCodeUnauthorized, "missing or invalid credentials")
				return
			}
			ctx := context.WithValue(r.Context(), subjectKey, subject)
			if scoper, ok := a.(NamespaceScoper); ok {
				if ns := scoper.Namespaces(subject); len(ns) > 0 {
					ctx = context.WithValue(ctx, namespacesKey, ns)
				}
			}
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// APIKeyAuth authenticates requests by API key. The key is read from the
// X-API-Key header first, then from an "Authorization: Bearer <key>"
// header. The subject returned is the key itself, so callers can map keys
// to identities outside the API.
type APIKeyAuth struct {
	keys map[string]struct{}
}

// NewAPIKeyAuth creates an API-key authenticator from the given keys.
func NewAPIKeyAuth(keys ...string) *APIKeyAuth {
	m := make(map[string]struct{}, len(keys))
	for _, k := range keys {
		if k != "" {
			m[k] = struct{}{}
		}
	}
	return &APIKeyAuth{keys: m}
}

// Authenticate implements Authenticator.
func (a *APIKeyAuth) Authenticate(r *http.Request) (string, bool) {
	if a == nil {
		return "", false
	}
	candidates := []string{r.Header.Get("X-API-Key"), bearerToken(r)}
	for _, c := range candidates {
		if _, ok := a.keys[c]; ok {
			return c, true
		}
	}
	return "", false
}

func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if len(h) > 7 && strings.EqualFold(h[:7], "Bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}

// KeySpec pairs an API key with the namespaces it may access. An empty
// Namespaces list grants access to all namespaces (the key behaves like a
// plain APIKeyAuth key).
type KeySpec struct {
	// Key is the API key value (required).
	Key string

	// Namespaces restricts the key to these namespaces. Empty means all.
	Namespaces []string
}

// ScopedAPIKeyAuth authenticates API keys with optional per-key namespace
// scope. The key is read from the X-API-Key header first, then from an
// "Authorization: Bearer <key>" header; the subject is the key itself. A key
// whose KeySpec lists namespaces is restricted to them: uploads must target
// an allowed namespace, and search, RAG, and graph results are limited to
// chunks in allowed namespaces. It implements NamespaceScoper, so the
// RequireAuth middleware injects the scope into the request context.
type ScopedAPIKeyAuth struct {
	scopes map[string][]string
}

// NewScopedAPIKeyAuth creates a scoped API-key authenticator from the given
// key specs. Empty keys are skipped; duplicate keys keep the first spec.
func NewScopedAPIKeyAuth(specs ...KeySpec) *ScopedAPIKeyAuth {
	m := make(map[string][]string, len(specs))
	for _, s := range specs {
		if s.Key == "" {
			continue
		}
		if _, exists := m[s.Key]; exists {
			continue
		}
		m[s.Key] = s.Namespaces
	}
	return &ScopedAPIKeyAuth{scopes: m}
}

// Authenticate implements Authenticator.
func (a *ScopedAPIKeyAuth) Authenticate(r *http.Request) (string, bool) {
	if a == nil {
		return "", false
	}
	candidates := []string{r.Header.Get("X-API-Key"), bearerToken(r)}
	for _, c := range candidates {
		if _, ok := a.scopes[c]; ok {
			return c, true
		}
	}
	return "", false
}

// Namespaces implements NamespaceScoper. It returns a copy of the subject's
// allowed namespaces, or nil when the subject is unknown or unrestricted.
func (a *ScopedAPIKeyAuth) Namespaces(subject string) []string {
	if a == nil {
		return nil
	}
	ns, ok := a.scopes[subject]
	if !ok || len(ns) == 0 {
		return nil
	}
	out := make([]string, len(ns))
	copy(out, ns)
	return out
}

// jwtClaims holds the standard (RFC 7519) JWT claims used by JWTAuth.
type jwtClaims struct {
	Issuer    string   `json:"iss"`
	Subject   string   `json:"sub"`
	Audience  audience `json:"aud"`
	ExpiresAt *int64   `json:"exp"`
	NotBefore *int64   `json:"nbf"`
	IssuedAt  *int64   `json:"iat"`
}

// audience accepts both the compact string form and the array form of the
// JWT "aud" claim.
type audience []string

func (a *audience) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*a = audience{s}
		return nil
	}
	var arr []string
	if err := json.Unmarshal(b, &arr); err != nil {
		return err
	}
	*a = arr
	return nil
}

func containsAudience(aud audience, want string) bool {
	for _, a := range aud {
		if a == want {
			return true
		}
	}
	return false
}

// JWTAuth validates HS256-signed JSON Web Tokens from the
// "Authorization: Bearer <token>" header. It verifies the signature and
// checks the exp and nbf claims, and (when configured) the iss and aud
// claims. The "sub" claim is returned as the subject; tokens without a
// subject are rejected.
type JWTAuth struct {
	secret   []byte
	issuer   string
	audience string
}

// JWTConfig configures JWTAuth.
type JWTConfig struct {
	// Secret is the shared HMAC-SHA256 signing secret (required).
	Secret string

	// Issuer, when non-empty, is required to match the "iss" claim.
	Issuer string

	// Audience, when non-empty, must appear in the "aud" claim.
	Audience string
}

// NewJWTAuth creates a JWT authenticator, validating the configuration.
func NewJWTAuth(cfg JWTConfig) (*JWTAuth, error) {
	if cfg.Secret == "" {
		return nil, errors.New("jwt auth: secret is required")
	}
	return &JWTAuth{secret: []byte(cfg.Secret), issuer: cfg.Issuer, audience: cfg.Audience}, nil
}

// Authenticate implements Authenticator.
func (a *JWTAuth) Authenticate(r *http.Request) (string, bool) {
	token := bearerToken(r)
	if token == "" {
		return "", false
	}
	claims, err := a.verify(token)
	if err != nil || claims.Subject == "" {
		return "", false
	}
	return claims.Subject, true
}

// verify parses and validates an HS256 JWT, returning its claims.
func (a *JWTAuth) verify(token string) (*jwtClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errors.New("malformed token")
	}
	headerJSON, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return nil, errors.New("bad header encoding")
	}
	var header struct {
		Alg string `json:"alg"`
	}
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, errors.New("bad header")
	}
	if header.Alg != "HS256" {
		return nil, errors.New(`unsupported signing algorithm (want "HS256")`)
	}

	payloadJSON, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, errors.New("bad payload encoding")
	}
	sig, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, errors.New("bad signature encoding")
	}

	mac := hmac.New(sha256.New, a.secret)
	mac.Write([]byte(parts[0] + "." + parts[1]))
	if !hmac.Equal(mac.Sum(nil), sig) {
		return nil, errors.New("invalid signature")
	}

	var claims jwtClaims
	if err := json.Unmarshal(payloadJSON, &claims); err != nil {
		return nil, errors.New("bad claims")
	}

	now := time.Now().Unix()
	if claims.ExpiresAt != nil && now >= *claims.ExpiresAt {
		return nil, errors.New("token expired")
	}
	if claims.NotBefore != nil && now < *claims.NotBefore {
		return nil, errors.New("token not valid yet")
	}
	if a.issuer != "" && claims.Issuer != a.issuer {
		return nil, errors.New("invalid issuer")
	}
	if a.audience != "" && !containsAudience(claims.Audience, a.audience) {
		return nil, errors.New("invalid audience")
	}
	return &claims, nil
}

// Composite tries each authenticator in order and returns the first valid
// result.
type Composite struct {
	auths []Authenticator
}

// NewComposite creates a composite authenticator from the given
// authenticators (nil entries are skipped).
func NewComposite(auths ...Authenticator) *Composite {
	var valid []Authenticator
	for _, a := range auths {
		if a != nil {
			valid = append(valid, a)
		}
	}
	return &Composite{auths: valid}
}

// Authenticate implements Authenticator.
func (c *Composite) Authenticate(r *http.Request) (string, bool) {
	for _, a := range c.auths {
		if subject, ok := a.Authenticate(r); ok {
			return subject, true
		}
	}
	return "", false
}

// Namespaces implements NamespaceScoper: the first sub-authenticator in
// order that reports a non-empty scope for the subject determines it,
// mirroring Authenticate's first-match semantics.
func (c *Composite) Namespaces(subject string) []string {
	for _, a := range c.auths {
		if scoper, ok := a.(NamespaceScoper); ok {
			if ns := scoper.Namespaces(subject); len(ns) > 0 {
				return ns
			}
		}
	}
	return nil
}
