package loader

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeJSON(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// A mapping shaped like a plausible export: conversations at the top level,
// turns nested, role and text per turn.
var testMapping = ConversationMapping{
	Name:  "test-export",
	Turns: "messages",
	Text:  "content",
	Role:  "sender",
	ID:    "uuid",
	Title: "name",
}

const twoConversations = `[
  {"uuid":"c1","name":"First chat","messages":[
    {"sender":"human","content":"How does the ingest path work?"},
    {"sender":"assistant","content":"It dispatches on file extension."},
    {"sender":"human","content":"ok"}
  ]},
  {"uuid":"c2","name":"Second chat","messages":[
    {"sender":"human","content":"And the chunker?"},
    {"sender":"assistant","content":"document_aware splits on a boundary."}
  ]}
]`

func TestConversationLoader_OneDocumentPerConversation(t *testing.T) {
	p := writeJSON(t, "export.json", twoConversations)
	l := &ConversationLoader{Mappings: []ConversationMapping{testMapping}}

	docs, err := l.Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("got %d documents, want 2 (one per conversation)", len(docs))
	}
	if docs[0].ID != "c1" || docs[0].Title != "First chat" {
		t.Errorf("conversation identity lost: id=%q title=%q", docs[0].ID, docs[0].Title)
	}
	if got := docs[0].Metadata["turn_count"]; got == nil || got.String() != "3" {
		t.Errorf("turn_count = %v, want 3", got)
	}
	if got := docs[0].Metadata["format"]; got == nil || got.String() != "test-export" {
		t.Errorf("format = %v, want test-export", got)
	}
}

func TestConversationLoader_TurnsAreBoundarySeparatedAndRolePrefixed(t *testing.T) {
	p := writeJSON(t, "export.json", twoConversations)
	l := &ConversationLoader{Mappings: []ConversationMapping{testMapping}}

	docs, err := l.Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	parts := strings.Split(docs[0].Content, DefaultConversationBoundary)
	if len(parts) != 3 {
		t.Fatalf("conversation split into %d turns, want 3: %q", len(parts), docs[0].Content)
	}
	if !strings.HasPrefix(parts[0], "human: ") || !strings.HasPrefix(parts[1], "assistant: ") {
		t.Errorf("role prefix missing: %q / %q", parts[0], parts[1])
	}
	// The short turn is present: the loader never drops on length.
	if !strings.Contains(parts[2], "ok") {
		t.Errorf("short turn absent: %q", parts[2])
	}
}

// A turn containing the boundary would split into two chunks and invent a turn
// that was never spoken.
func TestConversationLoader_TurnContainingTheBoundaryDoesNotSplit(t *testing.T) {
	// Build via encoding/json so the boundary's newlines are escaped correctly;
	// a raw newline inside a JSON string is not valid JSON.
	raw, err := json.Marshal([]map[string]any{{
		"uuid": "c1",
		"messages": []map[string]any{
			{"sender": "human", "content": "before" + DefaultConversationBoundary + "after"},
			{"sender": "assistant", "content": "reply"},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	p := writeJSON(t, "export.json", string(raw))
	l := &ConversationLoader{Mappings: []ConversationMapping{testMapping}}

	docs, err := l.Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	parts := strings.Split(docs[0].Content, DefaultConversationBoundary)
	if len(parts) != 2 {
		t.Errorf("a turn containing the boundary produced %d turns, want 2: %q", len(parts), docs[0].Content)
	}
}

// Non-conversation JSON must still load exactly as it did before.
func TestConversationLoader_FallsThroughForPlainJSON(t *testing.T) {
	p := writeJSON(t, "plain.json", `{"content":"just a document","title":"t"}`)
	l := &ConversationLoader{Mappings: []ConversationMapping{testMapping}}

	docs, err := l.Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 || docs[0].Content != "just a document" {
		t.Fatalf("fallback did not produce the JSONLoader result: %+v", docs)
	}
	if _, ok := docs[0].Metadata["format"]; ok {
		t.Error("a fallback document was tagged with a conversation format")
	}
}

// Sniffing must reject a mapping whose paths do not resolve, rather than
// producing empty documents.
func TestConversationLoader_MappingThatDoesNotMatchIsSkipped(t *testing.T) {
	wrong := ConversationMapping{Name: "wrong", Turns: "nope", Text: "also_nope"}
	p := writeJSON(t, "export.json", twoConversations)

	l := &ConversationLoader{Mappings: []ConversationMapping{wrong, testMapping}}
	docs, err := l.Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("got %d documents, want 2: the non-matching mapping should be skipped", len(docs))
	}
	if got := docs[0].Metadata["format"]; got == nil || got.String() != "test-export" {
		t.Errorf("matched the wrong mapping: format=%v", got)
	}
}

// A mapping whose Turns path resolves but whose Text path does not must still
// be rejected: it would otherwise claim the file and emit turn-less documents.
// The Turns guard alone does not cover this.
func TestConversationLoader_MappingWithResolvingTurnsButWrongTextIsSkipped(t *testing.T) {
	halfRight := ConversationMapping{
		Name:  "half-right",
		Turns: "messages",      // resolves
		Text:  "no_such_field", // does not
		Role:  "sender",
	}
	p := writeJSON(t, "export.json", twoConversations)

	l := &ConversationLoader{Mappings: []ConversationMapping{halfRight, testMapping}}
	docs, err := l.Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := docs[0].Metadata["format"]; got == nil || got.String() != "test-export" {
		t.Errorf("matched %v, want test-export: a mapping whose Text path does not "+
			"resolve must not claim the file", got)
	}
	if !strings.Contains(docs[0].Content, "How does the ingest path work?") {
		t.Errorf("turn text missing, so the wrong mapping was used: %q", docs[0].Content)
	}
}

func TestConversationLoader_NestedPathsResolve(t *testing.T) {
	body := `{"data":{"chats":[{"id":"x1","turns":[
	  {"author":{"role":"user"},"body":{"text":"nested question"}},
	  {"author":{"role":"model"},"body":{"text":"nested answer"}}
	]}]}}`
	p := writeJSON(t, "nested.json", body)
	m := ConversationMapping{
		Name: "nested", Conversations: "data.chats", Turns: "turns",
		Text: "body.text", Role: "author.role", ID: "id",
	}
	docs, err := (&ConversationLoader{Mappings: []ConversationMapping{m}}).Load(context.Background(), p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "x1" {
		t.Fatalf("nested mapping failed: %+v", docs)
	}
	if !strings.Contains(docs[0].Content, "user: nested question") {
		t.Errorf("nested role/text not resolved: %q", docs[0].Content)
	}
}

func TestConversationLoader_NotJSONGoesToFallback(t *testing.T) {
	p := writeJSON(t, "broken.json", `{"truncated": `)
	l := &ConversationLoader{Mappings: []ConversationMapping{testMapping}}
	if _, err := l.Load(context.Background(), p); err == nil {
		t.Error("a truncated file loaded without error")
	} else if !strings.Contains(err.Error(), "broken.json") {
		t.Errorf("error does not name the file: %v", err)
	}
}
