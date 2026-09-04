package loader

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/deagy/recall/core"
)

// DefaultConversationBoundary separates turns inside a conversation document.
//
// It must match store.chunking.boundary, because the boundary is how turn
// structure survives from here to the chunker: this loader emits one document
// per conversation and cannot express chunk edges itself. A document_aware
// chunker splitting on a different marker sees one undifferentiated blob.
const DefaultConversationBoundary = "\n␄\n" // U+2404 END OF TRANSMISSION

// ConversationMapping locates conversations and their turns inside a chat
// export. Paths are dot-separated and resolved with the same lookup the JSON
// loader uses, so "chat_messages.text" reaches a nested field.
//
// A built-in format is only a mapping literal; adding one needs no new code.
type ConversationMapping struct {
	// Name identifies the format and is recorded on every document.
	Name string

	// Conversations is the path to the array of conversations. Empty means the
	// top level is itself the array, or a single conversation object.
	Conversations string

	// Turns is the path, within one conversation, to its array of turns.
	// Required: a mapping without it cannot describe a conversation.
	Turns string

	// Text is the path, within one turn, to the turn's text. Required.
	Text string

	// Role is the path, within one turn, to the speaker. Optional; when it
	// resolves, each turn is prefixed with "<role>: " so the speaker survives
	// into the chunk. Per-turn metadata has nowhere else to live -- the
	// document is the conversation, and chunk metadata comes from the chunker.
	Role string

	// ID and Title are paths within one conversation. Optional.
	ID    string
	Title string
}

// BuiltinConversationMappings are the export formats recall recognises without
// configuration.
//
// It is deliberately empty. A mapping is only correct if it was written against
// a real export, and no built-in format has been verified against one yet --
// adding a guessed Claude or ChatGPT mapping here would ship a parser that
// matches nothing and silently falls through to JSONLoader, which is worse than
// having none. Add a literal here once an export has actually been read; the
// machinery below needs no other change.
func BuiltinConversationMappings() []ConversationMapping { return nil }

// ConversationLoader turns a chat export into one document per conversation,
// with turns joined by Boundary. Mappings are tried in order and the first that
// matches the file wins; when none matches, Fallback handles the file, so a
// plain JSON document still loads the way it always did.
type ConversationLoader struct {
	// Mappings are tried in order. A mapping matches when its Turns and Text
	// paths resolve on the file's first conversation.
	Mappings []ConversationMapping

	// Boundary separates turns. Empty means DefaultConversationBoundary.
	Boundary string

	// Fallback loads files no mapping matches. Empty means &JSONLoader{}.
	Fallback Loader
}

// Load parses path, dispatching to the first mapping that matches it.
func (l *ConversationLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("loader: read %s: %w", path, err)
	}
	var top any
	if err := json.Unmarshal(data, &top); err != nil {
		// Not JSON at all: this is the fallback's problem to report, not a
		// conversation-parsing failure.
		return l.fallback().Load(ctx, path)
	}

	for _, m := range l.Mappings {
		convs, ok := m.conversations(top)
		if !ok || len(convs) == 0 {
			continue
		}
		if !m.matches(convs[0]) {
			continue
		}
		return l.build(ctx, path, m, convs)
	}
	return l.fallback().Load(ctx, path)
}

func (l *ConversationLoader) fallback() Loader {
	if l.Fallback != nil {
		return l.Fallback
	}
	return &JSONLoader{}
}

func (l *ConversationLoader) boundary() string {
	if l.Boundary != "" {
		return l.Boundary
	}
	return DefaultConversationBoundary
}

// conversations resolves the mapping's conversation array out of a parsed file.
func (m ConversationMapping) conversations(top any) ([]any, bool) {
	v := top
	if m.Conversations != "" {
		found, ok, err := lookup(top, m.Conversations)
		if err != nil || !ok {
			return nil, false
		}
		v = found
	}
	switch t := v.(type) {
	case []any:
		return t, true
	case map[string]any:
		return []any{t}, true
	default:
		return nil, false
	}
}

// matches reports whether this mapping can describe conv: its required paths
// resolve, and the first turn actually carries text. Shape, not guesswork.
func (m ConversationMapping) matches(conv any) bool {
	if m.Turns == "" || m.Text == "" {
		return false
	}
	turns, ok := m.turns(conv)
	if !ok || len(turns) == 0 {
		return false
	}
	text, ok := lookupStringPath(turns[0], m.Text)
	return ok && text != ""
}

func (m ConversationMapping) turns(conv any) ([]any, bool) {
	v, ok, err := lookup(conv, m.Turns)
	if err != nil || !ok {
		return nil, false
	}
	arr, ok := v.([]any)
	return arr, ok
}

// build emits one document per conversation.
func (l *ConversationLoader) build(ctx context.Context, path string, m ConversationMapping, convs []any) ([]*Document, error) {
	boundary := l.boundary()
	docs := make([]*Document, 0, len(convs))

	for i, conv := range convs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		turns, ok := m.turns(conv)
		if !ok || len(turns) == 0 {
			continue // a conversation with no turns indexes nothing
		}

		parts := make([]string, 0, len(turns))
		for _, turn := range turns {
			text, ok := lookupStringPath(turn, m.Text)
			if !ok || strings.TrimSpace(text) == "" {
				continue
			}
			// A turn containing the boundary would split into two chunks and
			// invent a turn that was never spoken.
			text = strings.ReplaceAll(text, boundary, " ")
			if m.Role != "" {
				if role, ok := lookupStringPath(turn, m.Role); ok && role != "" {
					text = role + ": " + text
				}
			}
			parts = append(parts, text)
		}
		if len(parts) == 0 {
			continue
		}

		id := fmt.Sprintf("%s:%d", path, i)
		if m.ID != "" {
			if v, ok := lookupStringPath(conv, m.ID); ok && v != "" {
				id = v
			}
		}
		title := id
		if m.Title != "" {
			if v, ok := lookupStringPath(conv, m.Title); ok && v != "" {
				title = v
			}
		}

		doc := NewDocument(id, title, path, strings.Join(parts, boundary))
		doc.Metadata["format"] = core.String{Value: m.Name}
		doc.Metadata["turn_count"] = core.Number{Value: float64(len(parts))}
		docs = append(docs, doc)
	}

	if len(docs) == 0 {
		return nil, fmt.Errorf("loader: %s matched format %q but produced no conversations with turns", path, m.Name)
	}
	return docs, nil
}

// lookupStringPath resolves a dot-path and renders the result as a string.
func lookupStringPath(v any, path string) (string, bool) {
	found, ok, err := lookup(v, path)
	if err != nil || !ok {
		return "", false
	}
	s := valueToString(found)
	return s, s != ""
}
