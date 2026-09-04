package app

import (
	"strings"
	"testing"

	"github.com/deagy/recall/chunker"
	"github.com/deagy/recall/config"
	"github.com/deagy/recall/core"
)

// A chat export is ingested as one document per conversation with turns
// separated by the document_aware boundary, so every turn should become at
// least one chunk. Real conversations are full of short turns -- "yes",
// "do that", a one-line correction -- and those are disproportionately where
// decisions are recorded.
//
// Before document_aware was exposed, the only way to compose it was over a
// chunker built from chunker.DefaultConfig(), whose MinChunkSize is 50
// characters. fixed.go discards a single-part chunk below that, and
// document_aware chunks each section independently, so three of the five turns
// below vanished from the index with no error.
func TestChunkerFactory_DocumentAwareKeepsShortTurns(t *testing.T) {
	turns := []struct {
		name   string
		marker string
		text   string
	}{
		{"long question", "zeta", "Can you look at the ingest path end to end and tell me what it actually does, including which repository owns it? zeta"},
		{"short confirm", "yankee", "Yes, do that. yankee"},
		{"short decision", "xray", "One document per conversation. xray"},
		{"long answer", "whiskey", "cadre delegates ingest to recall upload, which dispatches on file extension and has no conversation concept at all. whiskey"},
		{"very short ack", "victor", "ok victor"},
	}

	cc := config.ChunkingConfig{Strategy: config.ChunkingDocumentAware}
	cfg := &config.Config{Store: config.StoreConfig{Chunking: cc}}
	cfg.WithDefaults()

	if got := cfg.Store.Chunking.MinChunkSize; got != 0 {
		t.Fatalf("document_aware MinChunkSize default = %d, want 0: a boundary-split "+
			"strategy must not discard whole sections", got)
	}

	texts := make([]string, 0, len(turns))
	for _, tn := range turns {
		texts = append(texts, tn.text)
	}
	content := strings.Join(texts, chunker.DefaultBoundary)

	ch := ChunkerFactory(cfg.Store.Chunking)(chunker.DefaultConfig())
	chunks, err := ch.Chunk(&core.Document{ID: "conversation-1"}, content)
	if err != nil {
		t.Fatalf("chunking the conversation: %v", err)
	}

	var missing []string
	for _, tn := range turns {
		found := false
		for _, c := range chunks {
			if strings.Contains(c.Content, tn.marker) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, tn.name)
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d of %d turns produced no chunk and are absent from the index: %s",
			len(missing), len(turns), strings.Join(missing, ", "))
	}
}

// The stores invoke their factory with chunker.DefaultConfig(), so a factory
// that read its argument would silently discard everything the operator
// configured. Guard the values actually taking effect.
func TestChunkerFactory_UsesConfiguredValuesNotTheArgument(t *testing.T) {
	cc := config.ChunkingConfig{
		Strategy:     config.ChunkingFixed,
		MaxTokens:    24,
		Overlap:      0,
		MinChunkSize: 0,
	}

	// Handed the package defaults, exactly as store/memory.go and
	// store/sqlite.go do.
	ch := ChunkerFactory(cc)(chunker.DefaultConfig())

	var b strings.Builder
	for i := 0; i < 40; i++ {
		b.WriteString("alpha bravo charlie delta echo foxtrot golf hotel. ")
	}
	chunks, err := ch.Chunk(&core.Document{ID: "sized"}, b.String())
	if err != nil {
		t.Fatalf("chunking: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks produced")
	}

	// With MaxTokens 24 the pieces must be far smaller than DefaultConfig's
	// 512 would produce. Compare against the same content chunked at the
	// default, which is what the bug produced regardless of configuration.
	atDefault, err := chunker.NewFixed(chunker.DefaultConfig()).Chunk(&core.Document{ID: "sized"}, b.String())
	if err != nil {
		t.Fatalf("chunking at defaults: %v", err)
	}
	if len(chunks) <= len(atDefault) {
		t.Errorf("configured MaxTokens=24 produced %d chunks, defaults produced %d: "+
			"the configured value did not reach the chunker", len(chunks), len(atDefault))
	}
}
