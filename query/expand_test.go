package query

import (
	"context"
	"errors"
	"testing"

	"github.com/deagy/recall/llm"
)

// llmMockLines returns a backend that replies with the given text.
func llmMockLines(text string) *llm.MockBackend {
	b := llm.NewMockBackend()
	b.Response = text
	return b
}

// llmMockError returns a backend that always fails.
func llmMockError() *llm.MockBackend {
	b := llm.NewMockBackend()
	b.ResponseFunc = func(ctx context.Context, req *llm.ChatRequest) (*llm.ChatResponse, error) {
		return nil, errors.New("llm down")
	}
	return b
}

func TestDetectLanguage(t *testing.T) {
	cases := []struct {
		text string
		want Language
	}{
		{"what is a vector index?", LanguageEnglish},
		{"什么是向量索引？", LanguageChinese},
		{"ベクトルインデックスとは？", LanguageJapanese},
		{"벡터 인덱스는 무엇인가요?", LanguageKorean},
		{"что такое векторный индекс?", LanguageRussian},
		{"ما هو الفهرس المتجهي؟", LanguageArabic},
		{"वेक्टर इंडेक्स क्या है?", LanguageHindi},
		{"מהו אינדקס וקטורי?", LanguageHebrew},
		{"τι είναι δείκτης διανυσμάτων;", LanguageGreek},
		{"เวกเตอร์อินเด็กซ์คืออะไร", LanguageThai},
		{"12345 !@#$", LanguageUnknown},
		{"", LanguageUnknown},
	}
	for _, c := range cases {
		if got := DetectLanguage(c.text); got != c.want {
			t.Errorf("DetectLanguage(%q) = %s, want %s", c.text, got, c.want)
		}
	}
	// Mixed: dominant script wins.
	if got := DetectLanguage("什么是向量的世界 hello"); got != LanguageChinese {
		t.Errorf("dominant script = %s", got)
	}
}

func TestLanguageName(t *testing.T) {
	if LanguageEnglish.Name() != "English" || LanguageUnknown.Name() != "unknown" {
		t.Fatal("LanguageName mapping wrong")
	}
}

type fakeTranslator struct {
	fail bool
}

func (f *fakeTranslator) Translate(_ context.Context, text string, target Language) (string, error) {
	if f.fail {
		return "", errors.New("no translation available")
	}
	return "[" + string(target) + "] " + text, nil
}

func TestMultilingual_Expand(t *testing.T) {
	m := NewMultilingual(&fakeTranslator{}, LanguageChinese, LanguageRussian)
	got, err := m.Expand(context.Background(), "how do HNSW indexes work")
	if err != nil {
		t.Fatal(err)
	}
	// English original + 2 targets.
	if len(got) != 3 {
		t.Fatalf("variants = %d", len(got))
	}
	if got[0].Language != LanguageEnglish || got[0].Text != "how do HNSW indexes work" {
		t.Fatalf("original variant wrong: %+v", got[0])
	}
	if got[1].Text != "[zh] how do HNSW indexes work" || got[2].Text != "[ru] how do HNSW indexes work" {
		t.Fatalf("translations wrong: %+v", got[1:])
	}

	// Target equal to original language is skipped.
	m2 := NewMultilingual(&fakeTranslator{}, LanguageEnglish)
	got2, err := m2.Expand(context.Background(), "english query")
	if err != nil {
		t.Fatal(err)
	}
	if len(got2) != 1 {
		t.Fatalf("duplicate-language expansion should be 1, got %d", len(got2))
	}

	// Errors: nil translator, failing translator.
	if _, err := NewMultilingual(nil).Expand(context.Background(), "q"); err == nil {
		t.Fatal("nil translator should error")
	}
	if _, err := NewMultilingual(&fakeTranslator{fail: true}, LanguageChinese).Expand(context.Background(), "q"); err == nil {
		t.Fatal("failing translator should error")
	}
}

func TestSubQueryDecompose_Heuristic(t *testing.T) {
	d := NewSubQueryDecomposer()
	ctx := context.Background()

	// Atomic question stays whole.
	got, err := d.Decompose(ctx, "What is a vector database?")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != "What is a vector database?" {
		t.Fatalf("atomic = %+v", got)
	}

	// Conjunction split.
	got, _ = d.Decompose(ctx, "How does HNSW work and how do I tune it?")
	if len(got) != 2 {
		t.Fatalf("conjunction = %+v", got)
	}

	// Multiple question marks.
	got, _ = d.Decompose(ctx, "What is recall? How does precision differ?")
	if len(got) != 2 || got[0] != "What is recall?" {
		t.Fatalf("multi-question = %+v", got)
	}

	// Empty query errors.
	if _, err := d.Decompose(ctx, "   "); err == nil {
		t.Fatal("empty query should error")
	}
}

func TestSubQueryDecompose_LLM(t *testing.T) {
	b := llmMockLines("What is recall?\nHow is it measured?\nWhat tools exist?")
	d := NewSubQueryDecomposer().WithBackend(b)
	got, err := d.Decompose(context.Background(), "Tell me everything about recall")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[1] != "How is it measured?" {
		t.Fatalf("llm decomposition = %+v", got)
	}

	// LLM error -> heuristic fallback.
	bFail := llmMockError()
	d2 := NewSubQueryDecomposer().WithBackend(bFail)
	got2, err := d2.Decompose(context.Background(), "What is A and what is B?")
	if err != nil {
		t.Fatal(err)
	}
	if len(got2) != 2 {
		t.Fatalf("fallback decomposition = %+v", got2)
	}

	// MaxSubQueries cap.
	d3 := NewSubQueryDecomposer()
	d3.MaxSubQueries = 2
	got3, _ := d3.Decompose(context.Background(), "What is A? What is B? What is C?")
	if len(got3) != 2 {
		t.Fatalf("cap not applied: %+v", got3)
	}
}
