package query

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"github.com/deagy/recall/llm"
)

// Language identifies a supported query language.
type Language string

// Supported languages. Unknown text maps to LanguageUnknown.
const (
	LanguageEnglish  Language = "en"
	LanguageChinese  Language = "zh"
	LanguageJapanese Language = "ja"
	LanguageKorean   Language = "ko"
	LanguageRussian  Language = "ru"
	LanguageArabic   Language = "ar"
	LanguageHindi    Language = "hi"
	LanguageHebrew   Language = "he"
	LanguageGreek    Language = "el"
	LanguageThai     Language = "th"
	LanguageUnknown  Language = "unknown"
)

// LanguageName is the display name for prompting.
func (l Language) Name() string {
	switch l {
	case LanguageEnglish:
		return "English"
	case LanguageChinese:
		return "Chinese (Simplified)"
	case LanguageJapanese:
		return "Japanese"
	case LanguageKorean:
		return "Korean"
	case LanguageRussian:
		return "Russian"
	case LanguageArabic:
		return "Arabic"
	case LanguageHindi:
		return "Hindi"
	case LanguageHebrew:
		return "Hebrew"
	case LanguageGreek:
		return "Greek"
	case LanguageThai:
		return "Thai"
	default:
		return "unknown"
	}
}

// DetectLanguage makes a fast, dependency-free language guess from the
// dominant writing system of the text. Latin script maps to English
// (the default corpus language); callers wanting finer Latin-script
// distinction should use an LLM-backed detector.
func DetectLanguage(text string) Language {
	counts := map[Language]int{}
	for _, r := range text {
		switch {
		case r >= 0x4E00 && r <= 0x9FFF:
			counts[LanguageChinese]++
		case (r >= 0x3040 && r <= 0x309F) || (r >= 0x30A0 && r <= 0x30FF):
			counts[LanguageJapanese]++
		case r >= 0xAC00 && r <= 0xD7AF:
			counts[LanguageKorean]++
		case r >= 0x0400 && r <= 0x04FF:
			counts[LanguageRussian]++
		case r >= 0x0600 && r <= 0x06FF:
			counts[LanguageArabic]++
		case r >= 0x0900 && r <= 0x097F:
			counts[LanguageHindi]++
		case r >= 0x0590 && r <= 0x05FF:
			counts[LanguageHebrew]++
		case r >= 0x0370 && r <= 0x03FF:
			counts[LanguageGreek]++
		case r >= 0x0E00 && r <= 0x0E7F:
			counts[LanguageThai]++
		case unicode.IsLetter(r):
			counts[LanguageEnglish]++ // any other script (Latin, Cyrillic-adjacent, ...)
		}
	}
	best, bestN := LanguageUnknown, 0
	for lang, n := range counts {
		if n > bestN {
			best, bestN = lang, n
		}
	}
	return best
}

// Translator translates text into a target language.
type Translator interface {
	// Translate returns text rendered into target.
	Translate(ctx context.Context, text string, target Language) (string, error)
}

// LLMTranslator implements Translator using an LLM backend.
type LLMTranslator struct {
	Backend llm.Backend
}

// Translate translates text into the target language via the LLM.
func (t *LLMTranslator) Translate(ctx context.Context, text string, target Language) (string, error) {
	prompt := fmt.Sprintf("Translate the following text into %s. Reply with the translation only, no commentary: %s", target.Name(), text)
	out, err := chatSystemUser(ctx, t.Backend, "You are a precise translator.", prompt)
	if err != nil {
		return "", err
	}
	if i := strings.IndexByte(out, '\n'); i >= 0 {
		out = strings.TrimSpace(out[:i])
	}
	return out, nil
}

// ExpandedQuery is one language variant of a query used for
// multilingual (multi-query) retrieval.
type ExpandedQuery struct {
	// Language is the variant's language.
	Language Language

	// Text is the query in that language.
	Text string
}

// Multilingual expands a query into the target languages for
// multi-query retrieval: each variant is searched independently and the
// result lists are merged upstream (e.g. via fuse or union-top-k).
type Multilingual struct {
	// Translator renders the variants. Required.
	Translator Translator

	// TargetLanguages to expand into. The original language is always
	// included first even if listed here.
	TargetLanguages []Language
}

// NewMultilingual creates a Multilingual expander with the given
// translator and target languages.
func NewMultilingual(tr Translator, targets ...Language) *Multilingual {
	return &Multilingual{Translator: tr, TargetLanguages: targets}
}

// Expand returns the query variants: the original (in its detected
// language) followed by one translation per target language, with
// duplicates (target == original language) dropped.
func (m *Multilingual) Expand(ctx context.Context, query string) ([]ExpandedQuery, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m.Translator == nil {
		return nil, fmt.Errorf("query: multilingual translator is required")
	}
	original := DetectLanguage(query)
	out := []ExpandedQuery{{Language: original, Text: strings.TrimSpace(query)}}
	for _, target := range m.TargetLanguages {
		if target == original {
			continue
		}
		text, err := m.Translator.Translate(ctx, query, target)
		if err != nil {
			return nil, fmt.Errorf("translate to %s: %w", target, err)
		}
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("translate to %s: empty result", target)
		}
		out = append(out, ExpandedQuery{Language: target, Text: strings.TrimSpace(text)})
	}
	return out, nil
}
