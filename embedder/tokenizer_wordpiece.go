package embedder

import (
	_ "embed"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/deagy/recall/embedder/onnx"
)

// bertUncasedVocab is the standard 30,522-entry BERT-uncased vocabulary
// (the vocab.txt shared by all-MiniLM-L6-v2, bge-small-en-v1.5 and
// nomic-embed-text-v1.5). It is embedded so the bundled tokenizers work
// fully offline with zero network and zero CGO.
//
//go:embed vocab/bert_uncased.txt
var bertUncasedVocab string

// WordpieceConfig configures a BERT-style WordPiece tokenizer.
type WordpieceConfig struct {
	// Vocab is the vocab file content, one token per line, in id order.
	// When empty, the embedded BERT-uncased vocabulary is used.
	Vocab string

	// DoLowerCase lowercases the input before tokenization (BERT-uncased
	// models set this to true).
	DoLowerCase bool

	// MaxLength caps the number of tokens produced by Encode, preserving
	// the leading CLS and trailing SEP tokens when present.
	MaxLength int

	// PadTo pads every Encode output to this length with the PadTokenID
	// (and a 0 in the attention mask). When <= 0 the output is unpadded.
	// Most ONNX exports expect fixed-shape inputs, so callers should set
	// this to the model's maximum sequence length.
	PadTo int

	// CLSToken and SEPToken are prepended/appended around the tokens.
	// Empty disables them (e.g. for models that expect raw input).
	CLSToken string
	SEPToken string

	// PadTokenID is used for padding. When <= 0, the id of "[PAD]" in the
	// vocabulary is used.
	PadTokenID int

	// TokenTypeIDs, when non-nil, is used verbatim for the token_type_ids
	// tensor (zero-filled beyond its length). When nil, all zeros are
	// produced.
	TokenTypeIDs []int
}

// Wordpiece is a BERT-style WordPiece tokenizer (compatible with
// HuggingFace's BertTokenizer for the supported options). It is safe for
// concurrent use: all state is read-only after construction.
type Wordpiece struct {
	cfg         WordpieceConfig
	idOf        map[string]int
	padID       int
	useCLSToken bool
	useSEPToken bool
	clsID       int
	sepID       int
}

// NewWordpiece builds a tokenizer from the given config. It returns an error
// when required tokens ([CLS], [SEP], [PAD]) are missing from the vocab.
func NewWordpiece(cfg WordpieceConfig) (*Wordpiece, error) {
	vocab := cfg.Vocab
	if vocab == "" {
		vocab = bertUncasedVocab
	}
	idOf := make(map[string]int)
	for i, line := range strings.Split(vocab, "\n") {
		if line == "" {
			continue
		}
		idOf[line] = i
	}
	w := &Wordpiece{cfg: cfg, idOf: idOf}
	if id, ok := idOf["[PAD]"]; ok {
		w.padID = id
	}
	if cfg.PadTokenID > 0 {
		w.padID = cfg.PadTokenID
	}
	if cfg.CLSToken != "" {
		id, ok := idOf[cfg.CLSToken]
		if !ok {
			return nil, fmt.Errorf("embedder: wordpiece tokenizer: token %q not found in vocab", cfg.CLSToken)
		}
		w.useCLSToken, w.clsID = true, id
	}
	if cfg.SEPToken != "" {
		id, ok := idOf[cfg.SEPToken]
		if !ok {
			return nil, fmt.Errorf("embedder: wordpiece tokenizer: token %q not found in vocab", cfg.SEPToken)
		}
		w.useSEPToken, w.sepID = true, id
	}
	if cfg.PadTo > 0 && w.padID < 0 {
		return nil, fmt.Errorf("embedder: wordpiece tokenizer: padding enabled but no [PAD] token in vocab")
	}
	return w, nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (w *Wordpiece) VocabSize() int { return len(w.idOf) }

// TokenID returns the id of a vocab token, or -1 when unknown.
func (w *Wordpiece) TokenID(tok string) int {
	id, ok := w.idOf[tok]
	if !ok {
		return -1
	}
	return id
}

// preTokenize splits text into raw pieces the way HuggingFace's
// BertBasicTokenizer does for BERT-uncased models: CJK ideographs are
// space-separated, then the text is split on whitespace and on ASCII
// punctuation (each punctuation character becomes its own piece).
func preTokenize(s string) []string {
	var pieces []string
	var cur []byte
	flush := func() {
		if len(cur) > 0 {
			pieces = append(pieces, string(cur))
			cur = cur[:0]
		}
	}
	for i := 0; i < len(s); {
		if s[i] < 0x80 {
			c := s[i]
			switch {
			case c == ' ' || c == '\t' || c == '\n' || c == '\r':
				flush()
			case c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9':
				cur = append(cur, c)
			default:
				// ASCII punctuation/symbol: its own piece (BERT splits "hi-there" into hi - there).
				flush()
				pieces = append(pieces, string(c))
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if r >= 0x4E00 && r <= 0x9FFF {
			flush()
			pieces = append(pieces, s[i:i+size])
			i += size
			continue
		}
		cur = append(cur, s[i:i+size]...)
		i += size
	}
	flush()
	return pieces
}

// knownCharSet mirrors HuggingFace's "known" characters for BERT-uncased:
// ASCII alphanumerics, CJK ideographs, and the common punctuation marks the
// tokenizer keeps inside words. Everything else becomes a [U+XXXX] token.
func isKnownWordpieceChar(r rune) bool {
	if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
		return true
	}
	if r >= 0x4E00 && r <= 0x9FFF {
		return true
	}
	switch r {
	case '-', '.', '(', ')', ',', '!', '?', ':', ';', '[', ']', '{', '}', '#', '%', '$', '&', '*', '+', '<', '>', '=', '@', '|', '_', '~', '^', '`', '"', '\'', '/':
		return true
	}
	return false
}

// wordpieceTokenize converts one pre-token into token ids: unknown
// characters become [U+XXXX] tokens (or [UNK]), and the remaining runs are
// split with greedy longest-prefix WordPiece matching using "##" subwords.
func (w *Wordpiece) wordpieceTokenize(text string) []int {
	if w.cfg.DoLowerCase {
		text = strings.ToLower(text)
	}
	var out []int
	var cur []rune
	first := true
	flushRun := func() {
		if len(cur) == 0 {
			return
		}
		out = append(out, w.matchWordpiece(string(cur), first)...)
		cur = cur[:0]
		first = false
	}
	for _, r := range text {
		if isKnownWordpieceChar(r) {
			cur = append(cur, r)
			continue
		}
		flushRun()
		tok := fmt.Sprintf("[U+%04X]", r)
		if id, ok := w.idOf[tok]; ok {
			out = append(out, id)
		} else {
			out = append(out, w.unkID())
		}
	}
	flushRun()
	return out
}

// matchWordpiece greedily matches the longest vocab prefix of s, then
// continues with "##" subwords; an unmatched remainder becomes [UNK].
func (w *Wordpiece) matchWordpiece(s string, firstPiece bool) []int {
	var out []int
	start := 0
	for start < len(s) {
		prefix := ""
		if !firstPiece {
			prefix = "##"
		}
		id, end := w.longestVocabPrefix(prefix, s, start)
		if id < 0 {
			out = append(out, w.unkID())
			return out
		}
		out = append(out, id)
		start = end
		firstPiece = false
	}
	return out
}

// longestVocabPrefix returns the id of the longest prefix of s[start:] with
// the given subword prefix that exists in the vocab, and the index just past
// the matched text. Returns (-1, 0) when no candidate matches.
func (w *Wordpiece) longestVocabPrefix(subwordPrefix, s string, start int) (int, int) {
	for end := len(s); end > start; end-- {
		if id, ok := w.idOf[subwordPrefix+s[start:end]]; ok {
			return id, end
		}
	}
	return -1, 0
}

func (w *Wordpiece) unkID() int {
	if id, ok := w.idOf["[UNK]"]; ok {
		return id
	}
	return 0
}

// Encode converts text into token ids, an attention mask and token type
// ids, each of the same length (padded to cfg.PadTo when set). The CLS and
// SEP tokens (when configured) are always preserved on truncation.
func (w *Wordpiece) Encode(text string) (ids, mask, types []int) {
	ids = make([]int, 0, 8)
	if w.useCLSToken {
		ids = append(ids, w.clsID)
	}
	for _, piece := range preTokenize(text) {
		ids = append(ids, w.wordpieceTokenize(piece)...)
	}
	if w.useSEPToken {
		ids = append(ids, w.sepID)
	}
	if maxLen := w.cfg.MaxLength; maxLen > 0 && len(ids) > maxLen {
		ids = truncateTokens(ids, maxLen, w.useCLSToken, w.useSEPToken)
	}
	mask = make([]int, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	types = make([]int, len(ids))
	if tt := w.cfg.TokenTypeIDs; tt != nil {
		for i := 0; i < len(types) && i < len(tt); i++ {
			types[i] = tt[i]
		}
	}
	if padTo := w.cfg.PadTo; padTo > len(ids) {
		n := padTo - len(ids)
		ids = append(ids, repeatInt(n, w.padID)...)
		mask = append(mask, make([]int, n)...)
		types = append(types, make([]int, n)...)
	}
	return ids, mask, types
}

// truncateTokens keeps the leading CLS and trailing SEP tokens and drops
// tokens from the tail of the middle sequence until maxLen is satisfied.
func truncateTokens(ids []int, maxLen int, hasCLS, hasSEP bool) []int {
	if maxLen <= 0 || len(ids) <= maxLen {
		return ids
	}
	clsN, sepN := 0, 0
	if hasCLS {
		clsN = 1
	}
	if hasSEP {
		sepN = 1
	}
	middleKeep := maxLen - clsN - sepN
	if middleKeep < 0 {
		middleKeep = 0
	}
	out := make([]int, 0, maxLen)
	out = append(out, ids[:clsN]...)
	out = append(out, ids[clsN:clsN+middleKeep]...)
	out = append(out, ids[len(ids)-sepN:]...)
	return out
}

func repeatInt(n, v int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// FeedsForModel converts the tokenizer's Encode output into the named input
// tensors the model actually declares (input_ids, attention_mask,
// token_type_ids). This is the standard bridge from a Wordpiece tokenizer
// to an ONNX model's feed inputs.
func (w *Wordpiece) FeedsForModel(m *onnx.Model, text string) (map[string]*onnx.Tensor, error) {
	ids, mask, types := w.Encode(text)
	declared := map[string]bool{}
	for _, in := range m.FeedInputs() {
		declared[in.Name] = true
	}
	shape := []int64{1, int64(len(ids))}
	toInt64 := func(vals []int) []int64 {
		out := make([]int64, len(vals))
		for i, v := range vals {
			out[i] = int64(v)
		}
		return out
	}
	feeds := map[string]*onnx.Tensor{}
	var err error
	if declared["input_ids"] {
		feeds["input_ids"], err = onnx.NewTensor(shape, onnx.Int64, toInt64(ids))
		if err != nil {
			return nil, fmt.Errorf("embedder: wordpiece tokenizer: input_ids: %w", err)
		}
	}
	if declared["attention_mask"] {
		feeds["attention_mask"], err = onnx.NewTensor(shape, onnx.Int64, toInt64(mask))
		if err != nil {
			return nil, fmt.Errorf("embedder: wordpiece tokenizer: attention_mask: %w", err)
		}
	}
	if declared["token_type_ids"] {
		feeds["token_type_ids"], err = onnx.NewTensor(shape, onnx.Int64, toInt64(types))
		if err != nil {
			return nil, fmt.Errorf("embedder: wordpiece tokenizer: token_type_ids: %w", err)
		}
	}
	if len(feeds) == 0 {
		return nil, fmt.Errorf("embedder: wordpiece tokenizer: model declares none of input_ids/attention_mask/token_type_ids")
	}
	return feeds, nil
}

// AsTokenizerFunc returns a TokenizerFunc bound to this tokenizer and model.
func (w *Wordpiece) AsTokenizerFunc(m *onnx.Model) TokenizerFunc {
	return func(text string) (map[string]*onnx.Tensor, error) {
		return w.FeedsForModel(m, text)
	}
}

// knownModels maps model names to their bundled tokenizer constructors.
var knownModels = map[string]func(*onnx.Model) (TokenizerFunc, error){
	"all-MiniLM-L6-v2": func(m *onnx.Model) (TokenizerFunc, error) {
		w, err := NewWordpiece(WordpieceConfig{
			DoLowerCase: true,
			MaxLength:   512,
			PadTo:       512,
			CLSToken:    "[CLS]",
			SEPToken:    "[SEP]",
		})
		if err != nil {
			return nil, err
		}
		return w.AsTokenizerFunc(m), nil
	},
	"bge-small-en-v1.5": func(m *onnx.Model) (TokenizerFunc, error) {
		w, err := NewWordpiece(WordpieceConfig{
			DoLowerCase: true,
			MaxLength:   512,
			PadTo:       512,
			CLSToken:    "[CLS]",
			SEPToken:    "[SEP]",
		})
		if err != nil {
			return nil, err
		}
		return w.AsTokenizerFunc(m), nil
	},
	"nomic-embed-text-v1.5": func(m *onnx.Model) (TokenizerFunc, error) {
		w, err := NewWordpiece(WordpieceConfig{
			DoLowerCase: true,
			MaxLength:   2048,
			PadTo:       2048,
			CLSToken:    "[CLS]",
			SEPToken:    "[SEP]",
		})
		if err != nil {
			return nil, err
		}
		return w.AsTokenizerFunc(m), nil
	},
}

// BundledTokenizerNames returns the model names that ship with a bundled
// tokenizer, in sorted order.
func BundledTokenizerNames() []string {
	names := make([]string, 0, len(knownModels))
	for name := range knownModels {
		names = append(names, name)
	}
	sortStrings(names)
	return names
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// BundledTokenizer returns a ready-to-use TokenizerFunc for a known ONNX
// model name. The function tokenizes text with the model's exact
// configuration (lowercasing, max length, special tokens) and emits only
// the input tensors the model declares, padded to the model's maximum
// sequence length. The vocab is embedded, so this works fully offline.
func BundledTokenizer(modelName string, m *onnx.Model) (TokenizerFunc, error) {
	fn, ok := knownModels[modelName]
	if !ok {
		return nil, fmt.Errorf("embedder: no bundled tokenizer for model %q (known: %v)", modelName, BundledTokenizerNames())
	}
	return fn(m)
}
