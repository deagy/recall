package embedder

import (
	"testing"

	"github.com/deagy/recall/embedder/onnx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestWordpiece(t *testing.T) *Wordpiece {
	t.Helper()
	w, err := NewWordpiece(WordpieceConfig{
		DoLowerCase: true,
		CLSToken:    "[CLS]",
		SEPToken:    "[SEP]",
	})
	require.NoError(t, err)
	return w
}

func TestWordpiece_Encode_BasicTokens(t *testing.T) {
	w := newTestWordpiece(t)
	ids, mask, types := w.Encode("Do not do this")
	assert.Equal(t, []int{101, 2079, 2025, 2079, 2023, 102}, ids)
	assert.Equal(t, []int{1, 1, 1, 1, 1, 1}, mask)
	assert.Equal(t, []int{0, 0, 0, 0, 0, 0}, types)
}

func TestWordpiece_Encode_Subwords(t *testing.T) {
	w := newTestWordpiece(t)
	ids, _, _ := w.Encode("tokenization")
	// "token" + "##ization" (longest-prefix greedy)
	assert.Equal(t, []int{101, 19204, 3989, 102}, ids)
}

func TestWordpiece_Encode_Punctuation(t *testing.T) {
	w := newTestWordpiece(t)
	cases := []struct {
		in  string
		ids []int
	}{
		{"hello world!", []int{101, 7592, 2088, 999, 102}},
		{"hello, world", []int{101, 7592, 1010, 2088, 102}},
		{"a  b", []int{101, 1037, 1038, 102}},
		{"foo_bar", []int{101, 29379, 1035, 3347, 102}},
		{"hello-world", []int{101, 7592, 1011, 2088, 102}},
		{"The quick brown fox", []int{101, 1996, 4248, 2829, 4419, 102}},
	}
	for _, c := range cases {
		ids, _, _ := w.Encode(c.in)
		assert.Equal(t, c.ids, ids, "input %q", c.in)
	}
}

func TestWordpiece_Encode_LowerCasing(t *testing.T) {
	w := newTestWordpiece(t)
	upper, _, _ := w.Encode("WORLD")
	lower, _, _ := w.Encode("world")
	assert.Equal(t, lower, upper)
}

func TestWordpiece_NoLowerCase(t *testing.T) {
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: false})
	require.NoError(t, err)
	lowerW := newTestWordpiece(t)
	upper, _, _ := w.Encode("World")
	lower, _, _ := lowerW.Encode("World")
	assert.NotEqual(t, lower, upper, "case-sensitive tokenizer must keep case")
}

func TestWordpiece_UnknownToken(t *testing.T) {
	w := newTestWordpiece(t)
	// A lone character not present in the BERT-uncased vocab and with no
	// [U+XXXX] entry must fall back to [UNK] (100).
	ids, _, _ := w.Encode("€")
	assert.Equal(t, []int{101, 100, 102}, ids)
}

func TestWordpiece_UnknownCharacter(t *testing.T) {
	w := newTestWordpiece(t)
	// "€" (U+20AC) is not a known wordpiece char; it should map to [U+20AC]
	// if in the vocab, else [UNK]. "world" is in the vocab.
	ids, _, _ := w.Encode("€ world")
	assert.Equal(t, []int{101, 100, 2088, 102}, ids)
}

func TestWordpiece_Padding(t *testing.T) {
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: true, CLSToken: "[CLS]", SEPToken: "[SEP]", PadTo: 8})
	require.NoError(t, err)
	ids, mask, types := w.Encode("Do")
	assert.Len(t, ids, 8)
	assert.Equal(t, []int{101, 2079, 102, 0, 0, 0, 0, 0}, ids)
	assert.Equal(t, []int{1, 1, 1, 0, 0, 0, 0, 0}, mask)
	assert.Equal(t, []int{0, 0, 0, 0, 0, 0, 0, 0}, types)
}

func TestWordpiece_Truncation(t *testing.T) {
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: true, CLSToken: "[CLS]", SEPToken: "[SEP]", MaxLength: 5})
	require.NoError(t, err)
	// "the quick brown fox jumps" -> many tokens, truncated to 5 keeping CLS+SEP.
	ids, mask, _ := w.Encode("the quick brown fox jumps over")
	assert.Len(t, ids, 5)
	assert.Equal(t, 101, ids[0])
	assert.Equal(t, 102, ids[4])
	assert.Equal(t, []int{1, 1, 1, 1, 1}, mask)
}

func TestWordpiece_Truncation_KeepsCLSSEP(t *testing.T) {
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: true, CLSToken: "[CLS]", SEPToken: "[SEP]", MaxLength: 3})
	require.NoError(t, err)
	ids, _, _ := w.Encode("the quick brown fox")
	assert.Equal(t, []int{101, 1996, 102}, ids)
}

func TestWordpiece_VocabSize(t *testing.T) {
	w := newTestWordpiece(t)
	assert.Equal(t, 30522, w.VocabSize())
}

func TestWordpiece_TokenID(t *testing.T) {
	w := newTestWordpiece(t)
	assert.Equal(t, 2088, w.TokenID("world"))
	assert.Equal(t, -1, w.TokenID("notarealtoken12345"))
}

func TestWordpiece_NoSpecialTokens(t *testing.T) {
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: true})
	require.NoError(t, err)
	ids, _, _ := w.Encode("hello")
	assert.Equal(t, []int{7592}, ids)
}

func TestWordpiece_FeedsForModel(t *testing.T) {
	m := buildMiniStModel(t)
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: true, PadTo: testSeqLen})
	require.NoError(t, err)
	feeds, err := w.FeedsForModel(m, "Do not do this")
	require.NoError(t, err)
	ids, ok := feeds["input_ids"]
	require.True(t, ok)
	assert.Equal(t, []int64{1, testSeqLen}, ids.Shape)
	mask, ok := feeds["attention_mask"]
	require.True(t, ok)
	assert.Equal(t, []int64{1, testSeqLen}, mask.Shape)
	// input_ids should be int64
	ids64, err := ids.AsInt64()
	require.NoError(t, err)
	// No CLS/SEP in this config, so the first token is the first word "do".
	assert.Equal(t, int64(w.TokenID("do")), ids64[0])
}

func TestWordpiece_FeedsForModel_NoMatchingInputs(t *testing.T) {
	// Build a model that declares none of the standard inputs.
	nodes := []onnx.NodeSpec{{Op: "Identity", Inputs: []string{"x"}, Outputs: []string{"y"}}}
	b := onnx.Encode(nodes, nil, []onnx.NamedType{{Name: "x", Dtype: onnx.Float32}}, []onnx.NamedType{{Name: "y", Dtype: onnx.Float32}})
	m, err := onnx.Load(b)
	require.NoError(t, err)
	w := newTestWordpiece(t)
	_, err = w.FeedsForModel(m, "hello")
	assert.Error(t, err)
}

func TestBundledTokenizerNames(t *testing.T) {
	names := BundledTokenizerNames()
	assert.ElementsMatch(t, []string{"all-MiniLM-L6-v2", "bge-small-en-v1.5", "nomic-embed-text-v1.5"}, names)
}

func TestBundledTokenizer_UnknownModel(t *testing.T) {
	_, err := BundledTokenizer("no-such-model", nil)
	assert.Error(t, err)
}

func TestBundledTokenizer_MiniLM(t *testing.T) {
	m := buildMiniStModel(t)
	fn, err := BundledTokenizer("all-MiniLM-L6-v2", m)
	require.NoError(t, err)
	feeds, err := fn("Do not do this")
	require.NoError(t, err)
	ids, ok := feeds["input_ids"]
	require.True(t, ok)
	ids64, err := ids.AsInt64()
	require.NoError(t, err)
	assert.Equal(t, int64(101), ids64[0])
	// MiniLM pads to 512
	assert.Equal(t, int64(512), ids.Shape[1])
}

func TestBundledTokenizer_BGE(t *testing.T) {
	m := buildMiniStModel(t)
	fn, err := BundledTokenizer("bge-small-en-v1.5", m)
	require.NoError(t, err)
	feeds, err := fn("query text")
	require.NoError(t, err)
	ids, ok := feeds["input_ids"]
	require.True(t, ok)
	assert.Equal(t, int64(512), ids.Shape[1])
}

func TestBundledTokenizer_Nomic(t *testing.T) {
	m := buildMiniStModel(t)
	fn, err := BundledTokenizer("nomic-embed-text-v1.5", m)
	require.NoError(t, err)
	feeds, err := fn("nomic text")
	require.NoError(t, err)
	ids, ok := feeds["input_ids"]
	require.True(t, ok)
	assert.Equal(t, int64(2048), ids.Shape[1])
}

func TestWordpiece_EmptyText(t *testing.T) {
	w := newTestWordpiece(t)
	ids, mask, _ := w.Encode("")
	assert.Equal(t, []int{101, 102}, ids)
	assert.Equal(t, []int{1, 1}, mask)
}

func TestWordpiece_ChineseChars(t *testing.T) {
	w := newTestWordpiece(t)
	// CJK ideographs are space-separated by the basic tokenizer.
	ids, _, _ := w.Encode("中文")
	// Both characters should be present as separate tokens (likely [UNK] if not in vocab).
	assert.GreaterOrEqual(t, len(ids), 4) // CLS + >=2 + SEP
}

func BenchmarkWordpiece_Encode(b *testing.B) {
	w, err := NewWordpiece(WordpieceConfig{DoLowerCase: true, CLSToken: "[CLS]", SEPToken: "[SEP]"})
	if err != nil {
		b.Fatal(err)
	}
	texts := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Recall is a Go library for building RAG applications.",
		"Retrieval-Augmented Generation combines retrieval with generation for grounded answers.",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.Encode(texts[i%len(texts)])
	}
}
