package ingest

import (
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

func strValue(s string) core.Value { return core.String{Value: s} }

func docWith(id, source, content string, meta map[string]string) *loader.Document {
	d := loader.NewDocument(id, id, source, content)
	for k, v := range meta {
		d.Metadata[k] = strValue(v)
	}
	return d
}

func TestValidator_AllPass(t *testing.T) {
	v := &Validator{Schema: Schema{
		MinContent:       5,
		MaxContent:       100,
		RequiredMetadata: []string{"author"},
		AllowedSources:   []string{"file://"},
	}}
	if err := v.Validate(docWith("x", "file://a", "hello world", map[string]string{"author": "me"})); err != nil {
		t.Errorf("should pass: %v", err)
	}
}

func TestValidator_EachRule(t *testing.T) {
	cases := []struct {
		name string
		sch  Schema
		doc  *loader.Document
	}{
		{"min", Schema{MinContent: 100}, docWith("x", "s", "short", nil)},
		{"max", Schema{MaxContent: 3}, docWith("x", "s", "way too long", nil)},
		{"meta", Schema{RequiredMetadata: []string{"author"}}, docWith("x", "s", "body", nil)},
		{"meta-empty", Schema{RequiredMetadata: []string{"author"}}, docWith("x", "s", "body", map[string]string{"author": "  "})},
		{"source", Schema{AllowedSources: []string{"file://"}}, docWith("x", "http://nope", "body", nil)},
	}
	for _, c := range cases {
		v := &Validator{Schema: c.sch}
		if err := v.Validate(c.doc); err == nil {
			t.Errorf("%s: expected validation error", c.name)
		}
	}
}

func TestValidator_NilDoc(t *testing.T) {
	if err := (&Validator{}).Validate(nil); err == nil {
		t.Error("expected nil doc error")
	}
}
