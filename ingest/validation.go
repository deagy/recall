package ingest

import (
	"fmt"
	"strings"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
)

// Schema describes the structural rules a document must satisfy before it
// is ingested. All fields are optional; zero values mean "no constraint".
type Schema struct {
	// MinContent is the minimum content length in bytes; 0 allows empty
	// content to pass this check (empty content is still rejected by the
	// store on upload).
	MinContent int

	// MaxContent is the maximum content length in bytes; 0 means unlimited.
	MaxContent int

	// RequiredMetadata lists metadata keys that must be present and
	// non-empty.
	RequiredMetadata []string

	// AllowedSources lists accepted source prefixes; empty means any
	// source is accepted.
	AllowedSources []string
}

// Validator checks documents against a Schema.
type Validator struct {
	Schema Schema
}

// Validate returns a descriptive error if the document violates the schema.
func (v *Validator) Validate(d *loader.Document) error {
	if d == nil {
		return fmt.Errorf("ingest: nil document")
	}
	var problems []string
	if len(d.Content) < v.Schema.MinContent {
		problems = append(problems, fmt.Sprintf("content length %d < min %d", len(d.Content), v.Schema.MinContent))
	}
	if v.Schema.MaxContent > 0 && len(d.Content) > v.Schema.MaxContent {
		problems = append(problems, fmt.Sprintf("content length %d > max %d", len(d.Content), v.Schema.MaxContent))
	}
	for _, key := range v.Schema.RequiredMetadata {
		val, ok := d.Metadata[key]
		if !ok || emptyValue(val) {
			problems = append(problems, "missing required metadata "+key)
		}
	}
	if len(v.Schema.AllowedSources) > 0 && !hasAllowedSource(d.Source, v.Schema.AllowedSources) {
		problems = append(problems, "source "+d.Source+" not in allowed prefixes")
	}
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("ingest: document %s invalid: %s", d.ID, strings.Join(problems, "; "))
}

// emptyValue reports whether a metadata value is effectively empty.
func emptyValue(val core.Value) bool {
	if val == nil {
		return true
	}
	switch val.Kind() {
	case core.ValueKindString, core.ValueKindURI, core.ValueKindLiteral:
		return strings.TrimSpace(val.String()) == ""
	default:
		return false
	}
}

// hasAllowedSource reports whether source starts with any allowed prefix.
func hasAllowedSource(source string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(source, p) {
			return true
		}
	}
	return false
}
