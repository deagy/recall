package loader

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/deagy/recall/core"
)

// JSONLoader loads a JSON file whose top level is either a single object or
// an array of objects. Each object becomes one document; field names are
// addressable via dotted paths (e.g. "meta.title") to reach nested values.
type JSONLoader struct {
	// IDField selects the field used as the document ID. When empty or
	// absent, IDs are "<path>:<index>".
	IDField string

	// ContentField selects the field used as the document content. It must
	// hold a string. Default "content".
	ContentField string
}

// Load parses the JSON file at path into documents.
func (l *JSONLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("loader: read %s: %w", path, err)
	}
	var top any
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, fmt.Errorf("loader: parse %s: %w", path, err)
	}
	contentField := l.ContentField
	if contentField == "" {
		contentField = "content"
	}

	var objects []map[string]any
	switch v := top.(type) {
	case map[string]any:
		objects = []map[string]any{v}
	case []any:
		for i, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("loader: %s array item %d is not an object", path, i)
			}
			objects = append(objects, obj)
		}
	default:
		return nil, fmt.Errorf("loader: %s top level must be an object or array of objects", path)
	}
	if len(objects) == 0 {
		return nil, fmt.Errorf("loader: %s contains no objects", path)
	}

	docs := make([]*Document, 0, len(objects))
	for i, obj := range objects {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		content, err := lookupString(obj, contentField)
		if err != nil {
			return nil, fmt.Errorf("loader: %s item %d: %w", path, i, err)
		}
		id := fmt.Sprintf("%s:%d", path, i)
		if l.IDField != "" {
			if idVal, ok, err2 := lookup(obj, l.IDField); err2 == nil && ok {
				id = valueToString(idVal)
				if id == "" {
					id = fmt.Sprintf("%s:%d", path, i)
				}
			}
		}
		doc := NewDocument(id, id, path, content)
		for k, v := range obj {
			if s, ok := v.(string); ok {
				doc.Metadata[k] = core.String{Value: s}
			} else if n, ok := v.(float64); ok {
				doc.Metadata[k] = core.Number{Value: n}
			} else if b, ok := v.(bool); ok {
				doc.Metadata[k] = core.Boolean{Value: b}
			}
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

// lookup resolves a dotted path (e.g. "a.b.c") against a decoded JSON value.
// It reports ok=false when any segment is missing or a non-object segment
// blocks the path.
func lookup(v any, path string) (any, bool, error) {
	for _, part := range strings.Split(path, ".") {
		obj, isObj := v.(map[string]any)
		if !isObj {
			return nil, false, nil
		}
		next, exists := obj[part]
		if !exists {
			return nil, false, nil
		}
		v = next
	}
	return v, true, nil
}

// lookupString resolves a dotted path and requires the final value to be a string.
func lookupString(obj map[string]any, path string) (string, error) {
	v, ok, err := lookup(obj, path)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("missing field %q", path)
	}
	s, isStr := v.(string)
	if !isStr {
		return "", fmt.Errorf("field %q is not a string", path)
	}
	return s, nil
}

// valueToString renders a JSON scalar as a string for use as an ID.
func valueToString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case float64:
		return strconv.FormatFloat(s, 'g', -1, 64)
	case bool:
		return strconv.FormatBool(s)
	case nil:
		return ""
	default:
		b, err := json.Marshal(s)
		if err != nil {
			return ""
		}
		return string(b)
	}
}
