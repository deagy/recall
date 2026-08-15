// Package core provides the fundamental data types for the recall knowledge store.
package core

import (
	"strconv"
)

// Value represents a typed value that can be stored in the knowledge store.
// It supports strings, numbers, booleans, URIs, and arbitrary literals.
type Value interface {
	// Kind returns the type kind of this value.
	Kind() ValueKind
	// String returns the string representation of this value.
	String() string
}

// ValueKind represents the type of a Value.
type ValueKind int

const (
	ValueKindString ValueKind = iota
	ValueKindNumber
	ValueKindBoolean
	ValueKindURI
	ValueKindLiteral
)

// String wraps a string value.
type String struct {
	Value string
}

func (s String) Kind() ValueKind { return ValueKindString }
func (s String) String() string  { return s.Value }

// Number wraps a numeric value (float64).
type Number struct {
	Value float64
}

func (n Number) Kind() ValueKind { return ValueKindNumber }
func (n Number) String() string  { return strconv.FormatFloat(n.Value, 'g', -1, 64) }

// Boolean wraps a boolean value.
type Boolean struct {
	Value bool
}

func (b Boolean) Kind() ValueKind { return ValueKindBoolean }
func (b Boolean) String() string  { return strconv.FormatBool(b.Value) }

// URI wraps a URI/URL string value.
type URI struct {
	Value string
}

func (u URI) Kind() ValueKind { return ValueKindURI }
func (u URI) String() string  { return u.Value }

// Literal wraps an arbitrary string literal value.
type Literal struct {
	Value string
}

func (l Literal) Kind() ValueKind { return ValueKindLiteral }
func (l Literal) String() string  { return l.Value }

// ToString converts a Value to a string.
func ToString(v Value) string {
	if v == nil {
		return ""
	}
	return v.String()
}

// ToFloat64 converts a Value to float64 if possible.
func ToFloat64(v Value) (float64, bool) {
	switch val := v.(type) {
	case Number:
		return val.Value, true
	case String:
		f, err := strconv.ParseFloat(val.Value, 64)
		if err == nil {
			return f, true
		}
		return 0, false
	default:
		return 0, false
	}
}

// ToBool converts a Value to bool if possible.
func ToBool(v Value) (bool, bool) {
	switch val := v.(type) {
	case Boolean:
		return val.Value, true
	case String:
		b, err := strconv.ParseBool(val.Value)
		if err == nil {
			return b, true
		}
		return false, false
	default:
		return false, false
	}
}
