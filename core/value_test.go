package core

import (
	"testing"
)

func TestString(t *testing.T) {
	s := String{Value: "hello"}
	if s.Kind() != ValueKindString {
		t.Errorf("expected KindString, got %v", s.Kind())
	}
	if s.String() != "hello" {
		t.Errorf("expected 'hello', got %q", s.String())
	}
}

func TestNumber(t *testing.T) {
	n := Number{Value: 3.14}
	if n.Kind() != ValueKindNumber {
		t.Errorf("expected KindNumber, got %v", n.Kind())
	}
	if n.String() != "3.14" {
		t.Errorf("expected '3.14', got %q", n.String())
	}
}

func TestBoolean(t *testing.T) {
	b := Boolean{Value: true}
	if b.Kind() != ValueKindBoolean {
		t.Errorf("expected KindBoolean, got %v", b.Kind())
	}
	if b.String() != "true" {
		t.Errorf("expected 'true', got %q", b.String())
	}
}

func TestURI(t *testing.T) {
	u := URI{Value: "https://example.com"}
	if u.Kind() != ValueKindURI {
		t.Errorf("expected KindURI, got %v", u.Kind())
	}
	if u.String() != "https://example.com" {
		t.Errorf("expected 'https://example.com', got %q", u.String())
	}
}

func TestLiteral(t *testing.T) {
	l := Literal{Value: "raw text"}
	if l.Kind() != ValueKindLiteral {
		t.Errorf("expected KindLiteral, got %v", l.Kind())
	}
	if l.String() != "raw text" {
		t.Errorf("expected 'raw text', got %q", l.String())
	}
}

func TestToString(t *testing.T) {
	tests := []struct {
		name     string
		value    Value
		expected string
	}{
		{"nil", nil, ""},
		{"string", String{Value: "hi"}, "hi"},
		{"number", Number{Value: 42}, "42"},
		{"bool", Boolean{Value: true}, "true"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ToString(tt.value); got != tt.expected {
				t.Errorf("ToString() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestToFloat64(t *testing.T) {
	tests := []struct {
		name     string
		value    Value
		expected float64
		ok       bool
	}{
		{"number", Number{Value: 3.14}, 3.14, true},
		{"string_number", String{Value: "42"}, 42, true},
		{"string_text", String{Value: "hello"}, 0, false},
		{"bool", Boolean{Value: true}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ToFloat64(tt.value)
			if ok != tt.ok {
				t.Errorf("ToFloat64() ok = %v, want %v", ok, tt.ok)
			}
			if got != tt.expected {
				t.Errorf("ToFloat64() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestToBool(t *testing.T) {
	tests := []struct {
		name     string
		value    Value
		expected bool
		ok       bool
	}{
		{"bool_true", Boolean{Value: true}, true, true},
		{"bool_false", Boolean{Value: false}, false, true},
		{"string_bool", String{Value: "true"}, true, true},
		{"string_text", String{Value: "hello"}, false, false},
		{"number", Number{Value: 42}, false, false},
		{"uri", URI{Value: "https://example.com"}, false, false},
		{"literal", Literal{Value: "text"}, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ToBool(tt.value)
			if ok != tt.ok {
				t.Errorf("ToBool() ok = %v, want %v", ok, tt.ok)
			}
			if got != tt.expected {
				t.Errorf("ToBool() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestToString_AllTypes(t *testing.T) {
	tests := []struct {
		name     string
		value    Value
		expected string
	}{
		{"string", String{Value: "hello"}, "hello"},
		{"number_zero", Number{Value: 0}, "0"},
		{"number_negative", Number{Value: -3.14}, "-3.14"},
		{"number_scientific", Number{Value: 1e10}, "1e+10"},
		{"bool_true", Boolean{Value: true}, "true"},
		{"bool_false", Boolean{Value: false}, "false"},
		{"uri", URI{Value: "https://example.com"}, "https://example.com"},
		{"literal", Literal{Value: "raw text"}, "raw text"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ToString(tt.value); got != tt.expected {
				t.Errorf("ToString() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestToFloat64_AllTypes(t *testing.T) {
	tests := []struct {
		name     string
		value    Value
		expected float64
		ok       bool
	}{
		{"number", Number{Value: 3.14}, 3.14, true},
		{"number_zero", Number{Value: 0}, 0, true},
		{"number_negative", Number{Value: -42}, -42, true},
		{"string_number", String{Value: "42"}, 42, true},
		{"string_float", String{Value: "3.14"}, 3.14, true},
		{"string_empty", String{Value: ""}, 0, false},
		{"string_text", String{Value: "hello"}, 0, false},
		{"bool", Boolean{Value: true}, 0, false},
		{"uri", URI{Value: "https://example.com"}, 0, false},
		{"literal", Literal{Value: "text"}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ToFloat64(tt.value)
			if ok != tt.ok {
				t.Errorf("ToFloat64() ok = %v, want %v", ok, tt.ok)
			}
			if got != tt.expected {
				t.Errorf("ToFloat64() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestValueKind_Values(t *testing.T) {
	if ValueKindString != 0 {
		t.Errorf("expected ValueKindString=0, got %d", ValueKindString)
	}
	if ValueKindNumber != 1 {
		t.Errorf("expected ValueKindNumber=1, got %d", ValueKindNumber)
	}
	if ValueKindBoolean != 2 {
		t.Errorf("expected ValueKindBoolean=2, got %d", ValueKindBoolean)
	}
	if ValueKindURI != 3 {
		t.Errorf("expected ValueKindURI=3, got %d", ValueKindURI)
	}
	if ValueKindLiteral != 4 {
		t.Errorf("expected ValueKindLiteral=4, got %d", ValueKindLiteral)
	}
}

func TestValue_Interface(t *testing.T) {
	var v Value = String{Value: "test"}
	if v.Kind() != ValueKindString {
		t.Errorf("expected KindString, got %v", v.Kind())
	}
	if v.String() != "test" {
		t.Errorf("expected 'test', got %q", v.String())
	}

	v = Number{Value: 42}
	if v.Kind() != ValueKindNumber {
		t.Errorf("expected KindNumber, got %v", v.Kind())
	}

	v = Boolean{Value: true}
	if v.Kind() != ValueKindBoolean {
		t.Errorf("expected KindBoolean, got %v", v.Kind())
	}

	v = URI{Value: "https://example.com"}
	if v.Kind() != ValueKindURI {
		t.Errorf("expected KindURI, got %v", v.Kind())
	}

	v = Literal{Value: "raw"}
	if v.Kind() != ValueKindLiteral {
		t.Errorf("expected KindLiteral, got %v", v.Kind())
	}
}
