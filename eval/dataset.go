package eval

import (
	"encoding/json"
	"fmt"
	"os"
)

// EvalQuery is a single evaluation case: a query plus its ground truth.
type EvalQuery struct {
	// ID is a stable identifier for this case (defaults to the query text).
	ID string

	// Query is the question to evaluate.
	Query string

	// RelevantIDs are the ground-truth relevant chunk IDs (binary relevance).
	RelevantIDs []string

	// Relevance is optional graded relevance (chunk ID -> grade, higher is
	// better). When present, it is used for NDCG and overrides RelevantIDs.
	Relevance map[string]int `json:"relevance,omitempty"`

	// Context is the ground-truth supporting context, used for answer-quality
	// (faithfulness) evaluation.
	Context string `json:"context,omitempty"`

	// ReferenceAnswer is the ground-truth answer, used for correctness
	// evaluation when provided.
	ReferenceAnswer string `json:"reference_answer,omitempty"`
}

// Dataset is a named, versioned collection of evaluation queries that can be
// loaded from and saved to a JSON file.
type Dataset struct {
	// Name identifies the dataset.
	Name string

	// Version is an optional free-form version string.
	Version string `json:"version,omitempty"`

	// Queries is the ordered list of evaluation cases.
	Queries []EvalQuery
}

// NewDataset creates an empty dataset with the given name.
func NewDataset(name string) *Dataset {
	return &Dataset{Name: name}
}

// Add appends an evaluation query, filling in its ID from the query text when
// the ID is empty.
func (d *Dataset) Add(q EvalQuery) {
	if q.ID == "" {
		q.ID = q.Query
	}
	d.Queries = append(d.Queries, q)
}

// Len returns the number of queries in the dataset.
func (d *Dataset) Len() int { return len(d.Queries) }

// Save writes the dataset to a JSON file at path (pretty-printed).
func (d *Dataset) Save(path string) error {
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return fmt.Errorf("eval: marshal dataset: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("eval: write dataset: %w", err)
	}
	return nil
}

// LoadDataset reads a JSON dataset from path.
func LoadDataset(path string) (*Dataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("eval: read dataset: %w", err)
	}
	var d Dataset
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("eval: parse dataset: %w", err)
	}
	return &d, nil
}
