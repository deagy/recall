// Package govern wraps a recall store in a fail-closed retrieval interface.
//
// recall's own Search is permissive by design: filters live in SearchOptions,
// a caller may pass none, and search spans every namespace in the store. That
// is right for a retrieval library. It is wrong for a system where a retrieval
// is an access decision — there, a caller who says nothing should be refused,
// not served everything.
//
// This package supplies the missing default. It refuses a request that has not
// decided its scope, has not declared a classification, or cannot be recorded,
// and it refuses before touching the store: an interface that only refuses
// after opening a connection has already revealed that the caller asked.
//
// # What it does not decide
//
// Classification is an opaque, required string. This package does not know
// what values exist, which dominate which, or who may see what — only that a
// caller must state one. Source names are likewise opaque. The policy belongs
// to the system embedding this; what belongs here is that no policy can be
// skipped by omission.
//
// The refusals are ported from a store that enforced them in production, and
// each one is there because its absence was a real hazard rather than a
// theoretical one. Their reasoning is recorded per-refusal below.
package govern

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/deagy/recall/index"
)

// The refusals, as sentinel errors so a caller can distinguish a governance
// refusal from a retrieval failure.
var (
	// ErrNoQuery: a retrieval with no question is not a retrieval, and
	// refusing early keeps the audit record meaningful.
	ErrNoQuery = errors.New("govern: query is required")

	// ErrNoClassification: classification is the access decision. Without one
	// the interface cannot know what the caller may see, and any default would
	// be a policy nobody wrote.
	ErrNoClassification = errors.New("govern: classification is required")

	// ErrNoScope: the central refusal. A caller must choose between naming
	// sources and deliberately spanning all of them. Silence would default to
	// reading everything, which is the behaviour this package exists to
	// prevent.
	ErrNoScope = errors.New("govern: source scope is required")

	// ErrAmbiguousScope: naming sources *and* asking for all of them is
	// ambiguous rather than resolvable. Picking either reading would silently
	// widen or narrow what the caller sees.
	ErrAmbiguousScope = errors.New("govern: source scope is ambiguous")

	// ErrBlankSource: a blank entry is a scope nobody chose. Treating it as a
	// wildcard or dropping it would both be guesses.
	ErrBlankSource = errors.New("govern: source filter entries must be non-empty")

	// ErrNoRecorder: retrieval is recorded, so a request that cannot be
	// recorded is refused rather than served unrecorded.
	ErrNoRecorder = errors.New("govern: an audit recorder is required")

	// ErrNoEmbedderIdentity: a recorded retrieval must say what produced the
	// vectors it searched, or it cannot be reproduced later. The store this
	// was ported from refused a search with no embedding provider for the same
	// reason; recall injects its embedder at construction, so the identity is
	// required there instead.
	ErrNoEmbedderIdentity = errors.New("govern: embedder identity is required")
)

// Searcher is the part of a recall store this package needs.
type Searcher interface {
	Search(ctx context.Context, query string, opts index.SearchOptions) ([]index.SearchResult, error)
}

// Recorder receives one entry per completed retrieval.
//
// Required rather than optional. A retrieval nobody can account for afterwards
// is the thing an audited store exists to make impossible, and an optional
// recorder is one a caller forgets.
type Recorder interface {
	RecordRetrieval(ctx context.Context, entry Entry) error
}

// Entry is what a completed retrieval records.
//
// Embedder and Model are here because a retrieval is only reproducible against
// the model that produced its vectors. The store this was ported from refused
// a search with no embedding provider for exactly that reason -- not because
// it could not have defaulted one, but because a silent default would make
// retrievals unattributable. recall injects its embedder at construction
// rather than per-request, so the same guarantee is met by requiring the
// identity once, when the governed view is built.
type Entry struct {
	Query          string
	Classification string
	SourceFilters  []string
	AllSources     bool
	Agent          string
	TaskID         string
	ResultCount    int
	Embedder       string
	Model          string
}

// Request is a governed retrieval. Every field that must be decided is a field
// a caller has to set; none of them default.
type Request struct {
	Query          string
	Classification string
	SourceFilters  []string
	AllSources     bool
	Agent          string
	TaskID         string
	TopK           int
}

// Store is a fail-closed view over a recall store.
type Store struct {
	// ClassificationKey and SourceKey name the chunk metadata fields carrying
	// each value. They are configurable because the vocabulary belongs to the
	// embedding system, not to this package.
	ClassificationKey string
	SourceKey         string

	// embedder and model identify what produced the vectors being searched.
	// Set once at construction and recorded on every retrieval.
	embedder string
	model    string

	search   Searcher
	recorder Recorder
}

// New returns a governed view.
//
// It refuses a nil recorder here rather than at the first search, so a system
// wired without auditing fails at construction rather than serving unrecorded
// retrievals until someone notices. The same reasoning applies to the embedder
// identity: an unattributable retrieval is refused at wiring time, when it is
// cheap to fix.
func New(search Searcher, recorder Recorder, embedder, model string) (*Store, error) {
	if search == nil {
		return nil, fmt.Errorf("govern: a searcher is required")
	}
	if recorder == nil {
		return nil, ErrNoRecorder
	}
	if strings.TrimSpace(embedder) == "" || strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("%w: name the embedder and model whose vectors are "+
			"being searched, so a recorded retrieval is reproducible", ErrNoEmbedderIdentity)
	}
	return &Store{
		ClassificationKey: "classification",
		SourceKey:         "source",
		embedder:          embedder,
		model:             model,
		search:            search,
		recorder:          recorder,
	}, nil
}

// Validate applies every refusal without performing a retrieval.
//
// Exported so a caller can check a request before building one, and so the
// refusals can be tested without a store — which is also how they are proven
// to happen before anything is touched.
func (s *Store) Validate(req Request) error {
	// Trimmed, not just compared to empty. A whitespace-only query passed this
	// check, reached the store, scored 0 against everything and wrote an audit
	// row -- a retrieval that happened, was recorded, and asked nothing. The
	// refusal exists so that no unqualified query is served; "  " is as
	// unqualified as "".
	//
	// Inherited faithfully from the engine this was ported from, which used
	// the same exact comparison. Ported behaviour is not automatically correct
	// behaviour, and the contract case only ever exercised the empty string.
	if strings.TrimSpace(req.Query) == "" {
		return ErrNoQuery
	}
	if req.Classification == "" {
		return ErrNoClassification
	}
	if req.AllSources && len(req.SourceFilters) > 0 {
		return ErrAmbiguousScope
	}
	if !req.AllSources {
		if len(req.SourceFilters) == 0 {
			return fmt.Errorf("%w: name at least one source, or set AllSources to "+
				"deliberately span every source in the store", ErrNoScope)
		}
		for _, source := range req.SourceFilters {
			if strings.TrimSpace(source) == "" {
				return ErrBlankSource
			}
		}
	}
	if s.recorder == nil {
		return ErrNoRecorder
	}
	return nil
}

// Search performs a governed retrieval, or refuses.
func (s *Store) Search(ctx context.Context, req Request) ([]index.SearchResult, error) {
	if err := s.Validate(req); err != nil {
		return nil, err
	}

	filters := []index.Filter{
		&index.TermFilter{Key: s.ClassificationKey, Value: req.Classification},
	}
	if !req.AllSources {
		filters = append(filters, &index.TermInFilter{Key: s.SourceKey, Values: req.SourceFilters})
	}

	topK := req.TopK
	if topK <= 0 {
		topK = 10
	}

	results, err := s.search.Search(ctx, req.Query, index.SearchOptions{
		TopK:    topK,
		Filters: filters,
	})
	if err != nil {
		return nil, err
	}

	// Recorded before returning, and a recording failure fails the retrieval.
	// Returning results the system cannot account for would make the audit
	// advisory, which is the same as not having one.
	if err := s.recorder.RecordRetrieval(ctx, Entry{
		Query: req.Query, Classification: req.Classification,
		SourceFilters: req.SourceFilters, AllSources: req.AllSources,
		Agent: req.Agent, TaskID: req.TaskID, ResultCount: len(results),
		Embedder: s.embedder, Model: s.model,
	}); err != nil {
		return nil, fmt.Errorf("govern: retrieval could not be recorded: %w", err)
	}
	return results, nil
}
