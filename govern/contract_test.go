package govern_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/deagy/recall/govern"
	"github.com/deagy/recall/index"
)

// The contract this package exists to satisfy, driven from the store it was
// ported from rather than restated here.
//
// cadre's knowledge store enforced these refusals in production. Its retrieval
// engine is being retired in favour of recall, and the refusals do not come
// with an engine — recall's own Search takes filters a caller may omit and
// spans every namespace by default. This fixture was captured there, while
// that store still existed, precisely so this package could be measured
// against it rather than against someone's memory of it.
const contractPath = "testdata/fail-closed-contract.json"

type contractCase struct {
	Name           string   `json:"name"`
	Why            string   `json:"why"`
	Query          string   `json:"query"`
	Classification string   `json:"classification"`
	Provider       bool     `json:"provider"`
	AllSources     bool     `json:"all_sources"`
	SourceFilters  []string `json:"source_filters"`
	ExpectRefusal  string   `json:"expect_refusal"`
}

// refusalFor maps the origin store's refusal text onto this package's sentinel
// errors. One case does not map, and that is recorded rather than hidden: see
// TestTheUnportedRefusalIsAccountedFor.
var refusalFor = map[string]error{
	"query is required":          govern.ErrNoQuery,
	"classification is required": govern.ErrNoClassification,
	"source scope is required":   govern.ErrNoScope,
	"source scope is ambiguous":  govern.ErrAmbiguousScope,
	"must be non-empty":          govern.ErrBlankSource,
}

func loadContract(t *testing.T) []contractCase {
	t.Helper()
	raw, err := os.ReadFile(contractPath)
	if err != nil {
		t.Fatalf("read %s: %v", contractPath, err)
	}
	var file struct {
		Cases []contractCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatalf("parse %s: %v", contractPath, err)
	}
	if len(file.Cases) == 0 {
		t.Fatalf("%s holds no cases", contractPath)
	}
	return file.Cases
}

type refusingSearcher struct{ t *testing.T }

func (r refusingSearcher) Search(context.Context, string, index.SearchOptions) ([]index.SearchResult, error) {
	r.t.Error("a refused request reached the store; a refusal that must happen " +
		"before anything is touched now happens after it")
	return nil, nil
}

type countingRecorder struct{ entries []govern.Entry }

func (c *countingRecorder) RecordRetrieval(_ context.Context, e govern.Entry) error {
	c.entries = append(c.entries, e)
	return nil
}

func TestEveryPortedContractCaseIsRefused(t *testing.T) {
	store, err := govern.New(refusingSearcher{t: t}, &countingRecorder{}, "test-embedder", "test-model")
	if err != nil {
		t.Fatal(err)
	}
	ported := 0
	for _, testCase := range loadContract(t) {
		want, mapped := refusalFor[testCase.ExpectRefusal]
		if !mapped {
			continue
		}
		ported++
		t.Run(testCase.Name, func(t *testing.T) {
			results, err := store.Search(context.Background(), govern.Request{
				Query:          testCase.Query,
				Classification: testCase.Classification,
				AllSources:     testCase.AllSources,
				SourceFilters:  testCase.SourceFilters,
			})
			if err == nil {
				t.Fatalf("accepted. %s", testCase.Why)
			}
			if results != nil {
				t.Error("results returned alongside a refusal")
			}
			if !errors.Is(err, want) {
				t.Errorf("refused for the wrong reason.\n  got:  %v\n  want: %v\n  %s",
					err, want, testCase.Why)
			}
		})
	}
	if ported == 0 {
		t.Fatal("no contract case mapped; this guard asserted nothing")
	}
	t.Logf("ported refusals exercised: %d", ported)
}

// The origin store refused a search with no embedding provider. recall injects
// its embedder at construction rather than per request, so that refusal has no
// per-request equivalent — but its reason does, and this is where it is kept
// honest rather than quietly dropped.
func TestTheUnportedRefusalIsAccountedFor(t *testing.T) {
	unported := 0
	for _, testCase := range loadContract(t) {
		if _, mapped := refusalFor[testCase.ExpectRefusal]; !mapped {
			unported++
			if testCase.ExpectRefusal != "embedding provider is required" {
				t.Errorf("contract case %q does not map to a refusal here and is not the "+
					"one known to be architectural: %s", testCase.Name, testCase.Why)
			}
		}
	}
	if unported != 1 {
		t.Errorf("expected exactly one unported case, found %d — either a refusal was "+
			"lost in the port or a new one appeared in the contract", unported)
	}

	// Its reason survives as a construction-time requirement.
	if _, err := govern.New(refusingSearcher{t: t}, &countingRecorder{}, "", ""); !errors.Is(err, govern.ErrNoEmbedderIdentity) {
		t.Errorf("an unattributable retrieval was permitted at construction: %v", err)
	}
}

// A recorder is not optional, and neither is recording succeeding.
func TestARetrievalThatCannotBeRecordedIsRefused(t *testing.T) {
	if _, err := govern.New(refusingSearcher{t: t}, nil, "e", "m"); !errors.Is(err, govern.ErrNoRecorder) {
		t.Errorf("a store with no recorder was constructed: %v", err)
	}
}

func TestTheContractMatchesItsOrigin(t *testing.T) {
	source := os.Getenv("CADRE_FAIL_CLOSED_CONTRACT")
	if source == "" {
		root, err := filepath.Abs("..")
		if err != nil {
			t.Fatal(err)
		}
		candidate := filepath.Join(filepath.Dir(root), "cadre",
			"internal", "knowledge", "testdata", "fail-closed-contract.json")
		if _, err := os.Stat(candidate); err == nil {
			source = candidate
		}
	}
	if source == "" {
		if os.Getenv("CI") != "" {
			t.Fatal("no origin contract reachable and this is CI, where this guard must " +
				"not be skipped. Set CADRE_FAIL_CLOSED_CONTRACT.")
		}
		t.Skip("no cadre checkout reachable; set CADRE_FAIL_CLOSED_CONTRACT to check")
	}
	want, err := os.ReadFile(source)
	if err != nil {
		t.Fatalf("read %s: %v", source, err)
	}
	got, err := os.ReadFile(contractPath)
	if err != nil {
		t.Fatalf("read %s: %v", contractPath, err)
	}
	var a, b any
	if err := json.Unmarshal(want, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &b); err != nil {
		t.Fatal(err)
	}
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if string(aj) != string(bj) {
		t.Errorf("%s differs from the contract at %s. This copy is not authoritative; "+
			"re-vendor rather than editing it.", contractPath, source)
	}
}
