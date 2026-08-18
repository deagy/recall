package ingest

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/deagy/recall/connector"
	"github.com/deagy/recall/core"
	"github.com/deagy/recall/loader"
	"github.com/deagy/recall/store"
)

// Transform rewrites a loaded document before ingestion (e.g. stripping
// boilerplate, translating, or enriching metadata). Returning (nil, nil)
// drops the document.
type Transform func(doc *loader.Document) (*loader.Document, error)

// Options configures a Pipeline. Store and exactly one of Connector or
// Loader are required; everything else is optional.
type Options struct {
	// Store receives the ingested documents (chunks + embeds + indexes).
	Store store.Store

	// Connector fetches from an external source (web, git, S3, ...).
	Connector connector.Connector

	// Loader fetches from a local source. Mutually exclusive with Connector.
	Loader loader.Loader

	// Source is the ref passed to the connector/loader.
	Source string

	// Transform is an optional preprocessing step applied per document.
	Transform Transform

	// Validator optionally rejects documents that violate a Schema.
	Validator *Validator

	// Dedup optionally suppresses documents whose content was already
	// ingested (persistable across runs).
	Dedup *Deduplicator

	// Incremental optionally re-ingests only documents whose content
	// changed since the last run (persistable across runs).
	Incremental *Incremental

	// Progress optionally receives counters and callbacks.
	Progress *Progress

	// Concurrency is the number of documents processed in parallel.
	// 0 or 1 means sequential.
	Concurrency int
}

// Pipeline runs the load -> filter -> transform -> upload flow against a
// configured store.
type Pipeline struct {
	opts Options
}

// NewPipeline validates options and returns a ready Pipeline.
func NewPipeline(opts Options) (*Pipeline, error) {
	if opts.Store == nil {
		return nil, errors.New("ingest: Store is required")
	}
	if (opts.Connector == nil) == (opts.Loader == nil) {
		return nil, errors.New("ingest: exactly one of Connector or Loader is required")
	}
	if opts.Concurrency < 0 {
		return nil, errors.New("ingest: Concurrency must be >= 0")
	}
	return &Pipeline{opts: opts}, nil
}

// Run ingests all documents available at the configured Source. It returns
// a per-run Result; a non-nil error is returned when the source itself
// failed, when the incremental state could not be saved, or (via
// Result.Error) when individual documents failed.
func (p *Pipeline) Run(ctx context.Context) (*Result, error) {
	result := &Result{Source: p.opts.Source}
	start := time.Now()
	prog := p.opts.Progress
	if prog == nil {
		prog = NewProgress()
	}
	defer func() {
		result.Duration = time.Since(start)
		if p.opts.Incremental != nil {
			if err := p.opts.Incremental.Save(); err != nil {
				result.Failed = append(result.Failed, Failure{ID: "<state>", Err: err})
			}
		}
	}()

	prog.SetPhase("load")
	var docs []*loader.Document
	var err error
	switch {
	case p.opts.Connector != nil:
		docs, err = p.opts.Connector.Fetch(ctx, p.opts.Source)
	case p.opts.Loader != nil:
		docs, err = p.opts.Loader.Load(ctx, p.opts.Source)
	}
	if err != nil {
		return result, fmt.Errorf("ingest: load %s: %w", p.opts.Source, err)
	}
	result.Loaded = len(docs)
	prog.Loaded(len(docs))

	prog.SetPhase("upload")
	p.process(ctx, docs, prog, result)

	return result, nil
}

// process filters and uploads documents, sequentially or with a worker
// pool when Concurrency > 1.
func (p *Pipeline) process(ctx context.Context, docs []*loader.Document, prog *Progress, result *Result) {
	var mu sync.Mutex
	fail := func(id string, err error) {
		mu.Lock()
		result.Failed = append(result.Failed, Failure{ID: id, Err: err})
		mu.Unlock()
		prog.Fail(id)
	}

	work := func(d *loader.Document) {
		if ctx.Err() != nil {
			fail(d.ID, ctx.Err())
			return
		}
		// Incremental: skip unchanged documents.
		if p.opts.Incremental != nil {
			if h := ContentHash(d.Content); p.opts.Incremental.ShouldSkip(d.ID, h) {
				prog.Skip(d.ID)
				result.Skipped++
				return
			}
		}
		// Dedup: skip content already ingested.
		if p.opts.Dedup != nil && p.opts.Dedup.IsDuplicate(d.Content) {
			prog.Skip(d.ID)
			result.Skipped++
			return
		}
		// Validation: reject documents that violate the schema.
		if p.opts.Validator != nil {
			if verr := p.opts.Validator.Validate(d); verr != nil {
				prog.Skip(d.ID)
				result.Skipped++
				return
			}
		}
		// Transform: optional preprocessing.
		doc := d
		if p.opts.Transform != nil {
			var terr error
			doc, terr = p.opts.Transform(doc)
			if terr != nil {
				fail(d.ID, terr)
				return
			}
			if doc == nil {
				prog.Skip(d.ID)
				result.Skipped++
				return
			}
		}
		// Upload: the store chunks, embeds, and indexes.
		if err := p.opts.Store.Upload(ctx, toCoreDocument(doc), doc.Content); err != nil {
			fail(doc.ID, err)
			return
		}
		prog.Upload(doc.ID)
		result.Uploaded++
		if p.opts.Dedup != nil {
			p.opts.Dedup.Mark(doc.Content)
		}
		if p.opts.Incremental != nil {
			p.opts.Incremental.Mark(doc.ID, ContentHash(doc.Content))
		}
	}

	if p.opts.Concurrency <= 1 || len(docs) <= 1 {
		for _, d := range docs {
			work(d)
		}
		return
	}
	workers := p.opts.Concurrency
	if workers > len(docs) {
		workers = len(docs)
	}
	jobs := make(chan *loader.Document)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for d := range jobs {
				work(d)
			}
		}()
	}
	for _, d := range docs {
		jobs <- d
	}
	close(jobs)
	wg.Wait()
}

// toCoreDocument converts a loader document into the store's document type,
// copying metadata by reference-safe map copy.
func toCoreDocument(d *loader.Document) *core.Document {
	cd := core.NewDocument(d.ID, d.Title, d.Source)
	if len(d.Metadata) > 0 {
		cd.Metadata = make(map[string]core.Value, len(d.Metadata))
		for k, v := range d.Metadata {
			cd.Metadata[k] = v
		}
	}
	return cd
}
