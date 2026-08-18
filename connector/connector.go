// Package connector provides source connectors that fetch raw content from
// external systems (web pages, git repositories, S3 buckets, GitHub, SQL
// databases) and normalize it into loader.Documents that the ingest
// pipeline can process.
//
// Connectors are deliberately dumb about format: they return raw text
// (or extracted text for HTML sources). Transformation, chunking, and
// embedding are the job of the ingest pipeline.
package connector

import (
	"context"

	"github.com/deagy/recall/loader"
)

// Connector fetches documents from an external source. The ref argument is
// connector-specific: a URL for WebConnector, "owner/repo" for
// GitHubConnector, "bucket/prefix" for S3Connector, a repo URL for
// GitConnector, and ignored by DatabaseConnector.
type Connector interface {
	// Name returns a stable identifier for this connector type.
	Name() string

	// Fetch retrieves all documents available at ref. Implementations
	// should honor context cancellation.
	Fetch(ctx context.Context, ref string) ([]*loader.Document, error)
}
