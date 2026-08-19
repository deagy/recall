# Migration Guide

This guide explains Recall's versioning policy and how to upgrade between
releases. Per-release breaking changes and their migration steps are recorded
here as they happen.

> **Status:** Recall has not tagged a release yet. All APIs are evolving on
> `main` toward `v0.1.0`. Until then, pin dependencies by commit or pseudo-
> version if you need stability.

## Versioning Policy

Recall follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v0.X.0`) — new functionality, backwards compatible.
  - *Pre-1.0 exception:* while the major version is `0`, minor releases **may
    contain breaking changes** as APIs stabilize. They are always listed here
    and in the [CHANGELOG](../CHANGELOG.md) with a **Breaking** label.
- **PATCH** (`v0.0.X`) — backwards-compatible bug fixes only.

**What counts as the public API:** all exported identifiers in non-`internal`
packages (`github.com/deagy/recall/...`). `testutil` is exported but intended
for `_test.go` imports; `cmd/*` binaries are user-facing but not importable
API.

**Go version support:** Recall supports the Go release declared in `go.mod`
(currently **Go 1.26.5+**).

## How to Upgrade

```bash
# Upgrade to a specific release
go get github.com/deagy/recall@v0.2.0
go mod tidy

# Upgrade to the latest release
go get -u github.com/deagy/recall@latest
go mod tidy
```

Then:

1. Read the release entry in [CHANGELOG.md](../CHANGELOG.md), especially any
   **Breaking** items.
2. Check the matching section below for step-by-step migration notes.
3. Run `go build ./...` and your test suite; fix any compile errors or
   behavior changes.

Service deployments (`recall-server`, `cmd/recall`): rebuild or pull the new
release binaries (published on the GitHub release page), then run
`recall store migrate` if the release notes mention a schema migration.
Back up first: `recall store backup`.

## Breaking-Change Practices

- **Deprecate first.** Unless impossible, deprecated APIs survive at least one
  minor release (with a `// Deprecated:` godoc note) before removal.
- **Document here.** Every breaking change gets an entry below with the
  affected symbols and the recommended replacement.
- **Config stability.** `config.Config` fields are additive; fields are not
  renamed or removed without a migration note.

## Release Notes

_No releases have been published yet. Use the template below when adding
entries:_

```markdown
### v0.X.0 → v0.Y.0

**Breaking changes**

- `pkg.Symbol` — what changed, why, and how to migrate.

**Behavioral changes**

- Description of any observable behavior change that is not API-breaking.

**Schema/config changes**

- Migration steps (e.g. `recall store migrate`).
```
