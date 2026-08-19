# Contributing to Recall

Thanks for your interest in Recall — a Go library and service toolkit for
building Retrieval-Augmented Generation (RAG) applications. This guide covers
the development workflow, coding standards, and the process for submitting
changes.

By participating you agree to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md). For security issues, please follow
[SECURITY.md](./SECURITY.md) instead of opening a public issue.

## Getting Started

Requirements:

- **Go 1.26.5+** (see `go.mod`)
- No C compiler or CGO needed — the whole project builds with `CGO_ENABLED=0`

```bash
git clone https://github.com/deagy/recall.git
cd recall
go build ./...
go test ./... -count=1
```

Useful entry points:

| File | Purpose |
| ---- | ------- |
| [README.md](./README.md) | Features, quick start, usage examples |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Package layout, interfaces, design decisions |
| [ROADMAP.md](./ROADMAP.md) | Planned phases and priorities |
| [PLANNING.md](./PLANNING.md) | Record of completed phases |
| [example/](./example/) | Runnable examples, including an end-to-end tutorial |

## Development Workflow

A change is ready for review when these all pass:

```bash
gofmt -s -w .                      # formatting (CI fails on unformatted code)
go vet ./...                       # static analysis (no warnings)
golangci-lint run                  # repo linter config (.golangci.yml)
go test ./... -count=1             # full test suite
```

Additional checks run in CI that you can run locally:

```bash
# Race detector (needs a C toolchain for the tsan runtime; the library
# itself remains CGO-free — this only affects the test binary link step)
CGO_ENABLED=1 go test ./... -race -count=1

# Coverage (CI enforces >= 80% overall)
go test ./... -count=1 -coverprofile=coverage.out -covermode=atomic
go tool cover -func=coverage.out | tail -1

# Dependency vulnerability scan
go run golang.org/x/vuln/cmd/govulncheck@latest ./...

# License compliance of dependencies
go run github.com/google/go-licenses@latest check ./...
```

See [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) for running and comparing
benchmarks.

## Coding Standards

- Follow `gofmt`/`go vet` strictly; packages are lowercase and singular.
- **Errors:** always check and return errors with context; never discard them.
- **Interfaces:** define them at the caller's site (dependency inversion);
  `-er` names (`Embedder`, `Store`, `Reranker`).
- **Concurrency:** protect shared state with `sync.RWMutex`; prefer immutable
  data; never panic in production paths.
- **Contexts:** long-running operations take `context.Context` first.
- **Zero CGO:** never introduce CGO or C-dependent dependencies into library
  code. SQLite stays on `modernc.org/sqlite`; new functionality should prefer
  the standard library.
- **No secrets:** never hardcode API keys or endpoints; keys come from
  environment variables.
- **Comments:** add godoc comments to all exported identifiers; comment the
  *why* for non-obvious invariants.

### Dependencies

Prefer the standard library and existing dependencies. Before adding a new
dependency, open an issue to discuss it — Recall deliberately keeps a small
dependency footprint.

### Testing

- Target **≥80% statement coverage** for every package (CI enforces the
  overall threshold).
- Name tests after the behavior: `TestKnowledgeGraph_AddEntity`.
- Use `t.Fatalf` for critical failures, `t.Errorf` for non-fatal ones.
- Cover error and boundary cases, and add `Benchmark*` functions for
  performance-sensitive code.
- Test helpers shared across packages live in `testutil` (import it only from
  `_test.go` files).
- Use deterministic doubles: `embedder.NewMockEmbedder`, `llm.NewMockBackend`,
  `testutil.NewFixtureStore`.

## Commit Messages

We use conventional commits, which also feed the auto-generated release notes:

| Prefix | Use |
| ------ | --- |
| `feat:` | New feature or phase completion |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Test additions or fixes |
| `refactor:` | Restructuring without behavior change |
| `chore:` | Build, CI, tooling |

Phase completions are committed as `feat: Phase N — <description>`.

## Submitting Changes

1. Open an issue for anything significant (new APIs, new packages, behavior
   changes) so the approach can be aligned before you invest time. Small fixes
   can go straight to a PR.
2. Fork the repo and create a branch from `main`.
3. Keep the change focused: one concern per PR; don't mix features, fixes,
   and unrelated reformatting.
4. Update docs alongside the change: README status, `PLANNING.md` for phase
   work, `CHANGELOG.md` under `[Unreleased]`, and godoc comments.
5. Make CI green: format, vet, lint, tests, coverage gate.
6. Submit the PR against `main` with a clear description, linked issue, and
   any verification you performed.

## Releases

- Recall follows [Semantic Versioning](https://semver.org/). Pre-1.0, minor
  releases may contain breaking changes; they are always documented in
  [CHANGELOG.md](./CHANGELOG.md) and [docs/MIGRATION.md](./docs/MIGRATION.md).
- Maintainers cut releases with the automated tag workflow
  (`.github/workflows/tag.yml`) and the release workflow builds binaries and
  publishes notes (`.github/workflows/release.yml`).

## Questions?

Open a GitHub issue for bug reports and feature requests, or a discussion for
usage questions.
