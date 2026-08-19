# Recall Governance

This document describes how the Recall project is governed, how decisions are
made, and how you can get involved.

## Current Model

Recall is currently a **single-maintainer project**:

- **Project lead:** Daniel Eagy ([@deagy](https://github.com/deagy))

The project lead is responsible for the project's direction, merges pull
requests, manages releases, and has final say in disagreements. This model is
deliberate for a young project: it keeps decision-making fast and the API
coherent.

## How Decisions Are Made

- **Roadmap.** Long-term direction lives in [ROADMAP.md](./ROADMAP.md),
  organized into phases. Completed phases are recorded in
  [PLANNING.md](./PLANNING.md).
- **Day-to-day changes.** Small changes (bug fixes, tests, docs, examples) can
  be proposed directly as pull requests.
- **Significant changes.** New public APIs, new packages, behavior changes, or
  new dependencies should be discussed in an issue before substantial work
  begins, to avoid wasted effort.
- **Design constraints.** All proposals must respect the project's hard
  constraints: zero CGO for library code, dependency injection for external
  services, and ≥80% test coverage. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Contributions

Anyone can contribute via issues and pull requests. The contribution workflow,
coding standards, and commit conventions are documented in
[CONTRIBUTING.md](./CONTRIBUTING.md). All participants are expected to follow
the [Code of Conduct](./CODE_OF_CONDUCT.md).

Contributions are merged at the project lead's discretion. Sustained,
high-quality contributions are the path to commit access and maintainership;
if the maintainer group grows, this document will be updated to describe
roles, voting, and escalation.

## Releases and Versioning

- Recall follows [Semantic Versioning 2.0.0](https://semver.org/).
- Releases are cut from `main` by the project lead (see the automated tag and
  release workflows described in [CONTRIBUTING.md](./CONTRIBUTING.md)).
- Version history and migration notes live in [CHANGELOG.md](./CHANGELOG.md)
  and [docs/MIGRATION.md](./docs/MIGRATION.md).

## Licensing

Recall is licensed under the [MIT License](./LICENSE). By submitting a
contribution, you agree that your contribution will be licensed under the same
license.

## Changes to This Document

Changes to this governance document are made by the project lead and are
announced in the [CHANGELOG.md](./CHANGELOG.md).
