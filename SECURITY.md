# Security Policy

## Supported Versions

Recall is under active development and has not yet reached a 1.0 release.
Security fixes are applied to the latest development version on `main` and to
the most recent tagged release only.

| Version | Supported          |
| ------- | ------------------ |
| latest tag on `main` | security fixes |
| older tags | upgrade to the latest release |

After 1.0, this policy will be revisited to cover at least the latest minor
release line.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately via GitHub Security Advisories:

1. Open the repository's **Security** tab on GitHub.
2. Click **Report a vulnerability**.
3. Include as much detail as possible: affected version/commit, reproduction
   steps, impact, and any suggested fix.

You should receive an acknowledgment within a few days. If you do not get a
response within a week, please follow up in the same advisory thread. We aim
to keep reporters informed about progress and will credit reporters in the
release notes unless anonymity is requested.

### Scope

In scope, for example:

- Vulnerabilities in the Recall library packages (`github.com/deagy/recall/...`)
- Vulnerabilities in the shipped binaries (`recall`, `recall-server`) and the
  REST API surface (authentication bypass, injection, path traversal, etc.)

Out of scope:

- Vulnerabilities in third-party dependencies with no exploit path in Recall
  (please report those upstream; we track them with `govulncheck` in CI)
- Issues caused by deployment misconfiguration (e.g. running without
  authentication on a public network) — see the operational guidance below

## Security-Relevant Features

- **Authentication** — the REST API (`api` package) supports static API keys
  (`X-API-Key` or Bearer) and HS256 JWTs verified with the standard library.
  Endpoints are unauthenticated only when no authenticator is configured;
  production deployments should always configure `auth` (see
  `deploy/config/recall.example.yaml`).
- **Namespace-scoped API keys** — `api.ScopedAPIKeyAuth` restricts keys to a
  set of namespaces and enforces the scope on every data endpoint, failing
  closed (out-of-scope data is reported as 404/403).
- **Hardening** — per-request body size limits, configurable CORS, graceful
  shutdown, and no secrets in configuration files (API keys are read from
  environment variables named by `api_key_env`).
- **Dependency scanning** — CI runs `govulncheck` on every change and license
  compliance checks on all dependencies.

For operational guidance (network exposure, key management, encryption
considerations), see the **Security Guidance** section of
[README.md](./README.md).
