#!/usr/bin/env bash
# release-build.sh — cross-compile the recall and recall-server binaries for
# release publishing, then write SHA-256 checksums.
#
# Usage:
#   scripts/release-build.sh <version>     # e.g. v0.1.0
#
# Output lands in ./dist/. The version is injected into the binaries via
# -ldflags "-X main.version=...".
set -euo pipefail

version="${1:?usage: release-build.sh <version> (e.g. v0.1.0)}"

case "$version" in
    v[0-9]*.[0-9]*.[0-9]*) ;; # ok
    *)
        echo "release-build: version must look like vMAJOR.MINOR.PATCH (got: $version)" >&2
        exit 2
        ;;
esac

mkdir -p dist
rm -f dist/recall* dist/checksums-sha256.txt 2>/dev/null || true

for goos in linux darwin windows; do
    for goarch in amd64 arm64; do
        for bin in recall recall-server; do
            name="${bin}-${version#v}-${goos}-${goarch}"
            out="dist/${name}"
            if [ "$goos" = "windows" ]; then
                out="${out}.exe"
            fi
            echo "building ${out}"
            GOOS="$goos" GOARCH="$goarch" CGO_ENABLED=0 go build \
                -trimpath \
                -ldflags "-s -w -X main.version=${version}" \
                -o "$out" "./cmd/${bin}"
        done
    done
done

(
    cd dist
    sha256sum recall* > checksums-sha256.txt
)
echo "built $(ls dist | grep -c '^recall') binaries + checksums in dist/"
