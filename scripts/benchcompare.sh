#!/usr/bin/env bash
# benchcompare.sh — compare two `go test -bench` outputs and fail on
# significant ns/op regressions.
#
# Usage:
#   scripts/benchcompare.sh <base.txt> <current.txt> [threshold_percent]
#
# Environment:
#   FLOOR_NS   absolute delta (ns/op) below which changes are ignored
#              (default 1000 — filters sub-microsecond noise)
#
# Exit codes:
#   0  no regressions beyond the threshold
#   1  at least one benchmark regressed beyond the threshold
#   2  usage / parse error
#
# The comparison is intentionally conservative: CI runners are shared and
# noisy, so a regression is only flagged when it is BOTH larger than the
# relative threshold (default 50%) AND larger than the absolute floor.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <base.txt> <current.txt> [threshold_percent]" >&2
    exit 2
fi

base="$1"
current="$2"
threshold="${3:-50}"
floor_ns="${FLOOR_NS:-1000}"

if [ ! -f "$base" ] || [ ! -f "$current" ]; then
    echo "benchcompare: missing input file(s): $base $current" >&2
    exit 2
fi

# Extract "BenchmarkName ns/op" pairs from go test benchmark output.
# Lines look like: BenchmarkFoo-8   100   12345 ns/op   64 B/op   2 allocs/op
# Fields: $1 name, $2 iterations, $3 value, $4 unit. The trailing -N
# (GOMAXPROCS) suffix is stripped so names match across runs.
extract() {
    awk '$4 == "ns/op" {
        name = $1
        sub(/-[0-9]+$/, "", name)
        print name, $3
    }' "$1"
}

base_extracted="$(extract "$base")"
current_extracted="$(extract "$current")"

if [ -z "$base_extracted" ] || [ -z "$current_extracted" ]; then
    echo "benchcompare: no benchmark results parsed (missing 'ns/op' lines)" >&2
    exit 2
fi

printf '%s\n' "$base_extracted" > /tmp/.bench-base.$$
printf '%s\n' "$current_extracted" > /tmp/.bench-current.$$
trap 'rm -f /tmp/.bench-base.$$ /tmp/.bench-current.$$' EXIT

awk -v thr="$threshold" -v floor="$floor_ns" '
    NR == FNR { base[$1] = $2; next }
    {
        name = $1; cur = $2
        if (!(name in base)) {
            printf "NEW      %-60s %12.0f ns/op (no baseline)\n", name, cur
            next
        }
        b = base[name]
        seen[name] = 1
        if (b <= 0) { next }
        delta_pct = (cur - b) / b * 100
        if (delta_pct > thr && (cur - b) > floor) {
            printf "REGRESS  %-60s base=%12.0f  cur=%12.0f  (+%.1f%%)\n", name, b, cur, delta_pct
            regress++
        } else if (delta_pct < -thr && (b - cur) > floor) {
            printf "IMPROVE  %-60s base=%12.0f  cur=%12.0f  (%.1f%%)\n", name, b, cur, delta_pct
        }
    }
    END {
        for (name in base) {
            if (!(name in seen)) {
                printf "REMOVED  %-60s (not run in current)\n", name
            }
        }
        if (regress > 0) {
            printf "\nbenchcompare: %d benchmark(s) regressed beyond +%s%% (floor %s ns/op)\n", regress, thr, floor
            exit 1
        }
        printf "\nbenchcompare: no regressions beyond +%s%% (floor %s ns/op)\n", thr, floor
    }
' /tmp/.bench-base.$$ /tmp/.bench-current.$$
