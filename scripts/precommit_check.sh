#!/usr/bin/env bash
# Gate for commits touching the k3-ledger / Metal work.
#
# This exists because a commit went out with a file that did not compile: the
# build was chained with `;` so its failure did not stop the commit, and a
# replacement check piped grep into head, which returns head's status rather
# than grep's. Both are shell-shaped mistakes that a script with `set -euo
# pipefail` cannot make. Run this, and commit only if it exits 0.
set -euo pipefail

PKGS=("${@:-larql-cli larql-compute-metal}")

for p in ${PKGS[@]}; do
  echo "== fmt $p =="
  # Scoped, not --all: an unrelated package with in-flight work must not make
  # this gate unusable, or it stops being run.
  cargo fmt -p "$p" -- --check
  echo "== build $p (lib + bins + examples) =="
  cargo build -q -p "$p" --all-targets
  echo "== test $p =="
  cargo test -q -p "$p" 2>&1 | tail -5
done

echo "== whitespace / conflict markers =="
git diff --check

echo
echo "OK — safe to commit."
