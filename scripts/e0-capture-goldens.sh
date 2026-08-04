#!/usr/bin/env bash
# E0 corpus C3 — capture pinned golden outputs from the PRE-V2 binary.
#
# E0 (docs/vindex2-experiments.md) asserts that adding VINDEX3 support to the
# binary changes nothing observable on any VINDEX2 path. That assertion needs a
# fixed record to compare against.
#
# WHY THIS EXISTS AT ALL: without committed goldens, "zero behavioural
# regression" silently degrades into "the two binaries agree with each other" —
# a condition any bug present in BOTH satisfies. Since the whole point of E0 is
# to catch an incumbent path damaged by successor work sharing one binary,
# comparing two builds of that binary to each other is close to circular. The
# baseline has to predate the successor code and has to be committed.
#
# WHEN: run this against a binary built from a commit BEFORE any lyrw2 work.
# Once VINDEX3 merges to the main line, the baseline stops being a checkout and
# starts being an archaeology exercise — and a reconstructed baseline is exactly
# the artefact that drifts without anyone noticing.
#
# THE OTHER HALF is `e0-verify-goldens.sh`, which replays this record against a
# current binary. Capturing without ever verifying leaves an assertion nobody
# makes. The corpus itself lives in `lib/e0-corpus.sh` and is shared by both, so
# the two cannot drift apart.
#
# Usage:
#   E0_BIN=<pre-v2 larql> E0_VINDEX=output/gemma.vindex ./scripts/e0-capture-goldens.sh
#
# Env vars:
#   E0_BIN              — larql binary to capture from   (required; must be pre-v2)
#   E0_VINDEX           — vindex to exercise             (required)
#   E0_MODEL            — checkpoint the vindex came from (recorded in the recipe)
#   E0_MODEL_REVISION   — that checkpoint's revision hash (recorded in the recipe)
#   E0_EXTRACT_FLAGS    — flags used to extract it        (recorded in the recipe)
#   E0_OUT              — golden output dir              (default: tests/goldens/e0/<vindex name>)
#   E0_TOKENS           — tokens to decode per prompt    (default: 24)
#   E0_WALK_K           — WALK top-K per layer           (default: 20)
#
# The vindex itself is NOT committed — it is regenerable. The recipe is.
#
# Determinism: `larql run` samples greedily (SamplingConfig::greedy), so decode
# is reproducible without a seed flag. Every command's output is normalised for
# paths and timings before writing, so a diff means a behavioural change rather
# than a different working directory or a faster machine.

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/e0-corpus.sh
source "${SCRIPT_DIR}/lib/e0-corpus.sh"

readonly TOKENS="${E0_TOKENS:-24}"
readonly WALK_K="${E0_WALK_K:-20}"

: "${E0_BIN:?set E0_BIN to a larql binary built BEFORE any lyrw2 commit}"
: "${E0_VINDEX:?set E0_VINDEX to a vindex directory}"

readonly OUT_DIR="${E0_OUT:-tests/goldens/e0/$(basename "$E0_VINDEX")}"
readonly TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OUT_DIR"

echo "E0 C3 golden capture"
echo "  binary: ${E0_BIN}"
echo "  vindex: ${E0_VINDEX}"
echo "  out:    ${OUT_DIR}"
echo

# Provenance + regeneration recipe.
#
# The corpus is reproducible, so it is pinned rather than stored: the multi-GB
# vindex is not committed, this recipe is. Two things must be here or the golden
# set is unfalsifiable — the baseline commit (without it, a reader cannot tell
# whether these goldens predate the successor work they police) and the exact
# extract flags (without them, "re-extract it" is not a reproducible
# instruction).
{
  echo "captured_from_commit=$(git rev-parse HEAD)"
  echo "captured_from_describe=$(git describe --always --dirty)"
  echo "vindex=$(basename "$E0_VINDEX")"
  echo "tokens=${TOKENS}"
  echo "walk_k=${WALK_K}"
  echo
  echo "# ── C1 regeneration recipe ──"
  echo "# rebuild the baseline binary:"
  echo "#   git checkout $(git rev-parse HEAD) && cargo build --release -p larql-cli"
  echo "# re-extract the vindex:"
  echo "model_source=${E0_MODEL:-<unrecorded — set E0_MODEL next capture>}"
  echo "model_revision=${E0_MODEL_REVISION:-<unrecorded — set E0_MODEL_REVISION next capture>}"
  echo "extract_flags=${E0_EXTRACT_FLAGS:-<unrecorded — set E0_EXTRACT_FLAGS next capture>}"
} > "${OUT_DIR}/PROVENANCE.txt"

# Every command row, driven from the shared corpus so capture and verify
# exercise an identical set.
while IFS= read -r row; do
  IFS='|' read -r -a parts <<< "$row"
  name="${parts[0]}"
  cmd=("${parts[@]:1}")
  printf '  → %-28s' "$name"
  e0_run_row "$name" "$E0_VINDEX" "$TMP" "${cmd[@]}" > "${OUT_DIR}/${name}.txt"
  echo "$(tail -n1 "${OUT_DIR}/${name}.txt")"
done < <(e0_rows "$E0_BIN" "$E0_VINDEX" "$TMP" "$TOKENS" "$WALK_K")

# index.json is the generation discriminator (spec §12.1).
printf '  → %-28s' index_version
e0_index_version "$E0_VINDEX" > "${OUT_DIR}/index_version.txt"
echo "ok"

echo
echo "captured $(find "$OUT_DIR" -name '*.txt' | wc -l | tr -d ' ') golden files into ${OUT_DIR}"
echo "COMMIT THESE. They are the only fixed record E0 has."
echo "Verify them later with: E0_BIN=... E0_VINDEX=... ./scripts/e0-verify-goldens.sh"
