#!/usr/bin/env bash
# E0-FULL — replay the committed goldens against the CURRENT binary.
#
# The capture script writes the record. Until now nothing read it back, so the
# goldens were an assertion no one made: committed, provenance-stamped, and
# never compared to anything.
#
# This is the other half. It runs the same corpus (shared via
# `lib/e0-corpus.sh`, never redefined here) against a current build and diffs
# every row. Any difference is a behavioural change on a VINDEX2 path caused by
# VINDEX3 work sharing one binary — which is the entire claim E0 exists to
# police.
#
# SCOPE. This is E0-FULL, not E0-CI. It needs the multi-GB vindex, which is not
# committed because it is regenerable; the recipe to rebuild it is in the
# golden set's PROVENANCE.txt. E0-CI covers the weight-free generation-boundary
# subset and runs on every push.
#
# Usage:
#   E0_BIN=target/release/larql E0_VINDEX=/path/to/gemma4-26b-a4b.vindex \
#     ./scripts/e0-verify-goldens.sh
#
# Env vars:
#   E0_BIN       — larql binary to verify              (required)
#   E0_VINDEX    — the VINDEX2 index to exercise       (required)
#   E0_GOLDENS   — golden dir  (default: tests/goldens/e0/<vindex name>)
#   E0_TOKENS    — tokens to decode per prompt         (default: 24)
#   E0_WALK_K    — WALK top-K per layer                (default: 20)
#   E0_DIFF      — set to 1 to print full diffs rather than a summary
#
# Exit status is the result: 0 = every row matched, 1 = at least one differed.

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/e0-corpus.sh
source "${SCRIPT_DIR}/lib/e0-corpus.sh"

: "${E0_BIN:?set E0_BIN to the larql binary under test}"
: "${E0_VINDEX:?set E0_VINDEX to a VINDEX2 directory}"

readonly GOLDEN_DIR="${E0_GOLDENS:-tests/goldens/e0/$(basename "$E0_VINDEX")}"

# Replay parameters come from the record itself.
#
# The capture wrote `tokens` and `walk_k` into PROVENANCE.txt precisely because
# they change the output: replaying 24 tokens against a 16-token golden differs
# on every decode row, and the diff would read as a behavioural regression
# rather than as a harness mismatch. Defaulting instead of reading would make
# E0-FULL fail loudly for the wrong reason — the most expensive kind of false
# positive, because it looks exactly like the thing being tested for.
provenance_field() {
  local field="$1" default="$2"
  local file="${GOLDEN_DIR}/PROVENANCE.txt"
  [[ -f "$file" ]] || { echo "$default"; return; }
  local value
  value="$(grep "^${field}=" "$file" 2>/dev/null | cut -d= -f2-)"
  echo "${value:-$default}"
}

readonly TOKENS="${E0_TOKENS:-$(provenance_field tokens 24)}"
readonly WALK_K="${E0_WALK_K:-$(provenance_field walk_k 20)}"
readonly TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ ! -d "$GOLDEN_DIR" ]]; then
  echo "no goldens at ${GOLDEN_DIR}" >&2
  echo "capture them first with scripts/e0-capture-goldens.sh" >&2
  exit 2
fi

echo "E0-FULL golden verification"
echo "  binary:  ${E0_BIN}"
echo "  vindex:  ${E0_VINDEX}"
echo "  goldens: ${GOLDEN_DIR}"
if [[ -f "${GOLDEN_DIR}/PROVENANCE.txt" ]]; then
  echo "  baseline: $(grep '^captured_from_describe=' "${GOLDEN_DIR}/PROVENANCE.txt" | cut -d= -f2-)"
fi
echo "  replay:  tokens=${TOKENS} walk_k=${WALK_K} (from the golden's provenance)"
echo

matched=0
differed=0
missing=0
declare -a failures=()

# Canonicalise a stored golden for comparison.
#
# Goldens for unordered rows were captured before the ordering fix, so they
# hold whatever permutation the incumbent binary happened to emit. Sorting the
# stored side too lets an existing golden remain usable instead of requiring a
# re-capture to test a property that was never about order. The trailing
# `exit_status=` line is held back so a sorted row still ends with its status.
canonical_golden() {
  local name="$1" golden="$2" out="$3"
  if e0_row_is_unordered "$name"; then
    sed '$d' "$golden" | sort > "$out"
    tail -n1 "$golden" >> "$out"
  else
    cp "$golden" "$out"
  fi
}

check_row() {
  local name="$1" actual="$2"
  local golden_raw="${GOLDEN_DIR}/${name}.txt"
  printf '  %-28s' "$name"
  if [[ ! -f "$golden_raw" ]]; then
    echo "NO GOLDEN"
    missing=$((missing + 1))
    return
  fi
  local golden="${TMP}/${name}.golden"
  canonical_golden "$name" "$golden_raw" "$golden"
  if diff -q "$golden" "$actual" >/dev/null 2>&1; then
    echo "match"
    matched=$((matched + 1))
    return
  fi
  echo "DIFFERS"
  differed=$((differed + 1))
  failures+=("$name")
  if [[ "${E0_DIFF:-0}" == "1" ]]; then
    diff -u "$golden" "$actual" | sed 's/^/      /'
  fi
}

# ── Every command row ──────────────────────────────────────────────────────
while IFS= read -r row; do
  # Fields are '|'-separated: name, then the command and its arguments. Split
  # on '|' rather than whitespace because prompts contain spaces.
  IFS='|' read -r -a parts <<< "$row"
  name="${parts[0]}"
  cmd=("${parts[@]:1}")
  e0_run_row "$name" "$E0_VINDEX" "$TMP" "${cmd[@]}" > "${TMP}/${name}.actual"
  check_row "$name" "${TMP}/${name}.actual"
done < <(e0_rows "$E0_BIN" "$E0_VINDEX" "$TMP" "$TOKENS" "$WALK_K")

# ── The generation discriminator ───────────────────────────────────────────
e0_index_version "$E0_VINDEX" > "${TMP}/index_version.actual"
check_row index_version "${TMP}/index_version.actual"

# ── Verdict ────────────────────────────────────────────────────────────────
echo
echo "  matched  ${matched}"
echo "  differed ${differed}"
[[ "$missing" -gt 0 ]] && echo "  missing  ${missing}  (golden absent — capture is stale)"

if [[ "$differed" -eq 0 && "$missing" -eq 0 ]]; then
  echo
  echo "E0-FULL PASS — no observable change on any VINDEX2 path."
  exit 0
fi

echo
echo "E0-FULL FAIL — VINDEX2 behaviour changed in: ${failures[*]:-<none>}"
echo "Re-run with E0_DIFF=1 to see the differences."
echo "If a change is intentional, re-capture deliberately and say why in the commit."
exit 1
