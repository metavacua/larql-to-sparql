#!/bin/bash
# Bracketed A/B for the KV attention ladder: baseline / candidate / baseline.
#
#   ./scripts/kv-ladder-bracket.sh B bench/prompts/gpt-oss-kv-ladder-b.txt 300
#
# The unit of measurement is the bracket, not the arm. A candidate counts
# only if the two baselines that surround it agree; if they disagree the
# machine changed underneath the block and the block is void. Do NOT
# average across a disagreement. See docs/kv-attention-scaling.md
# §Run hygiene for the measured cases that motivate each precondition.
#
# This script enforces the two preconditions that are arithmetic rather
# than environmental:
#
#   6. the two brackets agree within BRACKET_TOL_PCT
#   7. every arm completed its full decode budget
#
# Precondition 7 is the one a human reader misses. An arm that stops early
# still prints a perfectly plausible row, with a mean over however few
# steps it took, and at ~2K-token prompts gpt-oss does this intermittently
# (issue #229). A truncated CANDIDATE is invisible to the bracket check —
# both baselines still agree with each other, so precondition 6 passes and
# the block certifies a mean over a handful of steps. Hence the explicit
# step-count gate here.
#
# The remaining preconditions are environmental and this script cannot
# check them: rated adapter wattage, an explicitly warmed page cache,
# warm-to-plateau, and exclusivity by handshake with every peer session.

set -u

MODEL="${LARQL_LADDER_MODEL:-/Users/christopherhay/chris-models/gpt-oss-20b-q4k.vindex}"
ROUTED="${LARQL_LADDER_ROUTED:-/Users/christopherhay/chris-models/gpt-oss-20b-experts-mxfp4.v3}"
BIN="${LARQL_LADDER_BIN:-./target/release/larql}"

# Pinned by bench/prompts/README.md; changing either changes the number.
WARMUP=16
TOKENS=256
# A healthy run reports TOKENS-1 steps: the first token comes from
# prefill's logits and is never counted in decode_ms. Anything below that
# is a real early stop.
MIN_STEPS=$((TOKENS - 1))
BRACKET_TOL_PCT=1.0

LABEL="${1:?usage: kv-ladder-bracket.sh <label> <prompt-file> [rest-seconds]}"
PROMPT_FILE="${2:?usage: kv-ladder-bracket.sh <label> <prompt-file> [rest-seconds]}"
REST="${3:-300}"

[ -r "$PROMPT_FILE" ] || { echo "no such prompt file: $PROMPT_FILE" >&2; exit 2; }
[ -x "$BIN" ] || { echo "no larql binary at $BIN (cargo build --release)" >&2; exit 2; }
PROMPT="$(cat "$PROMPT_FILE")"

# Measure the DEFAULT, not LARQL_KV_SEQPAR=auto: the shipping question is
# whether an unset env fires the policy, which is a different code path.
run_arm() {
  if [ "$1" = off ]; then
    ENVV=(env LARQL_KV_SEQPAR=off)
  else
    ENVV=(env -u LARQL_KV_SEQPAR)
  fi
  "${ENVV[@]}" LARQL_GPU_ROUTE=1 "$BIN" bench "$MODEL" --routed-from "$ROUTED" \
    --prompt "$PROMPT" --warmup "$WARMUP" -n "$TOKENS" 2>&1 \
    | awk '/^ *larql-metal/{
        note=""; for (i=7; i<=NF; i++) note = note $i " ";
        gsub(/ms$/, "", $3);
        print $3, $6, note
      }'
}

echo "════ BLOCK $LABEL   bracketed   ${REST}s rest before every arm"
echo "     prompt $PROMPT_FILE (${#PROMPT} chars)   warmup $WARMUP   n $TOKENS"

MEANS=(); STEPS=(); VOID=0
for arm in off default off; do
  sleep "$REST"
  read -r mean steps note <<<"$(run_arm "$arm")"
  if [ -z "${mean:-}" ]; then
    echo "  $(printf '%-8s' "$arm") NO ROW — arm failed to produce a reading"
    VOID=1; continue
  fi
  flag=""
  if [ "${steps:-0}" -lt "$MIN_STEPS" ]; then
    flag="  ← TRUNCATED (precondition 7): $steps < $MIN_STEPS steps"
    VOID=1
  fi
  printf "  %-8s mean %8s ms   steps %5s   %s%s\n" "$arm" "$mean" "$steps" "${note:-}" "$flag"
  MEANS+=("$mean"); STEPS+=("$steps")
done

if [ "${#MEANS[@]}" -ne 3 ]; then
  echo "  VERDICT: VOID — block did not produce three arms"
  exit 1
fi

awk -v o1="${MEANS[0]}" -v c="${MEANS[1]}" -v o2="${MEANS[2]}" \
    -v tol="$BRACKET_TOL_PCT" -v void="$VOID" '
BEGIN {
  spread = (o1 > o2 ? o1 - o2 : o2 - o1) / (o1 < o2 ? o1 : o2) * 100
  printf "  brackets %.2f%% apart (tolerance %.1f%%)\n", spread, tol
  if (void) {
    print "  VERDICT: VOID — an arm did not complete its decode budget"
    exit 1
  }
  if (spread > tol) {
    print "  VERDICT: VOID — brackets disagree; do NOT average across them"
    exit 1
  }
  base = (o1 + o2) / 2
  printf "  VERDICT: VALID — candidate %.2f vs baseline %.2f ms", c, base
  printf "  (%+.1f%% latency, %+.1f%% throughput)\n", (c - base) / base * 100, (base / c - 1) * 100
}'
