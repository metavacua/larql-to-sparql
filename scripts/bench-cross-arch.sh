#!/usr/bin/env bash
# D-CROSS-PARITY — cross-architecture decode bench.
#
# Runs `larql bench` on a fixed model matrix (Gemma 3 4B, Gemma 4 31B
# dense, Llama 2 7B, Mistral 7B) and produces a combined per-arch
# report. Operationalises the model-agnosticity check from ADR-017:
# any kernel/dispatch change that's been A/B'd on Gemma 3 4B alone
# should be re-bench'd here before promotion.
#
# Also surfaces thermal artifacts: if EVERY arch in the matrix
# regresses simultaneously vs baseline, suspect thermal first
# (`feedback_thermal_perf_artifacts.md` memory entry).
#
# Usage:
#   ./scripts/bench-cross-arch.sh                  # bench all archs (default 30 tokens, 5 warmup)
#   ./scripts/bench-cross-arch.sh --save-baseline  # save current as baseline
#   ./scripts/bench-cross-arch.sh --compare        # compare to saved baselines
#
# Configure via env vars:
#   LARQL_BENCH_TOKENS         — decode tokens (default: 30)
#   LARQL_BENCH_WARMUP         — warmup tokens (default: 5)
#   LARQL_BENCH_PROMPT         — prompt (default: "The capital of France is")
#   LARQL_TOK_PER_S_THRESHOLD  — fail if tok/s drops below baseline × (1 - this). Default: 0.05
#
# Example with custom params:
#   LARQL_BENCH_TOKENS=15 LARQL_BENCH_WARMUP=3 ./scripts/bench-cross-arch.sh
#
# Exit codes:
#   0 — pass (or no baseline)
#   1 — regression detected
#   2 — script / bench error

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/target/release/larql"
LOCAL_CACHE="${HOME}/.cache/larql/local"
BASELINE_DIR="${REPO_ROOT}/bench/baselines/cross-arch"
TOKENS="${LARQL_BENCH_TOKENS:-30}"
WARMUP="${LARQL_BENCH_WARMUP:-5}"
PROMPT="${LARQL_BENCH_PROMPT:-The capital of France is}"
TOK_THRESHOLD="${LARQL_TOK_PER_S_THRESHOLD:-0.05}"

# Architecture matrix: (vindex-shorthand, family, hidden, layers, note).
# vindex resolves to ${LOCAL_CACHE}/${shorthand}.vindex; missing files are skipped.
#
# Gemma 4 E2B is a Per-Layer-Embeddings (PLE) arch.  D-METAL-PLE wired
# the PLE block onto Metal behind `LARQL_METAL_PLE=1`; with the env var
# unset, the bench still exercises the CPU fallback path so regressions
# in either backend surface here.
ARCH_MATRIX=(
  "gemma3-4b-q4k-v2|gemma3|2560|34|Gemma 3 4B (canonical Q4_K)"
  "gemma4-31b-q4k|gemma4|5376|60|Gemma 4 31B dense"
  "gemma4-e2b-q4k|gemma4|1536|35|Gemma 4 E2B (PLE — LARQL_METAL_PLE=1 for Metal)"
  "llama2-7b-q4k|llama|4096|32|Llama 2 7B"
  "mistral-7b-v0.1-q4k|mistral|4096|32|Mistral 7B v0.1 base"
)

# Optional MoE arch — included only if vindex is present (large file).
MOE_MATRIX=(
  "gemma4-26b-a4b|gemma4|5376|60|Gemma 4 26B A4B (MoE)"
)

if [ ! -x "${BIN}" ]; then
  echo "error: ${BIN} not found. Run 'cargo build --release --bin larql' first." >&2
  exit 2
fi

mkdir -p "${BASELINE_DIR}"

mode="report"
case "${1:-}" in
  --save-baseline) mode="save-baseline"; shift ;;
  --compare)       mode="compare";       shift ;;
  --help|-h)
    sed -n '/^# /p' "${BASH_SOURCE[0]}" | sed 's/^# //'
    exit 0 ;;
esac

run_one() {
  local short="$1"
  local family="$2"
  local hidden="$3"
  local layers="$4"
  local note="$5"
  local vindex="${LOCAL_CACHE}/${short}.vindex"
  local out_json="/tmp/cross-arch-${short}.json"

  if [ ! -d "${vindex}" ] && [ ! -L "${vindex}" ]; then
    printf "  %-32s  %-10s  (skipped: vindex not found)\n" "${short}" "${family}"
    return
  fi

  # Suppress per-stage stderr noise; capture only the JSON.
  "${BIN}" bench "${vindex}" \
    --tokens "${TOKENS}" \
    --warmup "${WARMUP}" \
    --prompt "${PROMPT}" \
    --output json \
    --output-file "${out_json}" \
    >/dev/null 2>&1 || {
      printf "  %-32s  %-10s  (bench failed)\n" "${short}" "${family}"
      return 1
    }

  python3 - "${out_json}" "${short}" "${family}" "${note}" <<'PY'
import json, sys
out_json, short, family, note = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(out_json) as f:
    data = json.load(f)
row = next((r for r in data['results'] if r['backend'] == 'larql-metal'), None)
if row is None:
    print(f"  {short:<32s}  {family:<10s}  (no larql-metal row)")
    sys.exit(0)
tps = row['tok_per_s']
mean_ms = row['ms_per_tok']['mean']
p99_ms = row['ms_per_tok']['p99']
n = row['n_steps']
print(f"  {short:<32s}  {family:<10s}  tok/s={tps:6.2f}  mean={mean_ms:6.2f}ms  p99={p99_ms:6.2f}ms  n={n:>3}  {note}")
PY
}

compare_one() {
  local short="$1"
  local current_json="/tmp/cross-arch-${short}.json"
  local baseline_json="${BASELINE_DIR}/${short}.json"

  if [ ! -f "${current_json}" ]; then
    return 0
  fi
  if [ ! -f "${baseline_json}" ]; then
    printf "  %-32s  (no baseline — run with --save-baseline)\n" "${short}"
    return 0
  fi

  python3 - "${baseline_json}" "${current_json}" "${TOK_THRESHOLD}" "${short}" <<'PY'
import json, sys
baseline_path, current_path, threshold_str, short = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
threshold = float(threshold_str)
with open(baseline_path) as f: bl = json.load(f)
with open(current_path) as f: cu = json.load(f)
b_row = next((r for r in bl['results'] if r['backend'] == 'larql-metal'), None)
c_row = next((r for r in cu['results'] if r['backend'] == 'larql-metal'), None)
if b_row is None or c_row is None:
    print(f"  {short:<32s}  (missing larql-metal in one of the JSONs)")
    sys.exit(0)
b_tps, c_tps = b_row['tok_per_s'], c_row['tok_per_s']
delta_pct = (c_tps - b_tps) / b_tps * 100.0
status = 'ok'
if delta_pct < -threshold * 100.0:
    status = 'REGRESSED'
print(f"  {short:<32s}  baseline={b_tps:6.2f}  current={c_tps:6.2f}  Δ={delta_pct:+6.2f}%  {status}")
PY
}

save_one() {
  local short="$1"
  local current_json="/tmp/cross-arch-${short}.json"
  local baseline_json="${BASELINE_DIR}/${short}.json"
  if [ -f "${current_json}" ]; then
    cp "${current_json}" "${baseline_json}"
    echo "  saved baseline: ${baseline_json}"
  fi
}

run_matrix() {
  local matrix=("$@")
  for entry in "${matrix[@]}"; do
    IFS='|' read -r short family hidden layers note <<< "${entry}"
    run_one "${short}" "${family}" "${hidden}" "${layers}" "${note}"
  done
}

echo "── Cross-architecture decode bench (D-CROSS-PARITY) ──"
echo "  prompt: \"${PROMPT}\""
echo "  warmup=${WARMUP}, tokens=${TOKENS}"
echo
echo "Dense archs:"
run_matrix "${ARCH_MATRIX[@]}"
echo
echo "MoE archs:"
run_matrix "${MOE_MATRIX[@]}"

if [ "${mode}" = "save-baseline" ]; then
  echo
  echo "── Saving baselines to ${BASELINE_DIR}/ ──"
  for entry in "${ARCH_MATRIX[@]}" "${MOE_MATRIX[@]}"; do
    IFS='|' read -r short _ _ _ _ <<< "${entry}"
    save_one "${short}"
  done
fi

if [ "${mode}" = "compare" ]; then
  echo
  echo "── Comparing to baselines (threshold: ${TOK_THRESHOLD} = $(echo "${TOK_THRESHOLD} * 100" | bc)%) ──"
  any_regressed=0
  for entry in "${ARCH_MATRIX[@]}" "${MOE_MATRIX[@]}"; do
    IFS='|' read -r short _ _ _ _ <<< "${entry}"
    output=$(compare_one "${short}")
    echo "${output}"
    if echo "${output}" | grep -q REGRESSED; then
      any_regressed=1
    fi
  done
  if [ "${any_regressed}" = "1" ]; then
    echo
    echo "── Tip: if EVERY arch regressed simultaneously, suspect thermal first."
    echo "   See feedback_thermal_perf_artifacts.md memory entry. Re-bench on a"
    echo "   cool machine before bisecting individual changes."
    exit 1
  fi
fi
