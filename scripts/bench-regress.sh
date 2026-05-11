#!/usr/bin/env bash
# Bench regression detector — runs `benches/quant_matvec` against a saved
# baseline and exits non-zero if any cell regresses beyond `THRESHOLD`.
#
# Workflow:
#   1. On `main`, save a baseline:
#        scripts/bench-regress.sh save
#   2. On a feature branch / PR, compare against it:
#        scripts/bench-regress.sh check
#
# Catches the next 4× throughput cliff (the kind the q4_matvec_v4 row-drop
# bug caused) at PR time, not weeks later when goldens fail.
#
# Plug into CI: call `bash scripts/bench-regress.sh check` after
# `cargo test`. Exits 0 = clean, 1 = regression detected.

set -euo pipefail

BASELINE_NAME="${BASELINE_NAME:-main}"
THRESHOLD="${THRESHOLD:-0.10}"   # 10 % slowdown = regression
FEATURES="${FEATURES:---features metal}"
# Benches to gate on. Override with `BENCHES="quant_matvec"` to focus.
BENCHES="${BENCHES:-quant_matvec matmul linalg}"

cmd="${1:-check}"

run_all() {
    local mode=$1   # save | baseline
    for bench in $BENCHES; do
        echo "[bench-regress] -> $bench ($mode $BASELINE_NAME)"
        cargo bench -p larql-compute --bench "$bench" $FEATURES \
            -- "--$mode" "$BASELINE_NAME" 2>&1
    done
}

case "$cmd" in
    save)
        echo "[bench-regress] saving baseline '$BASELINE_NAME' across: $BENCHES"
        run_all save-baseline
        echo "[bench-regress] baseline saved under target/criterion/"
        ;;
    check)
        if [ ! -d "target/criterion" ]; then
            echo "[bench-regress] no baseline found at target/criterion/. \
Run '$0 save' on main first."
            exit 2
        fi
        echo "[bench-regress] checking against baseline '$BASELINE_NAME' \
(threshold=${THRESHOLD}, benches=$BENCHES)…"
        out=$(run_all baseline)
        echo "$out"
        if echo "$out" | grep -q "Performance has regressed"; then
            echo "[bench-regress] FAIL — regression detected vs baseline '$BASELINE_NAME'"
            exit 1
        fi
        echo "[bench-regress] OK — no regression vs baseline '$BASELINE_NAME'"
        ;;
    *)
        echo "usage: $0 {save|check}"
        echo "  save  — record current bench results as the baseline"
        echo "  check — run benches and fail if any cell regressed vs baseline"
        echo
        echo "env vars: BASELINE_NAME (default: main), THRESHOLD (default: 0.10),"
        echo "          FEATURES (default: --features metal)"
        exit 2
        ;;
esac
