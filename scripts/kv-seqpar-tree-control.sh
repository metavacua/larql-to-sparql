#!/usr/bin/env bash
# Which tree owns the process-scoped startup fault (#229 parent)?
# One fresh process per row; arms interleaved. Integrity only — not timing.
#   ARMS="off auto"   LARQL_KV_SEQPAR values; "unset" = binary default; "VAR=VALUE" = any env arm
#   HANG_SECS=300     a process alive past this is a HANG row and is killed
set -u
MODEL="${LARQL_LADDER_MODEL:-/Users/christopherhay/chris-models/gpt-oss-20b-q4k.vindex}"
ROUTED="${LARQL_LADDER_ROUTED:-/Users/christopherhay/chris-models/gpt-oss-20b-experts-mxfp4.v3}"
BIN="${LARQL_LADDER_BIN:-./target/release/larql}"
ARMS="${ARMS:-off auto}"
HANG_SECS="${HANG_SECS:-300}"
TOKENS="${TOKENS:-8}"
PROMPT_FILE="${1:-bench/prompts/gpt-oss-kv-ladder-c.txt}"
ROUNDS="${2:-8}"
OUT="${3:-/dev/stdout}"
PROMPT="$(cat "$PROMPT_FILE")"
for r in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    case "$arm" in
      unset) ENVV=(env -u LARQL_KV_SEQPAR) ;;      # the binary's own default
      *=*)   ENVV=(env "$arm") ;;                  # any VAR=VALUE arm, e.g. LARQL_FUSED_DECODE_HEAD=0
      *)     ENVV=(env LARQL_KV_SEQPAR="$arm") ;;  # off | auto | <n>
    esac
    tmp=$(mktemp)
    "${ENVV[@]}" LARQL_GPU_ROUTE=1 "$BIN" bench "$MODEL" --routed-from "$ROUTED" \
      --prompt "$PROMPT" --warmup 0 -n "$TOKENS" >"$tmp" 2>&1 &
    pid=$!
    waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$HANG_SECS" ]; do sleep 2; waited=$((waited+2)); done
    if kill -0 "$pid" 2>/dev/null; then
      /usr/bin/sample "$pid" 1 -file "${OUT%.log}.hang-r${r}-${arm}.sample.txt" >/dev/null 2>&1
      kill -9 "$pid"; wait "$pid" 2>/dev/null
      row="HANG >${HANG_SECS}s pid=$pid (killed; stack sample beside log)"
    else
      wait "$pid"; rc=$?
      row=$(grep -E '^ *larql-metal' "$tmp" | sed 's/^ *//')
      [ -z "$row" ] && row="NO-ROW rc=$rc: $(grep -iE 'error|panic' "$tmp" | head -1)"
    fi
    # Keep every process's raw output beside the log: a catastrophic row
    # looks like a healthy one on the row alone, and stderr diagnostics
    # (e.g. `[metal] command buffer ... status Error`) live only here.
    if [ "$OUT" != /dev/stdout ]; then
      cp "$tmp" "${OUT%.log}.raw-r${r}-$(echo "$arm" | tr '=/' '--').txt" 2>/dev/null
    fi
    rm -f "$tmp"
    printf '%s round=%d arm=%-26s %s\n' "$(date +%H:%M:%S)" "$r" "$arm" "$row" | tee -a "$OUT"
  done
done
