#!/usr/bin/env bash
# LQL strategy-matrix runner — EMPIRICAL, NOT a validation suite.
#
# Runs every LQL command in a corpus against one already-extracted vindex and
# records RAW MECHANICAL OUTCOMES ONLY: exit code, wall time, stdout/stderr
# size + head, and a coarse mechanical bucket (ok / err / timeout). It makes
# NO judgement about whether output is "correct" — a break is data to read,
# not something to fix or interpret here. A failing cell never aborts the run.
#
# Usage:
#   run_matrix.sh <level> <vindex_dir> <corpus.jsonl> <out.jsonl>
# Env:
#   LARQL_BIN   path to larql binary            (default: target/release/larql)
#   MODEL_ID    model id/path for EXTRACT/USE MODEL cells (default: empty)
#   TMPROOT     scratch dir for cells that write outputs (default: mktemp)
#   WRAP        optional command prefix, e.g. containment on a dev box:
#               WRAP="larql-probe safe --mem 5600 --"   (default: empty -> CI)
#   CELL_TIMEOUT  per-command wall cap in seconds (default: 900)
set -uo pipefail

LEVEL="${1:?level}"; VINDEX="${2:?vindex dir}"; CORPUS="${3:?corpus.jsonl}"; OUT="${4:?out.jsonl}"
LARQL_BIN="${LARQL_BIN:-target/release/larql}"
MODEL_ID="${MODEL_ID:-}"
TMPROOT="${TMPROOT:-$(mktemp -d)}"
WRAP="${WRAP:-}"
CELL_TIMEOUT="${CELL_TIMEOUT:-900}"
mkdir -p "$TMPROOT"
: > "$OUT"

emit() { # emit one JSONL row via python (safe escaping); args are name=value pairs read from env
  python3 - "$@" <<'PY'
import json, sys
row = {}
for kv in sys.argv[1:]:
    k, _, v = kv.partition("=")
    # ints where sensible
    if k in ("exit_code","duration_ms","stdout_bytes","stderr_bytes"):
        try: v = int(v)
        except ValueError: pass
    row[k] = v
print(json.dumps(row))
PY
}

row_count=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$line")
  cat=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('cat',''))" "$line")
  lql=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['lql'])" "$line")
  # substitute placeholders
  lql=${lql//\{\{VINDEX\}\}/$VINDEX}
  lql=${lql//\{\{MODEL\}\}/$MODEL_ID}
  lql=${lql//\{\{TMP\}\}/$TMPROOT}

  so=$(mktemp); se=$(mktemp)
  start=$(date +%s%3N)
  # shellcheck disable=SC2086
  timeout "$CELL_TIMEOUT" $WRAP "$LARQL_BIN" lql "$lql" >"$so" 2>"$se"
  ec=$?
  end=$(date +%s%3N)
  dur=$((end - start))

  case "$ec" in
    0)   bucket=ok ;;
    124) bucket=timeout ;;
    137|139|134) bucket=crash ;;   # SIGKILL(OOM)/SIGSEGV/SIGABRT
    *)   bucket=err ;;
  esac
  sob=$(wc -c <"$so" | tr -d ' '); seb=$(wc -c <"$se" | tr -d ' ')
  sohead=$(head -c 800 "$so"); sehead=$(head -c 800 "$se")

  emit "level=$LEVEL" "id=$id" "cat=$cat" "lql=$lql" \
       "exit_code=$ec" "bucket=$bucket" "duration_ms=$dur" \
       "stdout_bytes=$sob" "stderr_bytes=$seb" \
       "stdout_head=$sohead" "stderr_head=$sehead" >> "$OUT"
  rm -f "$so" "$se"
  row_count=$((row_count + 1))
  echo "[$LEVEL] $id -> exit=$ec bucket=$bucket ${dur}ms" >&2
done < "$CORPUS"

echo "wrote $row_count rows to $OUT" >&2
