#!/usr/bin/env bash
# Legs 7-8 (patch-chain-single-token), docs/specs/2026-07-16-larql-goose-toolcalling-design.md
# ADR-6. Chains single-token compose-mode INSERTs to attempt reconstructing a fixed
# tool-call-syntax template. This mechanism is EXPLICITLY unvalidated for this use case
# (residual K42/K43 — it is a one-entity-to-one-token factual editor, not a multi-token
# template synthesizer); this script's job is to run it for real and record what
# actually happens, not to assume success.
#
# Usage: run_patch_chain.sh <leg_id> <approach_id> <config> <larql_bin> <vindex_dir> <out_json>
set -u
leg_id="$1"; approach_id="$2"; config="$3"; larql_bin="$4"; vindex_dir="$5"; out_json="$6"

# Each chained install targets one atomic step of a fixed tool-call template. True
# per-subword-token chaining would need model-specific tokenizer introspection this
# script doesn't have; this is a deliberate, stated scoping simplification -- one
# INSERT per template STEP, not per literal token -- which still genuinely exercises
# the chained-install mechanism this leg is measuring.
steps_shared="tool_call_open opens_with <tool_call>"
steps_extended="tool_call_open opens_with <tool_call>
tool_call_name_key follows {\"name\":"

case "$config" in
  tool-open-tag)   steps="$steps_shared" ;;
  json-key-tokens) steps="$steps_extended" ;;
  *) echo "unknown config: $config" >&2; exit 1 ;;
esac

lql='USE "'"$vindex_dir"'";'
installed=0
total=0
while IFS=' ' read -r entity relation target; do
  [ -z "$entity" ] && continue
  total=$((total + 1))
  lql="$lql INSERT INTO EDGES (entity, relation, target) VALUES (\"$entity\", \"$relation\", \"$target\") AT LAYER 6 CONFIDENCE 0.8 MODE compose;"
done <<< "$steps"

out=$("$larql_bin" lql "$lql" 2>&1)
ec=$?
echo "$out"

if [ "$ec" -ne 0 ]; then
  outcome="error"
  detail="larql lql exited $ec: $(echo "$out" | tail -3 | tr '\n' ' ' | sed 's/"/\\"/g')"
else
  # Verify: walk the first step's canonical prompt and check the target token
  # actually surfaces in the trace, rather than trusting the INSERT's own exit code
  # (this project's own N1-style "no silent no-op reported as success" concern).
  walk_out=$("$larql_bin" lql 'USE "'"$vindex_dir"'"; WALK "The opens_with of tool_call_open is" TOP 10;' 2>&1)
  echo "$walk_out"
  if echo "$walk_out" | grep -qF "<tool_call>"; then
    installed=1
    outcome="installed"
    detail="target token found in post-insert WALK trace top-10"
  else
    outcome="partial"
    detail="INSERT(s) exited 0 but target token did not surface in post-insert WALK trace — chained install did not durably take, consistent with K43's unvalidated-for-this-use prediction"
  fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/write_leg_result.py" \
  --leg-id "$leg_id" --approach-id "$approach_id" --outcome "$outcome" --detail "$detail" \
  --extra "{\"steps_attempted\": $total}" --out "$out_json"
