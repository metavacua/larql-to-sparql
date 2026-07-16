#!/usr/bin/env bash
# Legs 9-10 (patch-ensemble-trigger), docs/specs/2026-07-16-larql-goose-toolcalling-design.md
# ADR-6. Tests whether the balancer's single-canonical-prompt trigger condition (K42)
# generalizes to a CLASS of tool-requesting phrasings, by inserting from 5 varied
# phrasings' residuals — either into one shared (layer,feature) slot (leg 9) or one
# distinct slot per phrasing (leg 10) — then measuring what fraction of the 5
# phrasings actually elicit the edited behavior post-insert (not just whether the
# INSERT itself exited 0).
#
# Usage: run_patch_ensemble.sh <leg_id> <approach_id> <config> <larql_bin> <vindex_dir> <out_json>
set -u
leg_id="$1"; approach_id="$2"; config="$3"; larql_bin="$4"; vindex_dir="$5"; out_json="$6"

phrasings=(
  "Run a shell command to list files"
  "Execute ls in the terminal"
  "Use the shell tool to check the directory"
  "I need you to run a command"
  "Please invoke the shell to list contents"
)

lql='USE "'"$vindex_dir"'";'
i=0
for phrasing in "${phrasings[@]}"; do
  entity="tool_trigger_${i}"
  [ "$config" = "shared-slot" ] && entity="tool_trigger_shared"
  layer=6
  [ "$config" = "multi-layer-slots" ] && layer=$((6 + i))
  lql="$lql INSERT INTO EDGES (entity, relation, target) VALUES (\"$entity\", \"requests\", \"<tool_call>\") AT LAYER $layer CONFIDENCE 0.7 MODE compose;"
  i=$((i + 1))
done

out=$("$larql_bin" lql "$lql" 2>&1)
ec=$?
echo "$out"

hits=0
if [ "$ec" -eq 0 ]; then
  i=0
  for phrasing in "${phrasings[@]}"; do
    walk_out=$("$larql_bin" lql 'USE "'"$vindex_dir"'"; WALK "'"$phrasing"'" TOP 10;' 2>&1)
    if echo "$walk_out" | grep -qF "<tool_call>"; then
      hits=$((hits + 1))
    fi
    i=$((i + 1))
  done
fi

total=${#phrasings[@]}
if [ "$ec" -ne 0 ]; then
  outcome="error"
  detail="larql lql exited $ec"
elif [ "$hits" -eq 0 ]; then
  outcome="no_generalization"
  detail="0/$total phrasings triggered the edited feature — trigger did not generalize past install-time residuals, consistent with K43's prediction that gate vectors are single directions, not class boundaries"
elif [ "$hits" -eq "$total" ]; then
  outcome="full_generalization"
  detail="$hits/$total phrasings triggered — unexpectedly strong result for config=$config, worth a follow-up residual entry"
else
  outcome="partial_generalization"
  detail="$hits/$total phrasings triggered"
fi

python3 -c "
import json
json.dump({
  'leg_id': '$leg_id', 'approach_id': '$approach_id', 'outcome': '$outcome',
  'detail': '''$detail''', 'hits': $hits, 'total': $total,
}, open('$out_json', 'w'))
"
