#!/usr/bin/env bash
# Legs 11-12 (introspect-then-patch), docs/specs/2026-07-16-larql-goose-toolcalling-design.md
# ADR-6. Leg 11 (walk-rank-only): runs a labeled contrastive prompt set (valid vs
# invalid tool-call completions) through WALK, ranking features by how often they
# fire on the "valid" side vs the "invalid" side — discovery only, no patch applied.
# Leg 12 (circuit-ablate-apply): additionally runs circuit-discover for real
# OV-to-gate clustering. The ablation (`larql dev ov-rd`) + overlay-apply half of this
# leg is NOT YET IMPLEMENTED this phase — ov-rd is explicitly experimental and scoped
# to one research line per its own README (residual K44), and this script says so
# plainly rather than fabricating unverified ablation-flag syntax for a step that
# would not actually run the described technique.
#
# Usage: run_introspect_patch.sh <leg_id> <approach_id> <config> <larql_bin> <vindex_dir> <out_json>
set -u
leg_id="$1"; approach_id="$2"; config="$3"; larql_bin="$4"; vindex_dir="$5"; out_json="$6"

valid_prompts=(
  "Run the shell command to list files:"
  "Execute this command and show the output:"
)
invalid_prompts=(
  "The weather today is"
  "My favorite color is"
)

declare -A valid_features
declare -A invalid_features

collect_features() {
  local prompt="$1"
  "$larql_bin" lql 'USE "'"$vindex_dir"'"; WALK "'"$prompt"'" TOP 10;' 2>&1 \
    | grep -oE 'F[0-9]+' | grep -oE '[0-9]+' | sort -u
}

for p in "${valid_prompts[@]}"; do
  for feat in $(collect_features "$p"); do
    valid_features["$feat"]=$(( ${valid_features["$feat"]:-0} + 1 ))
  done
done
for p in "${invalid_prompts[@]}"; do
  for feat in $(collect_features "$p"); do
    invalid_features["$feat"]=$(( ${invalid_features["$feat"]:-0} + 1 ))
  done
done

ranked_count=0
top_feature=""
top_score=-999
for feat in "${!valid_features[@]}"; do
  v=${valid_features[$feat]:-0}
  n=${invalid_features[$feat]:-0}
  score=$((v - n))
  ranked_count=$((ranked_count + 1))
  if [ "$score" -gt "$top_score" ]; then
    top_score=$score
    top_feature=$feat
  fi
done

if [ "$ranked_count" -eq 0 ]; then
  outcome="inconclusive"
  detail="no features observed across either prompt set — WALK produced no parseable feature IDs (see raw output above for the actual reason)"
else
  outcome="ranked"
  detail="ranked $ranked_count candidate feature(s) by (valid-side hits - invalid-side hits); top candidate: feature $top_feature (score $top_score). No causal ablation or patch was applied this phase (leg 11 is explicitly discovery-only)."
fi

if [ "$config" = "circuit-ablate-apply" ]; then
  circuit_out=$("$larql_bin" circuit-discover "$vindex_dir" -k 20 --min-coupling 0.5 2>&1)
  circuit_ec=$?
  echo "$circuit_out"
  if [ "$circuit_ec" -eq 0 ]; then
    circuit_note="circuit-discover ran successfully (OV-to-gate clustering, real output above)."
  else
    circuit_note="circuit-discover exited $circuit_ec."
  fi
  detail="$detail $circuit_note The causal-ablation (larql dev ov-rd) + PatchedVindex-overlay-apply half of this leg is NOT YET IMPLEMENTED this phase — ov-rd is experimental/scoped to one research line (residual K44); reported honestly rather than fabricated."
  outcome="${outcome}_partial_pipeline"
fi

python3 -c "
import json
json.dump({
  'leg_id': '$leg_id', 'approach_id': '$approach_id', 'outcome': '$outcome',
  'detail': '''$detail''', 'ranked_feature_count': $ranked_count,
}, open('$out_json', 'w'))
"
