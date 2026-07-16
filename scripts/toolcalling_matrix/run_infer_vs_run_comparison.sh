#!/usr/bin/env bash
# Residual Q5: does larql run's cached-KV-decode path (what larql chat / our
# Goose backend actually uses) diverge from larql lql INFER (full re-forward
# per step) on identical input for SmolLM2-135M-Instruct? Precedent:
# chrishayuk/larql#177 found exactly this divergence on gemma-4 (root-caused
# to that model's heterogeneous per-layer KV geometry, which does not apply
# to SmolLM2's uniform architecture) -- this leg checks whether an
# unrelated cached-decode bug is present for OUR model, independent of and
# prior to the chat-template/decoding-strategy hypotheses (docs/specs/
# 2026-07-16-larql-goose-toolcalling-research-residual.md, D section).
#
# `run`'s primary comparison forces LARQL_RAW_PROMPT=1 so both sides tokenize
# the IDENTICAL string. Without this, `run` (via run_predict_inner's
# chat-template wiring) sends a ChatML-wrapped prompt while `lql INFER` (no
# chat-template code path at all -- confirmed absent from larql-lql) sends
# the raw string; a run against the first CI execution of this leg found
# outputs differing under exactly that confound and it was misreported as
# a possible cached-decode-path bug. Holding the prompt constant isolates
# the actual question this leg exists to answer.
#
# Usage: run_infer_vs_run_comparison.sh <leg_id> <approach_id> <config> <larql_bin> <vindex_dir> <out_json>
set -u
leg_id="$1"; approach_id="$2"; config="$3"; larql_bin="$4"; vindex_dir="$5"; out_json="$6"

PROMPT="The capital of France is"

run_out=$(LARQL_RAW_PROMPT=1 "$larql_bin" run "$vindex_dir" "$PROMPT" -n 5 2>&1)
run_ec=$?
echo "=== larql run (cached KV-decode, LARQL_RAW_PROMPT=1 to match INFER's untemplated input) ==="
echo "$run_out"

infer_out=$("$larql_bin" lql "USE \"$vindex_dir\"; INFER \"$PROMPT\" TOP 5;" 2>&1)
infer_ec=$?
echo "=== larql lql INFER (full re-forward) ==="
echo "$infer_out"

# Informational only, does not affect outcome: the chat-templated `run`
# behavior (what larql chat / the Goose backend actually invokes by
# default), captured separately so a degenerate echo-completion under
# templating isn't silently lost, but also never confused with the
# decode-path-parity question above.
templated_run_out=$("$larql_bin" run "$vindex_dir" "$PROMPT" -n 5 2>&1)
templated_run_ec=$?
echo "=== larql run (cached KV-decode, default chat-templated prompt -- informational, not part of the parity verdict) ==="
echo "$templated_run_out"

if [ "$infer_ec" -ne 0 ] && echo "$infer_out" | grep -qi "requires model weights"; then
  outcome="inconclusive_needs_all_level"
  detail="INFER refused: this vindex's has_model_weights legacy field is unset at --level inference, even though has_inference_weights() (the modern check) should honor this level per the codebase's own test (config/index.rs has_inference_weights_handles_legacy_and_modern_flags) -- infer.rs:77 checks the raw legacy field directly, not the method. Re-extracting at --level all (or --include-weights) would be needed to actually run this comparison; not done in this leg to avoid an unplanned re-extraction cost. This is itself a real finding, not just a blocker: the raw-field check appears out of sync with the method the codebase's own tests treat as authoritative."
elif [ "$run_ec" -ne 0 ] || [ "$infer_ec" -ne 0 ]; then
  outcome="error"
  detail="run exit=$run_ec, infer exit=$infer_ec -- see uploaded console output for both raw commands"
else
  # Extract just the first predicted/generated token from each for a direct,
  # minimal comparison (matches how chrishayuk/larql#177 characterized its
  # own divergence: top-1 parity, not full-text equality).
  run_first_word=$(echo "$run_out" | tr -s ' \n' '\n' | grep -v '^$' | head -1)
  infer_first_tok=$(echo "$infer_out" | grep -A1 "Predictions" | tail -1 | awk '{print $2}')
  if [ "$run_first_word" = "$infer_first_tok" ]; then
    outcome="no_divergence"
    detail="with prompt-treatment held constant (LARQL_RAW_PROMPT=1), run's first generated token ('$run_first_word') matches INFER's top-1 prediction ('$infer_first_tok') -- no evidence of a cached-decode-path bug independent of chat-template/decoding-strategy hypotheses for this prompt"
  else
    outcome="divergence_found"
    detail="with prompt-treatment held constant (LARQL_RAW_PROMPT=1), run's first generated token ('$run_first_word') does NOT match INFER's top-1 prediction ('$infer_first_tok') -- likely a genuine cached-decode-path bug independent of Q1/Q2 (chat-template/decoding-strategy); escalate to Root-Cause Analysis before trusting either uncommitted fix's own test results"
  fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/write_leg_result.py" \
  --leg-id "$leg_id" --approach-id "$approach_id" --outcome "$outcome" --detail "$detail" \
  --extra "{\"run_exit\": $run_ec, \"infer_exit\": $infer_ec, \"templated_run_exit\": $templated_run_ec}" --out "$out_json"
