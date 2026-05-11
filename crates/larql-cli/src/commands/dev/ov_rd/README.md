# OV/RD Dev Command

`larql dev ov-rd` is the experimental harness for attention output-vector
rate-distortion work. It is deliberately a `dev` command, not a production
extraction command.

The core question is whether an attention head's pre-`W_O` output can be
replaced by a compact table:

```text
runtime state -> address -> residual-space lookup/add
```

For the current L0H6 line of work, the stable findings are:

```text
oracle table exists
Mode D residual-table materialization works
held-out mean/p95 can pass
the current dominant group-0 code is not addressable from shallow state
full/reduced-QK attention-pattern clusters also fail on the hard L0H6 group
```

## Engine Boundary

The main engine now owns the reusable runtime pieces that were previously
embedded in this command:

```text
larql_inference::vindex::insert_q4k_layer_tensors
larql_inference::vindex::remove_layer_tensors
larql_inference::vindex::predict_q4k_hidden_hooked
larql_inference::vindex::predict_q4k_hidden_with_mapped_pre_o_head
larql_inference::vindex::predict_q4k_hidden_with_replaced_pre_o_head
larql_inference::vindex::predict_q4k_hidden_with_zeroed_pre_o_heads
larql_inference::vindex::predict_q4k_hidden_with_subtracted_pre_o_heads
larql_inference::vindex::predict_q4k_hidden_with_mapped_head_residual_delta
larql_inference::vindex::predict_q4k_hidden_with_replaced_head_residual_delta
larql_inference::vindex::predict_q4k_hidden_with_original_head_residual_delta
larql_inference::attention::run_attention_block_with_pre_o_and_all_attention_weights
larql_inference::attention::run_attention_block_with_pre_o_and_reduced_qk_attention_weights
```

Those APIs preserve the hard runtime invariants:

```text
Q4K layer tensor scope
PLE input propagation
Gemma 4 shared-KV routing
FFN / PLE / layer-scalar tail
target-layer intervention ordering
```

OV/RD code should use those APIs whenever it is evaluating a full-model
intervention. Do not reimplement the full Q4K layer loop in the command unless
the command is collecting intermediate training/capture data that the engine API
does not expose yet.

## What Belongs Here

Keep Rust code here when it needs exact model/vindex behavior:

- experiment-specific Q4K vindex loading and prompt orchestration
- attention `pre_W_O` capture for fitting/statistics passes
- `W_O`-visible projection and roundtrip checks
- oracle low-rank and PQ reconstruction
- direct residual-edit catalogue diagnostics
- base-PQ-plus-exception residual catalogue diagnostics
- Mode D residual-delta table materialization
- final-logit KL/top-k evaluation through the real forward path
- model-native discrete address probes whose inputs are already produced by a
  real forward pass, for example previous-layer FFN top-feature IDs and
  attention/relation summaries or learned attention-pattern cluster IDs
- targeted majority/stratum controls for selected PQ groups, so scale-up
  diagnostics do not need full 48-group importance sweeps
- balanced Stage-0 capture subsets via `capture --max-per-stratum`, so grouped
  prompt files can be sampled without creating one-off JSONL fixtures
- W_O-visible Stage-0 ranking controls, for example
  `zero-ablate --stage0-rank wo-visible-variance`, so Gate 1 promotes heads by
  residual-space impact rather than raw pre-W_O variance when available
- canonical JSON artifacts that other tools consume

The command should remain an orchestrator plus faithful runtime validator. It
should not become the place where every new probe, plot, or clustering variant
lives.

## What Should Move To Python

Use Python over exported artifacts for fast-changing analysis:

- code stability tables
- plotting and report tables
- window hashes, bag-of-token hashes, shingling, MinHash
- decision trees, nearest-centroid variants, and classifier sweeps
- feature/code correlation scans

If a Python probe becomes a serious runtime candidate, reimplement only that
candidate in Rust after its artifact contract is clear.

Small summary diagnostics that are part of the canonical JSON schema can stay in
Rust. For example, entropy/JS divergence helpers belong in `metrics.rs` if they
are emitted by `oracle-pq`, while broader exploratory scans should use Python
against exported artifacts.

## Artifact Contract

Rust should export enough canonical state that Python can iterate without
rerunning full model forward passes for every idea:

```text
prompt id / stratum / tokens
layer-input residual rows
captured pre-W_O head rows
oracle PQ codes by position
baseline and replacement logits or metrics
per-prompt KL/top-k summaries
```

Prefer compact binary arrays plus JSON metadata for large matrices. JSON alone
is fine for summaries and small diagnostics.

## Documentation Boundary

Use `~/chris-source/chris-experiments/shannon/38_ov_rate_distortion/RESULTS.md` as the lab notebook: commands
run, artifacts written, negative results, and interpretation.

When a result becomes architectural rather than experimental, promote it to a
short stable doc under `docs/`, for example:

```text
docs/attention-tableability.md
```

The experiment log should stay detailed and chronological. The docs should be
short, curated, and claim-focused.

## Current Refactor Direction

This directory replaced the old single-file
`commands/extraction/ov_rd_cmd.rs`. The command is now under `dev` because these
runs are experimental probes, not stable vindex extraction verbs.

Current split:

```text
cmd.rs             CLI dispatch only
address.rs         address predictor models and address-match helpers
basis.rs           W_O roundtrip basis, z-space PCA fitting, and eigensolver
capture.rs         stage-0 pre-W_O capture and head statistics
input.rs           prompt loading, held-out splits, and CLI string parsers
metrics.rs         KL, entropy, top-k, and distribution helpers
oracle.rs          roundtrip and low-rank oracle checks
edit_catalog.rs    full-vector residual-edit catalogue diagnostics in hidden/PCA space
gamma_address.rs   gamma-aligned supervised address probes over raw layer input,
                  diagonal-affine projections toward later residual states,
                  fixed random low-rank projections, and learned low-rank
                  target-residual bridges
oracle_pq.rs       PQ experiment orchestration, address probe evaluation, and
                  direct code-level rule diagnostics
oracle_pq_address.rs
                  address-probe, previous-FFN feature-key, FFN-first feature-key,
                  attention-relation-key, full/reduced-QK attention-cluster-key,
                  code-substitution/coarsening controls, code-occurrence export,
                  oracle binary code/default upper bounds, class-collapse
                  behavioral quotient probes, and majority-code fitting
oracle_pq_eval.rs  shared predicted-address evaluation helper
oracle_pq_fit.rs   PQ codebook fitting
oracle_pq_forward.rs
                  PQ/Mode-D model calls plus experiment-specific capture/mapping logic
oracle_pq_mode_d.rs
                  Mode D residual-table materialization helpers
oracle_pq_reports.rs
                  PQ/address report accumulators
oracle_pq_stability.rs
                  PQ code distribution stability diagnostics
pq.rs              PQ codebooks, Mode D tables, and k-means mechanics
pq_exception.rs    base-PQ-plus-exception residual catalogue diagnostics, with
                  residual-error/prompt-KL/position-restore-KL/CE tail
                  selectors and k-means/exemplar fits
reports.rs         JSON artifact schemas
runtime.rs         thin shim over inference Q4K tensor insertion/removal
sanity.rs          no-op/subtract/residual-delta equivalence checks
static_replace.rs  static mean replacement gate and shared static fitting
stats.rs           running head stats and static mean accumulators
types.rs           shared input/config identifiers
zero_ablate.rs     zero pre-W_O ablation gate
```

Remaining CLI-owned tensor-scope loops are mostly fitting/capture passes:

```text
capture.rs                stage-0 statistics
basis.rs                  W_O/PCA basis fitting
static_replace.rs         static mean fitting pass
oracle_pq_fit.rs          PQ training rows
oracle_pq_address.rs      layer-input residual capture for address probes
oracle_pq_stability.rs    code stability diagnostics
oracle_pq_mode_d.rs       Mode D table materialization
```

Those may move later if they become generally useful capture APIs, but they are
not production forward paths. Do this incrementally. The first invariant is that
existing `larql dev ov-rd` commands keep their behavior and artifact schema
unless a schema change is intentional and documented in the experiment results.
