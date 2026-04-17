# ADR 0001 — Python and LQL INFER Paths Must Be Byte-Identical

**Status:** Proposed
**Date:** 2026-04-17
**Depends on:** `larql-python`, `larql-lql`, `larql-inference`, `larql-vindex`

---

## Context

`larql` exposes two surfaces that run the same logical operation — a forward
pass through the walk FFN, with the `PatchedVindex` overlay and the L0 `KnnStore`
side-channel consulted at every layer that holds stored keys:

1. **LQL executor** — `SELECT ... INFER` / `INFER` via
   `larql-lql/src/executor/query/infer.rs`.
2. **Python binding** — `PyVindex::infer` in `larql-python/src/vindex.rs`.

The N=1000 KNN validation run on Gemma 3-4B uncovered that these two paths had
silently diverged:

- `PyVindex::infer` was running the walk FFN but **not** consulting
  `PatchedVindex.knn_store`. On a vindex with installed entries the Python API
  returned the pre-install top-1, while `SELECT ... INFER` on the same vindex
  returned the installed target. A user doing `v.infer(prompt)` after
  `v.insert(...)` saw stale predictions with no error.
- Even after the KNN lookup was added (2026-04-17), a second divergence remained:
  the LQL path calls `WalkFfn::new_unlimited_with_trace` — every feature at every
  layer — while `PyVindex::infer` defaults to `top_k_features=8192`. The LQL
  path's own in-file comment documents why: once a compose-mode INSERT lands at
  gate_scale ≈ 30, dropping features weakens the baseline disproportionately and
  the installed slot dominates every prompt. The Python default therefore
  produces different post-INSERT predictions from LQL on Gemma (16384 features)
  whenever compose-mode entries are present.

Both are divergences between surfaces that users reasonably expect to agree.
The first was a missing call; the second is a parameter default drift. The
underlying class is the same: **two code paths implementing the same
user-facing operation with no mechanical guarantee they stay in sync.**

The second divergence is still live as of this ADR.

## Decision

**The Python and LQL INFER paths MUST produce byte-identical predictions on any
vindex, for any prompt, at all N.**

Concretely:

1. **Single source of truth for the INFER pipeline.** The forward pass + KnnStore
   override logic lives in one place — `larql_inference` — as a function that
   takes `(weights, tokenizer, patched_vindex, prompt, top_k_predictions)` and
   returns a `Vec<(String, f64)>`. Both `executor::query::infer::exec_infer` and
   `PyVindex::infer` call this function. Neither implements the pipeline locally.
2. **No surface-specific defaults on load-bearing walk FFN parameters.** The
   KNN cosine threshold (`0.75`), `top_k_features` (unlimited when a KnnStore or
   compose-mode patch is present), and the set of layers consulted are chosen
   inside the shared function, not by the caller. Surfaces expose them as
   explicit overrides, not defaults.
3. **Parity test at every N tier.** A single integration test asserts that
   `PyVindex::infer(prompt)` and `Session::exec_infer(prompt)` return
   token-for-token identical top-k predictions on a vindex at N=0, N=50, N=200,
   N=1000. Runs in CI. Any future divergence — new feature added to one path,
   default changed on one path — fails this test before it ships.
4. **Other forward-pass surfaces fall under the same rule.** Whenever a third
   surface appears (`infer_trace`, `infer_stream`, a future gRPC server), it
   calls the shared function. The parity test expands to cover it.

## Consequences

**Positive.**

- A `v.infer()` result on a post-install vindex can never silently disagree with
  the LQL executor again. CI catches it.
- Tuning decisions (KNN threshold, feature cap) happen in exactly one file, so
  the reasoning documented in the LQL path's comments doesn't have to be
  duplicated — or worse, rediscovered — elsewhere.
- The N=1000 KNN ceiling claim in the LSM spec rests on a measured number that
  both surfaces reproduce. The result generalises to every consumer, not just
  the one that happened to be tested.

**Negative / cost.**

- Requires a refactor before the next Python release: move the KNN override
  block and the `new_unlimited_with_trace` selection out of both
  `exec_infer` and `PyVindex::infer` into a `larql_inference::infer_patched()`
  (or similar) entry point. Estimated < 150 LOC moved, no new logic.
- The parity test needs a vindex-with-weights fixture. **Resolved:** uses the
  v11 tiny-model vindex at `../tiny-model/model/v11/vindex` (auto-detected) or
  `V11_VINDEX_PATH` (explicit). 446 MB, 20 layers, real 26 K tokenizer —
  already maintained in the tiny-model repo, no duplication. CI runs the
  parity suite whenever tiny-model is checked out as a sibling.

**Not in scope.**

- `infer_trace` **now routes through `infer_patched`** as of the 2026-04-17
  refactor (previously returned pre-attention residuals via
  `predict_with_ffn_trace`, contradicting its own docstring). Return type
  changed to `list[(layer_index, PyArray1)]` so callers can't silently
  mis-index. See `docs/training-free-insert.md` for the updated usage example.
- MLX / chuk-lazarus bindings are downstream of `larql-python` and inherit
  parity for free as long as they route through `PyVindex::infer`.

## Open questions

- **Where should the shared entry point live?** `larql-inference` already owns
  `predict_with_ffn` and `WalkFfn`. Adding `infer_patched(&weights, &tokenizer,
  &PatchedVindex, prompt, top_k) -> Vec<(String, f64)>` there keeps the
  dependency graph clean: both `larql-lql` and `larql-python` already depend on
  `larql-inference`, and neither needs to learn about the other.
- **What exactly does "byte-identical" mean for floating-point probabilities?**
  Top-k *tokens* must match exactly. Probabilities should match to within
  f32 round-off (same ops in the same order ⇒ bitwise identical on the same
  platform). The parity test asserts token equality and `|p1 - p2| < 1e-6` on
  probabilities, which catches reorderings without flapping on harmless
  numerical noise.
- **Do we backport this to the Python release that shipped the KNN-less
  `infer`?** No — it was never tagged. The fix lands on `architecture-b`
  before the next Python release tag.
