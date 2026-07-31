# Proposal: Canonical Vindex Extraction Pipeline

## Status: in-flight

## Why

Vindexes today are coordinate-system-relative: gate vectors live in whatever
basis the model trainer used, with no formal invariants beyond format
correctness. Cross-model comparison, formal verification, and
`wasm32v1-none` inference all require a canonical form with provable uniqueness
properties.

The flat-canonical (semi-intrinsic, single G) form is the smallest useful
canonical vindex: it requires only data already present in a browse-level
vindex (`embeddings.bin`, `gate_vectors.bin`, `down_meta.bin`) and produces a
`canonical_meta.json` sidecar that is a pure function of the model weights.

## What

Add `larql canonicalize <vindex-path>` that writes `canonical_meta.json` into
an existing vindex directory. The command:

1. Estimates the activation covariance G from token embeddings (semi-intrinsic
   calibration — no external corpus).
2. Computes the Cholesky factor L of G (the whitening operator).
3. Scores each feature's "on-shell" property using the c_score percentile from
   existing down_meta.
4. Classifies each layer's activation regime (Wave / Particle / Wavelet).
5. Writes all results to `canonical_meta.json` as a pure function of the
   existing vindex files.

## Non-goals (this change)

- Writing whitened gate vectors (`gate_vectors_whitened.bin`). Deferred to
  Plan 3 (cross-model alignment) which needs them first.
- Per-layer G_l (general-relativistic form). Flat canonical only.
- Knowledge graph integration. Separate change.
- Cross-model Procrustes alignment. Separate change.
- wasm32v1-none inference target. Tracked in epic #TBD.

## Related issues

- Epic: canonical-vindex-pipeline (this change)
- Blocked-by: none
- Blocks: knowledge-graph-integration, cross-model-alignment, wasm32v1-none-lm

## Risk

Low. Purely additive — no existing files are modified by `larql canonicalize`.
