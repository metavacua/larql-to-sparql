# Transformation, Alignment, and Knowledge-Coupling Theory for Vindexes

**Status:** Draft / Proposed (living research spec)
**Scope:** State-of-the-art (SOTA), fork-only (`metavacua/larql-to-sparql`). Not Apache prior-art.
**License:** **CC-BY-SA 4.0** (this document, as a creative work / formalization — *not* the
repository's default Apache-2.0). Attribution: the `larql-to-sparql` project. Per the project
licensing architecture (`metavacua/larql-to-sparql#175` — https://github.com/metavacua/larql-to-sparql/issues/175),
novel formalization is CC-BY-SA 4.0; the concrete bug-fixes it motivates are Apache-2.0 prior-art and
go upstream separately.
**Date:** 2026-06-17.
**Related issues:** `metavacua/larql-to-sparql#131,#132,#134,#135,#136` (canonical-vindex / constructibility),
`#86,#82,#83,#84,#85` (LQL↔SPARQL↔GraphQL triangle), `#186` (DESCRIBE paradigms),
`#152` (feature_labels producer), `#88` (signed probe), `#184` (silent label-format).
**Evidence:** empirical results in the project's labeling-experiments record (smollm2-360m, qwen3-0.6b,
gemma-3-4b reference), §§11–20.

---

## 0. Motivation

`larql extract` transforms a language model **M** into a **vindex** **V** — a queryable, compressed
weight database — over which LQL operations (`DESCRIBE`, `WALK`, `STATS`, …) run. We observed that for
some models the resulting vindex yields **incorrect** output for certain operations (witnessed:
`DESCRIBE` on smollm2-360m returns no/incorrect relation edges, while the gemma-3-4b reference returns
correct edges). The local engineering goal is correctness; this document formalizes *why* the
incorrectness arises and *what* a correct, model-universal family of operations is, so the fix is
principled rather than per-model tuning.

The central reframing, evidenced below, is: **the vindex is a transform of M, and an LQL operation is
correct iff that transform preserves the operation's semantics.** Where it does not, we either (i) repair
the transform with a fitted *alignment*, or (ii) replace an approximate operation with its faithful
form. Both are special cases of one transformation theory.

## 1. Objects and morphisms

Three representations of "what a model knows":

- **M** — a transformer LM in numeric form (weights + forward dynamics).
- **V** — a vindex: gate vectors, `down_meta` (per-feature predicted-token distributions), and an FFN
  served by sparse retrieval (`walk_ffn`). `V = decompile(M)`; `M ≈ compile(V)` (the existing
  vindex compiler/decompiler pair).
- **K** — an external knowledge graph / ontology (e.g. a Wikidata subgraph) of triples `(s, r, o)`,
  and its SPARQL surface.

Morphisms of interest (each a *transformation* whose faithfulness is the object of study):

- `decompile : M → V` (`extract` / `convert`) and `compile : V → M`.
- `externalize  E : M → K-shaped catalog` — the model's *asserted* triples (§3).
- `internalize  I : K → presence-in-M` — grounding external triples against M (§3).
- `V ⇄ SPARQL` — the canonical typed-property-graph IR and SPARQL adapter
  (`#82,#83`; the LQL↔SPARQL↔GraphQL triangle `#86`).

**Faithfulness.** A transform `T` is *faithful for operation `op`* iff `op(T(M)) ≡ op(M)` up to the
operation's intended semantics. `decompile` is lossy (quantization, sparse-FFN approximation); the
question is whether the loss preserves each `op`. §2 shows a specific, repairable failure.

## 2. Representation alignment (the static-routing failure and its fix)

`DESCRIBE`'s fast path routes a **static** query — the average input-token embedding `q₀ = mean(E[entity]) · scale`
(layer-0) — through the gates at *every* layer L via `gate_knn(L, q₀)`. But the gates at layer L are
trained against the **layer-L residual** `h_L = Φ_L(q₀)`, where `Φ_L` is the model's forward map through
L layers (attention + FFN). The static path silently assumes `q₀ ≈ h_L` (direction). 

**Evidenced failure (§17–§18).** On smollm2-360m the assumption fails: the relation features the model
*does* use (recoverable from the true residual) rank 95–2002 under `gate_knn(L, q₀)`, two with *negative*
gate (the static query points opposite the residual), all below the `DESCRIBE` gate threshold. On the
gemma-3-4b reference the same labels surface — the assumption holds *there*. So static routing is a
**scale-dependent approximation**, "correct enough" only in a regime, not a strictly-valid operation.

**Conjecture (alignment).** The failure is a missing, *fittable* transformation. For each layer L there
is a map `T_L` with `T_L(q₀) ≈ h_L` on a calibration set, recovering correct routing without a per-query
forward. Order candidates by generality and fit the least general that suffices:

> isometry ⊂ similarity (scale·orthogonal) ⊂ affine ⊂ projective (homography) ⊂ polynomial ⊂
> diffeomorphism ⊂ homeomorphism.

Start with **orthogonal Procrustes** (rotation/reflection) + scale + translation (a similarity), escalate
to affine / low-rank only if residuals demand it. Fit by Procrustes / least-squares / gradient descent;
**ICP** if the correspondence between `{q₀}` and `{h_L}` must be discovered rather than given. This is a
*calibration*, and — as observed — a calibration is a training-like operation: LQL correctness and model
construction share machinery.

**Regime as a measurement.** `‖T_L − scale·I‖` (deviation from a pure rescale) is the **adaptive
tradeoff signal**: near-zero ⇒ static-as-is is faithful (large/strong models — fast path), large ⇒ apply
`T_L` (small/weak models). The current threshold/scale constants are calibrated for the large-model
regime; that is the literal sense in which the method is "scaled wrong" for smollm2. The regime signal
relates to the canonical-vindex regime classifier (`#134`), which is presently degenerate — a sign the
regime metric is underspecified and should be derived from alignment, not a single `c_score` cutoff.

**Faithful fallback.** Where no cheap `T_L` suffices, the faithful operation runs the actual forward — the
**relation-template residual** (§3, §6) — at the cost of a forward pass. Aligned-static is the fast path;
template-residual is the correct floor; the regime signal chooses between them.

## 3. Knowledge inversion and coupling calculus

The catalog and `feature_labels.json` are **not** for *answering* `DESCRIBE` (the model answers directly,
§6) — they are for **measurement, detection, and coupling** between the model's internal knowledge and
the external world.

- **Externalize** `E(M) = { (x, r, predict_M(r, x)) : x ∈ X, r ∈ R }` — the model's asserted KG, read by
  forward-passing each relation template and taking the predicted object (§6). Internal knowledge made
  external.
- **Internalize / check** `I(K) = { (s,r,o) ∈ K : predict_M(r,s) = o }` — external facts grounded in M.
- **Coupling** `E(M) ∩ K` — the shared subgraph: where internal and external worlds agree. This is the
  model's *grounding* and the formal **interface** between its latent world and the open semantic web.
- **Divergence** `E(M) \ K` (model asserts beyond the snapshot — new knowledge or hallucination) and
  `K \ E(M)` (the model's **gaps** — the *absence detection* the catalog enables).
- **Double inversion ≈ closure.** Externalize → compare → re-internalize lands on the grounded core, as
  `decompile ∘ compile ≈ id` lands on the faithful core. The pair has a Galois/adjunction flavor
  (`I ⊣ E`); the coupling is its fixed-point set. The "pumping-lemma" instinct is the structural
  characterization of the difference: which triples the model's KG *must*/*cannot* contain relative to K,
  obtained by cycling valid symbols (templates) through M to emit new valid words and feeding them back.

This makes the apparatus a **model-KG ⇄ world-KG comparison/coupling engine**, of which `DESCRIBE` is one
view. It is a *constructive* interpretability result: M's KG is built as a catalog object and compared
formally; the feature-structure layer (which feature carries a relation) makes it constructive at the
*circuit* level, not merely behavioral.

## 4. Consequences for LQL operations

`DESCRIBE` (and kin) admit three paradigms — genuine alternatives with different trade-offs, evidenced and
tracked in `metavacua/larql-to-sparql#186` (https://github.com/metavacua/larql-to-sparql/issues/186):

- **A. Feature-structure** — `gate_knn` → activated features → `feature_labels.json` labels → edges.
  Uses the vindex circuit (mechanistic); needs the labeling infra; subject-specific labels are
  catalog-bound (no generalization); requires §2 alignment to be correct on weak models.
- **B. Prediction-readout** — per relation template, forward-pass and read the predicted object;
  relation = template. Needs only templates + forward + LM head; **no labels, no catalog**; **generalizes
  to any entity** (evidenced 12/12 incl. non-catalog, §20). Behavioral; runs N forwards.
- **C. Hybrid** — a *relation-general* feature confirms the circuit (structural) while the prediction
  supplies the object (general). Keeps structure **and** generalizes; reopens frame-subtraction.

**Correctness criterion.** An LQL operation on V is correct iff it agrees with the same operation on M
under the operation's semantics. For `DESCRIBE`, running the readout through V's own forward (`walk_ffn` +
LM head) makes correctness *equivalent to* a faithfulness test of `decompile` — the deepest sense of
"the vindex is a faithful transform."

## 5. Research agenda (open questions)

1. **Fit the alignment.** Empirically fit `T_L` (Procrustes → affine → low-rank) on smollm2 and test
   whether aligned-static recovers the labeled features; measure `‖T_L − scale·I‖` across smollm2 /
   qwen3 / gemma to map the regime. (Apache-2.0 *fix* of the existing static path; this *theory* is the
   CC-BY-SA frame.)
2. **Derive the regime classifier** from alignment rather than a `c_score` cutoff (resolves `#134`).
3. **Formalize `V ⇄ SPARQL`** as the transform the theory must preserve (`#82,#83,#85`; triangle `#86`).
4. **Constructibility.** Connect to the canonical-vindex / wasm32v1-none constructible-LM programme
   (`#131,#132`) and pure-Rust numeric portability (`#135,#136`): the alignment + coupling calculus is a
   constructibility statement about which knowledge graphs are *realizable* by a model of a given basis /
   dimension.
5. **Extract/convert unification** (deferred; Apache-2.0 upstream proposal) — one streaming transform
   parameterized by source format; see the dedicated issue and the resource/maintenance issues
   `#178,#166,#181,#169` under the governor `#182`.

## 6. Evidence base (provenance)

- Prediction-readout is universal and generalizes: smollm2 12/12 incl. non-catalog entities (§20).
- Static routing fails on smollm2 (rank 95–2002, negative gates, below threshold) but holds on gemma
  (§17). Bare-entity and generic-prompt routing do **not** reach the relation features; only the
  relation-template residual does (§18–§19).
- The method that produced gemma's *working* labels (`probe_mlx.py`) is residual-on-templates — identical
  to the project's producer; the divergence is model-regime, not method (§17).

This document is a living spec; revise as alignment fits and the `V ⇄ SPARQL` formalization land.
