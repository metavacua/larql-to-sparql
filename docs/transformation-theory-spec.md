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

---

## 7. LQL primitive basis

LQL operations split into **primitive** (not expressible as a composition of other LQL statements) and
**derived** (definable in terms of primitives). The end condition for correct extraction is stated only
over the primitive basis — if each primitive is correct on a vindex V, all derived operations are
correct by composition.

### 7.1 Primitive basis

**Browse level** — require only `gate_vectors.bin`, `down_meta.bin`, `embeddings.bin`, `tokenizer.json`,
`index.json`:

| ID | Operation | Distinguishing property |
|----|-----------|------------------------|
| P1 | `WALK` | `gate_knn` on **prompt-last-token** embedding → ranked feature list per layer |
| P2 | `DESCRIBE` | `gate_knn` on **averaged entity** embedding → deduplicated edge set with optional relation labels; distinct query construction and output projection from P1 |
| P3 | `SELECT` | relational condition scan over `down_meta`; the only operation with arbitrary WHERE predicates |
| P4 | `STATS` | aggregate counts from `index.json` + artifact manifests |

**Inference level** — additionally requires `attn_weights.bin` (and `lm_head.bin` unless tied to embeddings):

| ID | Operation | Distinguishing property |
|----|-----------|------------------------|
| P5 | `INFER` | attention-gated full forward pass → next-token logits |

**Mutation level** — additionally write back to vindex:

| ID | Operation |
|----|-----------|
| P6 | `INSERT` |
| P7 | `DELETE` |
| P8 | `UPDATE` |

**Lifecycle**:

| ID | Operation |
|----|-----------|
| P9 | `EXTRACT` (model → vindex artifacts) |
| P10 | `COMPILE` (vindex → model weights; inverse of P9) |

### 7.2 Derived operations (not separately checkable)

| Derived | Derives from |
|---------|-------------|
| `EXPLAIN WALK` | P1 + label join |
| `EXPLAIN INFER` | P5 + per-layer attribution |
| `TRACE` | P5 + residual stream capture |
| `SHOW RELATIONS` | aggregate over P3 label dimension |
| `SHOW LAYERS`, `SHOW FEATURES`, `SHOW ENTITIES`, `SHOW COMPACT STATUS` | specialisations of P3 / P4 |
| `REBALANCE` | iterative (P5; P8) until convergence |
| `COMPACT MINOR` | batch P6 MODE COMPOSE from L0 WAL |
| `COMPACT MAJOR` | MEMIT decomposition: batch P8 over L1 edges |
| `MERGE` | P3 from source + batch P6 into target |
| `DIFF` | symmetric P3 over two vindexes |
| `BEGIN/SAVE/APPLY/REMOVE PATCH`, `COMPILE INTO VINDEX` | recording overlay + materialisation of P6/P7/P8 batch |
| `COMPILE INTO MODEL` | COMPILE INTO VINDEX + weight serialisation (P10) |

---

## 8. End condition for correct extraction (per-model and per-class)

### 8.1 Setup

Let:
- **M** be a model under test.
- **V_M** = `EXTRACT(M)` — the vindex produced.
- **K** = the `larql-knowledge` catalog (Wikidata CC0 + WordNet triples), partitioned as:
  - **K_E** — *entity-activated* facts: satisfied from static entity embedding alone (nationality, continent, borders, language, morphological relations).
  - **K_C** — *completion-required* facts: only satisfiable via full forward pass (capital, genre, …).
- **E_known** = entities appearing in K.
- **P_test** = a finite set of known prompt → answer pairs (for inference parity).

The retrieval closure under operation O on vindex V is:  
`cl_O(V, e) = {edges/tokens reachable from entity e or prompt p by applying O to V}`.

### 8.2 Per-model witnesses

**W1 (STATS — P4):**  
`STATS(V_M).layer_count = M.n_layers  ∧  features_per_layer_min > 0  ∧  vocab_size = M.vocab_size`

**W2 (WALK fires — P1):**  
`∀ e ∈ E_known : |top_features(WALK(V_M, e))| > 0  at knowledge-band layers`  
Validates gate_vectors + embeddings present and aligned.

**W3 (DESCRIBE-positive — P2):**  
`∀ (e, r, t) ∈ K_E : t ∈ cl_DESCRIBE(V_M, e)`  
Entity-activated relations surface in the DESCRIBE closure.

**W4 (DESCRIBE-negative — P2):**  
`∀ (e, r, t) ∈ K_C : t ∉ cl_DESCRIBE(V_M, e)`  
Completion-required relations are absent from the DESCRIBE closure; DESCRIBE is not INFER.

**W5 (SELECT round-trip — P3):**  
`∀ e ∈ E_known : edges(SELECT * FROM EDGES WHERE entity = e)  ⊆  edges(DESCRIBE(V_M, e))`  
down_meta and gate_vectors are self-consistent.

**W6 (labeled coverage — P2 + label join):**  
`relations(SHOW RELATIONS(V_M))  ⊇  {r : (e, r, t) ∈ K_E}`  
`feature_labels.json` covers catalog relations when present.

**W7 (INFER parity — P5, inference level):**  
`∀ p ∈ P_test : top_1(INFER(V_M, p)) = top_1(dense_forward(M, p))`  
Round-trip parity between vindex-backed and dense inference.

**W8 (INSERT closure — P6):**  
After `INSERT (e₀, r₀, t₀)` into V_M → V_M':  
`t₀ ∈ cl_DESCRIBE(V_M', e₀)`

**W9 (DELETE exclusion — P7):**  
After `DELETE WHERE entity = e₀ AND relation = r₀` → V_M'':  
`t₀ ∉ cl_DESCRIBE(V_M'', e₀)`

**W10 (UPDATE swap — P8):**  
After `UPDATE SET target = t₁ WHERE entity = e₀ AND relation = r₀` → V_M''':  
`t₁ ∈ cl_DESCRIBE(V_M''', e₀)`

### 8.3 The end condition (conjunctive, levelled)

| Level | Condition | Enables |
|-------|-----------|---------|
| Browse | W1 ∧ W2 ∧ W3 ∧ W4 ∧ W5 ∧ W6 | P1, P2, P3, P4 and all derived browse operations |
| Browse + Inference | Browse ∧ W7 | P5 and all derived inference operations |
| Browse + Mutation | Browse ∧ W8 ∧ W9 ∧ W10 | P6, P7, P8 and all derived mutation operations |
| Full | Browse ∧ W7 ∧ W8 ∧ W9 ∧ W10 | All LQL operations |

**W3 and W4 are the critical pair.** W3 failing (entity-activated relations not found) is the extraction
bug — evidenced on smollm2-360m and qwen3-0.6b (`metavacua/larql-to-sparql#193`). W4 failing (capital
appearing in DESCRIBE) would mean DESCRIBE has been accidentally redefined as INFER — the overfitting
guard. The fix must satisfy both simultaneously.

### 8.4 Per-class end condition

The per-model end condition extends to a **class of models** ℒ (e.g., all Transformer-family models
extractable to vindexes) as follows.

**C1 (K_E / K_C partition stability):**  
The same facts are entity-activated across the class:  
`∀ M ∈ ℒ : (e,r,t) ∈ K_E ⟺ t ∈ cl_DESCRIBE(V_M, e)` (once per-model witnesses hold)  
This is the empirical hypothesis to verify once #193 is fixed: does the partition differ between
smollm2, qwen, and gemma architectures?

**C2 (Browse-level closure under mutation):**  
If V satisfies Browse, then for any finite sequence σ of INSERT/DELETE/UPDATE operations,
the resulting V' also satisfies W1, W5, W6 (STATS and SELECT self-consistency are preserved by mutations).

**C3 (Round-trip stability):**  
`EXTRACT(COMPILE(V_M)) ≈ V_M` up to quantisation noise — the class is closed under the COMPILE → EXTRACT cycle.

**C4 (Threshold universality):**  
The Browse end condition holds for all M ∈ ℒ under a **model-relative** threshold
`θ_M = f(V_M.gate_scale)` — not an absolute constant. This is the direct statement of the
#193 fix: the current `DESCRIBE_GATE_THRESHOLD = 5.0` is a per-class-member calibration
that belongs in the EXTRACT output (`index.json`) not as a hard constant.

---

## 9. Formal language theory analogy

The end condition has the structure of a closure / recognizability problem from formal language theory.
The analogy is not merely metaphorical — it is structural.

### 9.1 Retrieval hierarchy (Chomsky analogy)

Define the **retrieval closure** of each primitive operation as the set of facts it can recover from
vindex V for entity e:

| LQL primitive | Retrieval closure | Formal analog |
|---------------|------------------|---------------|
| P2 `DESCRIBE` | cl_DESCRIBE(M, e): entity-activated facts, no context | Regular (finite automaton): bounded derivations, no stack |
| P1 `WALK` | cl_WALK(M, p): prompt-context-dependent features | Context-free: depends on context stack (prompt tokens) |
| P5 `INFER` | cl_INFER(M, p): full attention-gated generation | Unrestricted: arbitrary cross-token dependencies |

Strict inclusions hold: `cl_DESCRIBE ⊊ cl_WALK ⊊ cl_INFER`  
(evidenced for capital: not in cl_DESCRIBE, reachable via cl_INFER).

### 9.2 Recognizability and the satisfiability split

A fact (e, r, t) is **DESCRIBE-recognizable** iff `t ∈ cl_DESCRIBE(V_M, e)`. The recognizability
test is the gate_knn derivation: entity embedding e → gate scores → feature f → down_meta top_token t.

The satisfiability split K_E / K_C is the LQL analog of the **pumping lemma boundary**:

> There exists a structural threshold (the model's entity-activation regime) such that facts in K_C
> cannot be pumped into the DESCRIBE derivation for *any* model in the class — they require the full
> generative machinery (INFER). Facts in K_E *can* be derived by DESCRIBE for all models where the
> extraction is correct.

W4 (DESCRIBE-negative) is the formal statement that K_C is **not** recognizable by DESCRIBE — analogous
to the pumping lemma proof that certain languages are not regular. W3 (DESCRIBE-positive) is the
statement that K_E *is* recognizable — the language is in the right class.

### 9.3 Closure properties of the "correctly-extractable" class

Define the class of correctly-extracted vindexes:

`𝒞 = { V_M : M ∈ ℒ, V_M satisfies Browse end condition }`

Closure properties (each is an open hypothesis or a known result):

| Operation | Closed? | Note |
|-----------|---------|------|
| INSERT(V, (e,r,t)) → V' | **Yes** by design (W8) | New fact enters DESCRIBE closure |
| DELETE(V, (e,r)) → V'' | **Yes** by design (W9) | Fact exits DESCRIBE closure |
| MERGE(V₁, V₂) → V₃ | **Hypothesis** | Requires W3/W4 to hold on V₃; relation label conflicts could violate W6 |
| COMPILE(V) → M', EXTRACT(M') → V' | **C3 hypothesis** | Round-trip stability; violated if COMPILE loses gate structure |
| Architecture scaling M → M_large | **Hypothesis** | C4 (threshold universality): larger models may have larger gate scale, but model-relative θ_M preserves the class |

### 9.4 The Nerode equivalence

Two models M₁, M₂ are **LQL-equivalent** (M₁ ≅_LQL M₂) if for all (e, r, t) ∈ K:  
`t ∈ cl_DESCRIBE(V_M₁, e)  ⟺  t ∈ cl_DESCRIBE(V_M₂, e)`

EXTRACT is **faithful** for model M iff `M ≅_LQL V_M` — i.e., the vindex and the model are in the
same LQL-equivalence class. The Browse end condition (W1–W6) is the finite witness set for this
equivalence over the catalog K.

The number of LQL-equivalence classes (over K_E) bounds the *granularity* of interpretability
available via DESCRIBE: models in the same class are structurally indistinguishable at the
entity-activation level of the catalog.
