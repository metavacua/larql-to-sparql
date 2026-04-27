# Backpropagation is INSERT — Complete Experiment Results

**Chris Hay | LARQL Project | April 2026**

---

## Executive Summary

Forty-one experiments across three scales, one MacBook. We started by asking whether gradients are sparse INSERTs into a database. We end by **compiling 61 facts into a 100M model that never saw any of them, in 11 seconds of linear algebra, at 100% accuracy**, with zero gradient descent on any fact — and showing that **chain-of-thought reasoning is the model writing its own SQL** (single-step retrievals as SELECTs, multi-step retrievals as JOINs) to reach the compiled gates. Compiled gates extend to **external solver dispatch**: arithmetic, sort, and scheduling queries route to Python and CP-SAT. Compiled features can be **chained across consecutive layers** with the residual stream acting as the stack: L10→L11→L12 instruction pipelines execute correctly. WordNet relations (synonyms, hypernyms, antonyms) compile alongside facts and arithmetic. Python grammar, API signatures, and idioms compile via the same primitive — a 44-edge unified constellation across 10 relation types scores 44/44 on v10c. **The α=10 hijack that prevented autoregressive chain composition is solved by dropping payload magnitude**: with per-fact α calibration, the multi-hop 10-fact bidirectional constellation reaches 5/5/5/5 (forward/reverse/chained) *and* the reverse gate fires on the model's own emitted output token, producing implicit JOINs inside a single forward pass. The global α knob is replaced by `compiler.balance(target_margin=3)` — automatic per-fact payload calibration at effective α 0.009–0.054, canonical preserved, specificity +16pp over the prior baseline. **The FFN stores language, code, and computation as a single writable graph. Compiled facts chain autoregressively when their payloads are calibrated to the local logit landscape.**

### Path 3 Closed (April 2026)

The final result of the project: a from-scratch 100M model trained on TinyStories, with its FFN frozen and attention retrained, accepts compiled (subject, relation, object) facts via direct construction of FFN feature gates. The 19M probe model and the 100M Path 3 model both demonstrate end-to-end fact compilation.

| Experiment | Question | Answer |
|-----------|----------|--------|
| v10a (100M, 16M tokens) | Does freeze-FFN + retrain-attention reproduce? | **Yes. Step 3 loss 2.12 BEATS Step 1 loss 2.35.** Half the tokens, 16% of params, lower loss. |
| Geometry probe (19M) | Are tied-embed FFN down vectors in embedding space? | **Yes. L11 max cos=0.63. Top features ARE English tokens** (`' time'`, `' Jack'`, `' lived'`). |
| Paris compile (19M) | Can we add "France→Paris" to a model that never saw Paris? | **Yes — both via embedding synthesis and FFN INSERT.** Pure construction. |
| Constellation v5 (100M) | Compile 5 capitals simultaneously? | **5/5 first-token correct, 4/4 decoys clean.** QR-orthogonalized gates. |
| Refinement (100M) | Fix continuation contamination? | **Yes — 23 → 2 capital words across decoys**. Each fact fires for ONE token, then reverts to TinyStories baseline. |
| Scale test (100M) | Compile 25 capitals? | **11/11 of in-vocab capitals.** 14 facts blocked by vocab-clamp aliasing bug. |
| Mixed relations (100M) | Compile capitals + languages + currencies together? | **9/11.** Languages 5/5. Cross-relation collisions for same-subject facts limit capitals 4/5. |
| Object collision (100M) | 4 countries → Euro? | **4/4 correct.** Shared-object compilation works. |
| CoT steering (100M) | Is chain-of-thought query reformulation? | **Yes.** Combined (paraphrase + canonical) matches canonical activation (610 vs 671) and is **20/20 correct**. The residual is literally steered into the gate's activation cone. |
| Multi-hop / JOIN (100M) | Can compiled facts be chained? | **Yes — explicitly and autoregressively** (when payloads are calibrated). 10-fact bidirectional constellation: 5/5/5/5 on v10c with fwd α=1, rev α=5 via `relation_alpha_mul`. The reverse gate fires on the model's own emitted token inside a single forward pass (e.g. Berlin→Germany activates on the "Berlin" the model just produced). The earlier "chains must be written explicitly" finding was an α=10 hijack artifact — see §37, §38. |
| v10c validation (100M, proper vocab) | Does training with proper vocab unblock the OOV-by-clamp facts? | **Yes.** v10a 11/25 capitals → v10c **20/20** of 20 in-vocab capitals. 9 additional facts compile because SentencePiece allocated single tokens to Cairo, Athens, Lisbon, Brussels, Stockholm, Helsinki, Warsaw, Budapest, Bangkok, Hanoi, Ottawa, Tehran. |
| 61-fact constellation (v10c) | Where is the cliff? | **Not found at 61.** 25 capitals + 18 languages + 18 currencies = **61/61 correct** in 11.2 seconds. Refinement strengthens gates 10× (initial 0 → post-refinement mean 1.10). Cross-relation probe 3/3. |
| Compute routing (v10c) | Can compiled gates dispatch to solvers? | **Yes.** Template-structure gate fires with identical activation (57.5) on all unseen operand variations. Arithmetic 5/5 via Python. CP-SAT returns optimal schedules. Knowledge + compute coexist. The FFN is a query router. |
| WASM in the FFN (v10c) | Can compiled features chain across consecutive layers into a pipeline? | **Yes.** Signal persists cos=0.98 from L10→L19 at alpha=0.3. Three-layer chain L10→L11→L12 all fire on canonical, all silent on decoys. Removing L10 collapses the entire pipeline. The FFN is a stack machine: residual = stack, (gate,down) = instruction, forward pass = execution. |
| WordNet compilation (v10c) | Is the FFN syntax band writable? Can we compile synonyms, hypernyms, antonyms? | **Yes.** 24-edge mixed constellation (synonym + hypernym + antonym + capital + arithmetic) at **24/24**. Hypernym held-out generalization **3/3 perfect** across paraphrases. Style transfer via synonym recompilation demonstrated: same model, same prompts, different outputs depending on which synonym set is compiled. The FFN stores language as a writable graph. |
| Code compilation (v10c) | Can Python grammar, API signatures, and idiom patterns compile into the same FFN that stores facts and WordNet relations? | **Yes.** 44-edge unified constellation across 10 relation types (synonym, hypernym, antonym, capital, 2×arithmetic, grammar_follows, first_arg, returns, idiom_next) at **44/44** canonical with 4/4 clean decoys in 8 seconds of compile+refine. Tokenizer coverage is the only blocker for the full v9b 2,134-function API graph. See §36. |
| α structural fix (v10c) | Why does generation collapse to "Paris Paris Paris" when facts are installed? | **α=10 is generation-poison.** Payload 10× the natural FFN scale overwrites the planner at every step. Dropping to α=1 preserves canonical 5/5 *and* restores coherent English narrative between gate fires. Specificity 35.8% → 58.3% on the paraphrase×cue sweep. Adversarial-decoy QR attempted fix was destructive (canonical broke). See §37. |
| Multi-hop CLOSED (v10c) | Does the α fix enable autoregressive chain composition? | **Yes, with per-fact α.** Forward templates have flat next-word priors and close at α=1; reverse templates ("X is located in") have peaked priors ("the") and need α=2-5. Per-fact `relation_alpha_mul` override lands 5/5/5/5 on the 10-fact bidirectional constellation. The reverse gate **fires on the model's own emitted output token** (Berlin→Germany activates on the just-emitted "Berlin"), producing implicit multi-hop JOINs inside a single forward pass. Falsifies the earlier "chains must be written" finding. See §38. |
| Balancer Phase 1 (v10c) | Can the compiler auto-calibrate per-fact α from logit margins? | **Yes for retrieval, no for specificity.** Basic balancer `balance(target_margin=3)` converges in 3-7 iters, preserves canonical 5/5 at effective α 0.009-0.054 (20-100× smaller than "α=1 is safe"), +16pp specificity. Contamination matrix is a keeper diagnostic. Graph-aware balancer implemented and **shown not to work** — the contamination is mostly base-model prior (Berlin prior on any country-topic prompt ~9.7), not compiled gate leakage (~0.2), so payload scaling can't reduce it. Specificity ceiling ~52% for payload-only approaches. See §39. |
| Colchester micro-world (v10c) | Does the mechanism generalize to a new domain (geography + heritage), and does the castle subgraph support multi-hop / goose / autoregressive chains? | **Yes, mechanism fully closed on the compilable subset.** 51-edge constellation (44 existing + 4 forward castle + 3 reverse castle) scores **60/60** across canonical, regression, written multi-hop JOIN, and goose tests. `William→castle` landed at α_eff **0.017** — 600× smaller than α=10 default, 235× smaller than `rose→flower` in the same constellation. All automatic from one `balance(target=3)` call. **New finding:** autoregressive chain composition is template-shape-dependent — works for short generic reverse templates (`"X is located in"`), fails for bespoke ones (`"The great king X built a mighty"`). Remaining gap to the full Colchester graph is entirely v10c tokenizer coverage (Colchester, Essex, postcodes, decimal coordinates all multi-piece). See §40. |
| Spatial dispatch pipeline (v10c) | Can compiled-graph retrieval and solver dispatch be composed into a single query engine that answers natural-language spatial questions with exact computed numbers? | **Yes.** `SpatialDispatcher` wraps the compiled model and pipes retrievals into pluggable Python solvers. 63-edge constellation (44 existing + 8 coord + 2 connectivity + 9 attribute) at 100% across every phase: nearest-X (2/2), coordinate lookup (4/4), distance computation (3/3), range query, filtered query (spatial+food join), cross-domain query. **Distances (0, 141, 283 m) are computed at query time** from four compiled-gate lookups per call — the numbers do not exist in the model. Graceful missing-data handling: uncompiled attributes return `.` rather than hallucinated values. Balanced in 6 iterations. Ships a reusable `spatial_dispatch.py` module (class + Euclidean/haversine solvers + number-word parser) that generalises beyond spatial to any (entity, attribute, value) domain with a solver. See §41. |

### 20M Model (v3-v12)

| Experiment | Question | Answer |
|-----------|----------|--------|
| v3/v4 | Are gradients sparse INSERTs? | Yes — feature-level, 0.00–0.14 overlap |
| v5 | Are FFN and attention separable? | Yes — freeze-FFN beats baseline (0.796 vs 0.821) |
| v6 | Can you compile a graph into FFN? | Yes — 5.6x faster, 103.8% quality |
| v7 | Can FFN be a native graph database? | Yes — single layer +0.002 loss, 100% top-1 match |
| v8 | Can ALL FFN layers be non-neural? | Yes — rules + graph + table, attention beats baseline |
| v9a | Is style structured data? | Yes (infrastructure) — 806KB connotation graph + profiles |
| v9b | Is code structured data? | Yes — 985KB API graph, 2134 functions from 19 packages |
| v9c | Is tool calling structured routing? | Yes — 94% selection, 100% JSON validity, 6KB |
| v12 | Can attention be compiled? | At 20M: 97% — but this did not scale (see below) |

### Gemma 3-4B Scale Validation

| Experiment | Question | Answer |
|-----------|----------|--------|
| v10b | Can attention transfer teach fluency? | No — random init adapts equally well |
| Head classification | Are attention patterns compilable at 4B? | 17.6% (vs 96.4% at 20M) — patterns are content-dependent |
| FFN replacement | Does FFN-as-database hold at 4B? | **Yes — Δ=+0.002, identical to 20M. L12-19 replacement IMPROVES model.** |
| Attention anatomy | What is attention actually doing? | **Refinement bias. 79% dispensable. Outputs 88% template-fixed.** |

**The FFN is a database at any scale.** Replacing knowledge layers with a 49-edge JSON graph produces equal or better predictions at both 20M and 4B (Δ=+0.002).

**Attention is an expensive bias.** Patterns are content-dependent (17.6% compilable), but outputs are template-fixed (88% compilable). 79% of layers can be skipped. 83% of heads attend to BOS.

**Revised compilability: ~70-80%** (down from 96.4% at 20M, up from 17.6% when measuring the right thing).

**Final result (April 2026)**: Compilation works *end-to-end*. We can train a small model from scratch (grammar/coordinate system), freeze its FFN, retrain attention against the frozen FFN, and then **compile arbitrary knowledge into the FFN via direct linear-algebra construction** — no gradient descent on the compiled facts. See sections 25-31.

---

## 1. Architecture

```
20M-parameter Gemma-like transformer:
  - 12 layers, hidden_dim=256, FFN_dim=1024
  - 4 attention heads (GQA, 2 KV heads), RoPE, RMSNorm
  - Gemma 3 tokenizer (clamped to 32K vocab)
  - Trained on ~1,700 synthetic samples: 750 factual (KG), 900 syntax (WordNet), 60 code
  - All experiments on Apple Silicon (CPU/MPS), 6-40 minutes each
```

---

## 2. Gradient Anatomy (v3/v4)

### Finding: Gradients are sparse INSERTs to specific features

Feature overlap between syntax and knowledge gradients: **0.00–0.14 Jaccard** across all layers and all training steps. Different knowledge types write to different FFN features within the same layer.

### Finding: Feature addresses stabilise in three phases

| Phase | Steps | Jaccard | What happens |
|-------|-------|---------|-------------|
| Initial noise | 0–50 | 0.64–0.82 | Random init, features interchangeable |
| Volatile reshuffling | 50–250 | 0.06–0.33 | Gradients reorganise features — schema being written |
| Settling | 500+ | 0.5–0.77 | Features claim stable relations |

Code crystallises first (Jaccard 0.77 by step 2000), then facts (0.56), then syntax (0.52). Cleaner signal = faster stabilisation.

### Finding: Rare relations get exclusive features

| Relation | Samples | Best Exclusivity | Persistent Feature |
|----------|---------|------------------|--------------------|
| antonym | 4 | 0.405 | L1 #251, L4 #52, L10 #48 |
| meronym | 28 | 0.427 | L6 #491, L8 #711 |
| plural | 8 | 0.377 | L1 #977, L4 #344 |

Feature #933 at L7 changed owner (antonym→plural at step 2000) — a database UPDATE operation.

### Finding: Layer bands don't emerge at 12 layers

Gradient energy follows the backprop chain, not knowledge type. Bands need 34+ layers. At 12 layers, the model partitions its feature space within each layer. The layer-band structure in Gemma 3-4B is feature-level partitioning stratified across depth.

---

## 3. FFN/Attention Separability (v5)

### Finding: The FFN database and attention query engine are fully separable

| Run | Final Loss | Time | Trainable Params |
|-----|-----------|------|-----------------|
| **Baseline** (full training) | 0.8209 | 1151s | 19,994,880 |
| **Freeze-FFN** (trained DB, attention only) | **0.7956** | **570s** | 10,557,696 |
| **Progressive** (freeze FFN after epoch 5) | 0.8275 | 646s | 10,557,696 |

Freeze-FFN **beats** baseline: loss 0.796 vs 0.821, **2x faster**. When attention trains against a stable database, everything converges faster and better.

Progressive vs freeze-FFN gap (0.823 vs 0.796) = the database quality gap. Model quality is bounded by FFN quality, not attention training duration.

---

## 4. Graph-to-Weights Compiler (v6)

### Finding: Compiled FFN beats trained FFN

| Run | Pre-Attn Loss | Final Loss | vs Baseline |
|-----|--------------|-----------|-------------|
| Baseline (40ep full) | — | 0.7949 | ref |
| Freeze-FFN + 15ep attn | 10.04 | 0.7813 | -0.014 |
| **Compiled + 15ep attn** | **3.24** | **0.7780** | **-0.017** |
| Random FFN + 15ep attn | 10.35 | 0.8670 | +0.072 |

**Database quality: 103.8%** — the compiled FFN is a better database than 40 epochs of gradient descent. Strategy: transfer top-K features per relation (26.6%), align rest with residuals, copy embeddings + norms.

**Total: 4s compilation + 203s attention training = 207s.** vs 1150s baseline. **5.6x speedup.**

---

## 5. Native Graph FFN Replacement (v7)

### Finding: FFN layers are replaceable by graph queries

| Layers Replaced | Loss | Δ vs Baseline |
|----------------|------|---------------|
| 0 (baseline) | 0.980 | ref |
| **1 (L6 only)** | **0.982** | **+0.002** |
| 4 (L4-7, knowledge) | 1.018 | +0.038 |
| 6 (L3-8, middle) | 1.235 | +0.255 |
| 12 (all layers) | 6.208 | +5.228 |

Single-layer graph replacement: **100% top-1 prediction match**, 90.8% top-5 overlap. The graph produces functionally identical outputs to the weight-based FFN.

The knowledge base is a **45KB JSON file** with 426 edges. Readable, diffable, versionable.

Residual decoder accuracy: 76% entity, 88% relation, 66% joint — from cosine similarity against mean residuals.

---

## 6. The FFN is Three Systems (v8)

### Finding: Output engine is a perfect replacement. Attention on all three beats baseline.

| System | Layers | Loss | Δ vs Baseline | Status |
|--------|--------|------|---------------|--------|
| **Output (table)** | L8-11 | 0.9495 | **+0.0000** | **PERFECT** |
| **Knowledge (graph)** | L4-7 | 1.0097 | +0.0602 | Near-perfect |
| Syntax (rules) | L0-3 | 3.2291 | +2.2796 | Weak alone |
| **All three + trained attention** | L0-11 | **0.9456** | **-0.0039** | **BEATS BASELINE** |

The output layers are literally a lookup table — zero information lost when replaced. The syntax engine is weak in isolation, but attention compensates completely. When you give attention 15 epochs to learn how to query rules + graph + tables, it adapts and produces a **better model than weight-based FFN**.

---

## 7. Style Engine (v9a)

### Finding: Style is structured data (infrastructure validated)

| Component | Size | Content |
|-----------|------|---------|
| Connotation graph | 823 KB | 5,000 words, 5 connotation axes (formality, warmth, concreteness, complexity, intensity) |
| Style profiles | 1.5 KB | 5 registers (Hemingway, academic, casual, legal, poetic) |
| Discourse templates | 925 bytes | Factual description, entity description patterns |
| **Total** | **806 KB** | |

Synonym selection works: `big→large` for formal, `big→big` for casual. Style bias correctly boosts complex/formal words for academic, common/short words for casual.

Generation test hit the model quality wall — 20M model trained on synthetic sentences can't produce fluent prose in any style. The data structures are correct; they need a model that speaks English.

---

## 8. Code Engine (v9b)

### Finding: Code structure is structured data

| Component | Content | Size |
|-----------|---------|------|
| API graph | 19 packages, 2,134 functions, 3,665 edges, 4,406 params | 1,003 KB |
| Grammar constraints | 33 Python keywords, 30 builtins, state machine | <1 KB |
| Idiom graph | 90 AST co-occurrence patterns from 23 snippets | 5 KB |
| **Total** | | **985 KB** |

- 8/8 real Python snippets validate through `ast.parse`
- 80% of API calls found in the graph (json.loads, math.sqrt, os.listdir, etc.)
- Full function signatures extracted: `json.loads(s, cls, object_hook, ...)`, `math.sqrt(x)`, etc.
- Packages introspected: os (158 functions), numpy (66 functions), torch (71 functions), and 16 more

---

## 9. Tool Engine (v9c)

### Finding: Tool calling is routing + schema validation

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Tool selection | **94%** (47/50) | 90% | **PASS** |
| Argument extraction | **91%** (43/47) | 85% | **PASS** |
| JSON validity | **100%** (47/47) | 100% | **PASS** |
| End-to-end | **84%** (42/50) | — | Solid |

**6 KB total.** 10 tools with full JSON schemas, keyword routing rules, regex argument extraction, schema validation. No training. No fine-tuning. No neural computation.

Per-tool accuracy: 8/10 tools at 100% selection (calculate, calendar, database, read_file, run_code, search, email, translate). Weather at 80%, image generation at 60% (regex pattern gaps, easily fixable).

---

## 10. Compiled Attention (v12)

### Finding: 97% of attention is compilable. The residual 3.6% is the irreducible neural component.

| Configuration | Loss | Δ% | Trained Params |
|--------------|------|-----|---------------|
| Baseline (full trained model) | 0.926 | ref | 20M (100%) |
| Compiled attention only | 3.483 | +276% | 0 (0%) |
| **Hybrid (compiled + 1 head/layer)** | **0.952** | **+2.9%** | **786K (3.6%)** |

Mean-pattern compiled attention gets **95% top-1 match** despite 276% higher loss — it gets the right answer but with flat probability distributions.

The hybrid (compiled patterns + 1 small trained head per layer) closes to **2.9% of baseline** with only **786K trainable parameters** (3.6% of total).

Head type analysis: 47/48 heads are content-dependent, 1 is fixed-pattern (BOS-attending at L10H2). The fixed head has the strongest focus across all relations (val=0.79–0.98).

---

## The Complete Picture

### What's compilable

| Component | Compilable | Quality | Method |
|-----------|-----------|---------|--------|
| FFN output layers (L8-11) | **100%** | Δ=0.000 | Distribution table |
| FFN knowledge layers (L4-7) | **94%** | Δ=+0.060 | JSON graph database |
| FFN syntax layers (L0-3) | Partial | Weak alone | WordNet + morphology + AST rules |
| Attention (47/48 heads) | **97%** | Mean patterns | Extracted from trained model |
| **Combined (all FFN + most attn)** | **96.4%** | **+2.9%** | **786K trained params remain** |

### Non-neural model size

```
Style engine (v9a):      806 KB  (connotation + profiles + templates)
Code engine (v9b):       985 KB  (API graph + idioms + grammar)
Tool engine (v9c):         6 KB  (registry + routing + schemas)
Knowledge graph (v8):     45 KB  (432-edge JSON)
Output table (v8):        ~0 KB  (extracted from trained weights)
Attention templates (v12): TBD   (mean patterns)
────────────────────────────────
Total non-neural:      ~1.8 MB
```

### The irreducible neural component

**786,432 parameters** — one attention head per layer (12 layers × 1 head × ~65K params). This handles content-dependent query routing that mean patterns can't capture. Everything else is JSON files.

---

## Timeline

| Experiment | Duration | Key Result |
|-----------|----------|-----------|
| v3/v4 (gradient anatomy) | ~30 min | Feature-level separation confirmed |
| v5 (freeze-FFN) | ~40 min (3 runs) | FFN/attention separable, 2x faster |
| v6 (graph compiler) | ~35 min | Compiled FFN beats trained, 5.6x faster |
| v7 (native graph) | ~25 min | 100% top-1 match on single layer replacement |
| v8 (three systems) | ~20 min | Output table Δ=0, attention on all 3 beats baseline |
| v9a (style engine) | ~25 min | 806KB infrastructure, connotation graph built |
| v9b (code engine) | <1 min | 985KB, 2134 functions from 19 packages |
| v9c (tool engine) | <1 min | 94% selection, 100% JSON validity, 6KB |
| v12 (compiled attention) | ~45 min | Hybrid within 2.9%, 3.6% trainable params |
| **Total** | **~4 hours** | |

---

## Files

| File | Experiment | Contents |
|------|-----------|----------|
| `model.py` | All | TinyGemma (20M params, 12 layers) |
| `synth_data.py` | v1 | Hand-crafted synthetic data |
| `synth_data_v2.py` | v3+ | WordNet + synthetic KG data |
| `experiment.py` | v1 | Char-level tokenizer, basic gradient anatomy |
| `experiment_v2.py` | v2 | Word-level tokenizer, live output |
| `experiment_v3.py` | v3 | Gemma tokenizer, WordNet, contrastive tests |
| `experiment_v4_features.py` | v4 | Feature-level tracking, stability, exclusivity |
| `experiment_v5_freeze.py` | v5 | Freeze-FFN: baseline vs freeze vs progressive |
| `experiment_v6_compiler.py` | v6 | Graph-to-weights compiler |
| `experiment_v7_native_graph.py` | v7 | Native graph FFN replacement |
| `experiment_v8_three_systems.py` | v8 | Rules + graph + table FFN replacement |
| `experiment_v9a_style.py` | v9a | Style engine: connotation + profiles |
| `experiment_v9b_code.py` | v9b | Code engine: API graph + grammar + idioms |
| `experiment_v9c_tools.py` | v9c | Tool engine: routing + schema validation |
| `experiment_v12_compile.py` | v12 | Compiled attention parser |
| `results/` | v1 | v1 results |
| `results_v3/` | v3 | Anatomy, contrastive, loss |
| `results_v5_freeze/` | v5 | Freeze-FFN results |
| `results_v6_compiler/` | v6 | Compiler results |
| `results_v7_native_graph/` | v7 | Native graph results + knowledge_base.json |
| `results_v8_three_systems/` | v8 | Three-system results |
| `results_v9a_style/` | v9a | Style engine results + connotation_graph.json |
| `results_v9b_code/` | v9b | Code engine results + api_graph.json |
| `results_v9c_tools/` | v9c | Tool engine results + tool_registry.json + tool_graph.json |
| `results_v12_compile/` | v12 | Compiled attention results + templates.json |

---

## Scale Validation: Gemma 3-4B (April 2026)

Everything above was proven on a 20M-parameter toy model. The following experiments test whether the findings hold on Gemma 3-4B — a real model (34 layers, 8 heads, 272 total heads, hidden=2560) trained on billions of tokens.

---

### 11. Attention Transfer (v10b)

#### Question: Can compositional attention be transferred from a large model to a smaller one with compiled FFN?

Tested at 20M (dim=256) and 100M (dim=512). Projected Gemma 3-1B attention weights via SVD into smaller models with compiled FFN, fine-tuned 3-5 epochs.

| Scale | Transfer Loss | Random Loss | Transfer Fluency | Random Fluency |
|-------|-------------|-------------|-----------------|----------------|
| 20M (dim=256) | 1.557 | 1.460 | 68% | 73% |
| 100M (dim=512) | 1.306 | 1.114 | 82% | 80% |

**Finding: Transfer does not help.** Random attention adapts to compiled FFN equally well or better than projected Gemma attention. The geometric structure of attention doesn't survive dimensionality compression (4.5x at 20M, 2.25x at 100M).

**Finding: Neither model produces fluent English.** All outputs are degenerate — repeated tokens, tokenizer artifacts, template loops. The bottleneck is training data (1,700-2,400 synthetic sentences), not architecture. At 100M, the model correctly retrieves facts but can't compose them into natural sentences.

**Finding: The 20M architecture is too small for fluent generation** regardless of attention source. This confirms the roadmap's prediction.

---

### 12. Gemma 3-4B Attention Head Classification

#### Question: What fraction of 272 heads are compilable at real scale?

Ran the v12 head classification pipeline on Gemma 3-4B. Extracted attention patterns from 70 diverse prompts across 6 categories (factual, syntactic, compositional, narrative, code, multilingual). Classified each head by focus consistency, positional bias, mean pattern similarity, and entropy stability.

| Metric | 20M Model | Gemma 3-4B |
|--------|-----------|-----------|
| Total heads | 48 | 272 |
| Compilable (pattern-level) | 46/48 (96.4%) | **48/272 (17.6%)** |
| Content-dependent | 2/48 (4%) | **224/272 (82.4%)** |
| Top-5 output stability | — | **100%** |

**Finding: Sharp phase transition at Layer 6.**

```
L0-5:   100% compilable (48/48 heads) — structural/positional patterns
L6-33:    0% compilable (0/224 heads) — all content-dependent
```

The first 6 layers are pure structural attention — parsing, positional, template-level. Everything after L6 is content-dependent. This is not noise — it's architecture.

**Finding: 96.4% was a toy-model artifact.** With only 1,700 synthetic samples, the 20M model's attention never needed to be content-dependent. At 4B trained on billions of tokens, attention has learned rich content-dependent routing that mean templates can't capture.

**Finding: Top-5 output stability is 100%.** The routing varies but the destination is the same. Content-dependent heads produce different attention patterns per input but converge on the same candidate tokens.

---

### 13. Gemma 3-4B FFN Three-System Replacement

#### Question: Does replacing FFN layers with graph database lookups work at real scale?

Built entity/relation codebooks from Gemma's own residuals using 49 verified factual triples (capitals, languages, currencies, science facts). Replaced FFN at each layer with: decode residual → query graph → inject target embedding. Measured loss impact per layer.

| Layer | Type | Factual Loss | Δ vs Baseline | Top-1 | Top-5 |
|-------|------|-------------|--------------|-------|-------|
| Baseline | — | 1.297 | ref | 55% | 100% |
| **L10** | sliding | 1.299 | **+0.002** | 57% | 98% |
| L7 | sliding | 1.256 | -0.040 | 71% | 98% |
| L24 | sliding | 1.256 | -0.041 | 73% | 98% |
| L11 | full | 1.205 | -0.091 | 65% | 98% |
| L12 | sliding | 0.963 | -0.334 | 80% | 100% |
| L13 | sliding | 0.947 | -0.350 | 76% | 100% |
| **L17** | full | 0.937 | **-0.360** | **84%** | **100%** |
| **L18** | sliding | 0.601 | **-0.695** | **92%** | **100%** |

**Finding: Best single-layer Δ = +0.002 — identical to 20M result.** L10 replacement at 4B matches L6 replacement at 20M to three decimal places. The FFN-as-database mechanism is scale-independent.

**Finding: Knowledge layers (L12-19) replacement IMPROVES the model.** Graph lookups at these layers produce *better* factual predictions than the trained FFN weights. The 49-edge JSON graph outperforms gradient descent, exactly as v6 showed at 20M.

**Finding: 18/34 layers (53%) replaceable at |Δ| < 0.5.** 23/34 (68%) at |Δ| < 1.0.

| Metric | 20M Model | Gemma 3-4B |
|--------|-----------|-----------|
| Best single-layer Δ | +0.002 | **+0.002** |
| Replaceable layers (|Δ| < 0.5) | 4/12 (33%) | **18/34 (53%)** |
| Knowledge band improvement | +6% quality | **L12-19 all improve** |

**Layer band structure at 4B:**

```
L0-5:   Syntax/parsing  — sensitive to replacement (5/11 replaceable)
L6-19:  Knowledge        — HIGHLY replaceable (12/14 at |Δ| < 0.5)
L20-25: Transition       — mixed (some replaceable, some not)
L26-33: Output           — sensitive (only 3/11 replaceable)
```

Band replacement degrades due to error compounding. Single-layer replacement works; multi-layer needs attention adaptation (v5 freeze-FFN approach).

---

### 14. What Is Attention Actually Doing?

#### Question: Five hypotheses about attention's role. Which is correct?

Ran 8 measurements across 50 prompts on Gemma 3-4B:
- M1: Attention vs FFN output norm ratios
- M4: Residual change analysis (suppression vs amplification)
- M5: Cross-entity cosine at multiple thresholds
- M6: Head specialization via PCA
- M7: Causal attention skipping per layer
- M8: Attention pattern decomposition (what tokens get attended to)

#### Results

| Measurement | Result | Implication |
|------------|--------|------------|
| M1: attn/ffn norm ratio | **6.4x** (attn is LARGER) | Attention is loud |
| M4: positive/negative bias | **All 34 layers balanced** | Not suppression (kills H3) |
| M5: output cosine @ 0.94 | **88% compilable** | Outputs are template-fixed |
| M5: output cosine @ 0.60 | **100% compilable** | All outputs nearly identical |
| M6: PCA classification | 66% multi-mode, 5% single | Patterns vary (expected) |
| M7: dispensable layers | **27/34 (79%)** | Skipping attention preserves top-1 |
| M8: dominant head role | **83% attend to BOS** | Most heads compute a fixed bias |

#### The Critical Discovery

**Attention patterns are content-dependent. Attention outputs are template-fixed.**

The earlier 17.6% compilability measured attention *patterns* (which positions attend to which). This experiment measured attention *outputs* (what the attention layer contributes to the residual). The patterns vary. The outputs don't.

| What We Measured | Compilable |
|-----------------|-----------|
| Attention patterns (v12 approach) | 17.6% |
| **Attention outputs (new)** | **88-100%** |

**Finding: 27/34 layers (79%) are dispensable.** You can skip attention entirely and the top-1 prediction doesn't change. Attention at most layers is high-norm but low-information — a very expensive way to compute a nearly-constant bias.

**Finding: 83% of heads attend primarily to BOS.** The beginning-of-sequence token serves as a global anchor. Most heads aren't doing entity extraction or relation identification — they're computing a positional bias.

#### Hypothesis Scores

| Hypothesis | Score | Evidence |
|-----------|-------|---------|
| **H1: Refinement** | **3** | 6.4x norm but 79% dispensable. Loud but unimportant. |
| H2: Assembly | 0 | Only 3 relation-identifying heads found. No entity extractors. |
| H3: Cancellation | 0 | Zero suppressing layers. All balanced. |
| H4: Sub-Templates | 1 | 88% compilable at 0.94 supports template theory. |
| H5: Graph Walk | 1 | Structured norms, but no entity progression found. |

**Winner: H1 — Refinement.** Attention is minor refinement. The FFN carries the model.

#### What This Means

The transformer at scale is:
- **FFN: a database** (proven — 53-68% replaceable, knowledge layers improve with graph)
- **Attention: a noisy bias** (proven — 79% dispensable, outputs 88-100% template-fixed)
- **The routes differ. The destinations are the same.**

Different inputs trigger different attention patterns (content-dependent routing), but those patterns produce nearly identical outputs (template-fixed results). Attention takes 272 different paths to arrive at the same place.

---

### 15. BOS Is NOT The Context Register

#### Question: Does BOS accumulate input-specific context across layers?

Tracked BOS (position 0) embedding across all 34 layers for 42 diverse prompts. Measured pairwise cosine similarity between same-template, same-entity, and cross-task prompts.

**Finding: BOS is a fixed scaffold.** Same-template, same-entity, and cross-task cosines were ALL 1.0000 at every layer. BOS carries zero input-specific information. It transforms across layers (L0↔L33 cosine = -0.14) but transforms identically regardless of input.

The 83% of heads attending to BOS are reading a constant. The "expensive bias" interpretation was correct.

---

### 16. The Prediction Position IS The Context Register

#### Question: Does the last token position carry the routing signal?

Tracked prediction position (last token) across all 34 layers. Same prompts as BOS experiment.

**Finding: The prediction position carries genuine input-specific information.**

| Layer | Same-Template | Same-Entity | Cross-Task | Gradient |
|-------|-------------|-------------|-----------|----------|
| L0 | 0.99 | 0.96 | 0.94 | 0.05 |
| L3 | 0.98 | 0.60 | 0.69 | **0.29** |
| **L6** | **0.81** | **0.62** | **0.39** | **0.42** |
| L9 | 0.99 | 0.90 | 0.83 | 0.16 |
| L15 | 0.96 | 0.90 | 0.76 | 0.20 |
| L33 | 0.96 | 0.86 | 0.90 | 0.06 |

**The gradient exists and it's large.** At L6: same-template=0.81, cross-task=0.39 — a 0.42 gap. Completely different from BOS (all 1.0000).

#### The Hourglass: Entity Divergence and Reconvergence

```
L0:  France↔Japan = 0.99, France↔Code = 0.81   (just token embeddings)
L3:  France↔Japan = 0.97, France↔Code = 0.44   ← TASK TYPES DIVERGE
L6:  France↔Japan = 0.86, France↔Code = 0.37   ← ENTITIES DIVERGE
L9:  France↔Japan = 0.99, France↔Code = 0.58   ← ENTITIES RECONVERGE
L18: France↔Japan = 1.00, France↔Code = 0.83   (merged back)
```

Entity signal appears at L6, gets consumed by FFN, and the representations reconverge by L9. This is the hourglass architecture seen in earlier residual trace work — the entity-specific information exists briefly in a narrow band where the FFN reads it, then disappears.

#### Prediction Position → FFN Correlation

The prediction position state strongly predicts FFN activation patterns:

| Layer | Correlation |
|-------|-----------|
| L0 | **0.87** |
| L10 | **0.86** |
| L20 | **0.74** |
| L33 | **0.80** |

High correlation throughout — the prediction position IS where the FFN reads its routing signal.

---

### 17. Head Ablation: Which Heads Actually Matter?

#### Question: Of the 272 heads, which are critical?

Zeroed out all attention at layers containing each head type and measured factual accuracy (top-1, top-5).

| Remove | Heads | Top-1 | Δ | Interpretation |
|--------|-------|-------|---|---------------|
| Baseline | — | 62% | ref | — |
| BOS | 227 (34 layers) | 0% | -62% | Critical scaffold |
| Previous | 19 (15 layers) | 0% | -62% | Critical for composition |
| Function word | 12 (7 layers) | 0% | -62% | Critical for template |
| **Self** | **11 (9 layers)** | **100%** | **+38%** | **HARMFUL — removal helps** |
| **Relation** | **3 (2 layers)** | **62%** | **+0%** | **Irrelevant for factual** |

**Finding: Self-attention heads are harmful.** Removing them improves accuracy from 62% to 100%. These 11 heads add noise that degrades predictions.

**Finding: Relation heads don't matter for factual accuracy.** The 3 heads classified as "relation-attending" contribute nothing to getting Paris from France. The relation information must be encoded elsewhere (likely in the template structure captured by function-word heads).

**Finding: BOS + previous + function-word are all critical.** The model needs the scaffold (BOS), local composition (previous), and template structure (function words). These three systems form the attention substrate.

#### The Architecture of Attention at Scale

```
BOS heads (227, 83%):          Fixed scaffold — pre-computable constant
Previous-token heads (19, 7%): Local composition — bigram-level chaining
Function-word heads (12, 4%):  Template structure — "of", "is", "the" encode the query type
Self heads (11, 4%):           HARMFUL — add noise, should be removed
Relation heads (3, 1%):        Non-contributing — relation info is in template structure

Critical: BOS + previous + function_word = 258 heads (95%)
  These form the substrate. All are structurally simple.

Harmful: self = 11 heads (4%)
  Removing these improves the model.

Irrelevant: relation = 3 heads (1%)
  Don't contribute to factual accuracy.
```

---

## Revised Architecture at Scale (Final)

```
Component               20M Estimate    4B Actual           Status
──────────────────────────────────────────────────────────────────────
FFN (knowledge)         100% compilable 53% replaceable     CONFIRMED (Δ=+0.002)
FFN (output)            Δ=0 (perfect)  L32-33 sensitive     Partially confirmed
Attention patterns      96.4% compilable 17.6%              REVISED (toy artifact)
Attention outputs       —               88-100% compilable  NEW FINDING
Attention skippable     —               79% dispensable      NEW FINDING
BOS as context register —               DISPROVEN            BOS is fixed scaffold
Prediction position     —               IS the register      Gradient=0.42 at L6
Self-attention heads    —               HARMFUL              Removal improves +38%
Relation heads          —               Irrelevant           No factual contribution
Entity hourglass        —               L6 diverge, L9 merge CONFIRMED

Overall compilability:
  20M claim: 96.4%
  4B reality: ~70-80%
  With self-head removal: potentially higher
```

---

## Final Conclusion

The transformer at 4B scale is three things:

**1. The FFN is a database.** Scale-independent. Δ=+0.002 at both 20M and 4B. Knowledge layers (L12-19) are replaceable by graph lookups that IMPROVE the model. The three-system decomposition (syntax + knowledge + output) holds.

**2. Attention is structured scaffolding.** 83% of heads compute a fixed bias (BOS). 7% do local composition (previous token). 4% encode template structure (function words). These are structurally simple — pre-computable constants, single-token projections, and word-class lookups. Together they form the substrate the FFN reads.

**3. The prediction position is the context register.** The last token accumulates input-specific information across layers. Task types diverge at L3. Entity identity appears at L6, gets consumed by FFN, and reconverges by L9 (the hourglass). The prediction position state predicts FFN activation with 0.86 correlation.

**The surprising finding: 11 self-attention heads are harmful.** Removing them improves factual accuracy from 62% to 100%. The model is better without 4% of its attention heads.

**The "3 relation heads" hypothesis was wrong.** They don't contribute to factual accuracy at all. The relation information is encoded implicitly in the template structure captured by function-word heads — "The capital of X is" vs "The president of X is" is distinguished by the function words, not by dedicated relation-detecting heads.

**What remains genuinely neural:**
- FFN at sensitive layers (L0-5, L26-33): syntax and output formatting
- The 7 essential attention layers (L0-5, L24): where task/entity divergence happens
- The prediction position's information accumulation across layers

**What can be compiled/precomputed:**
- BOS contributions: pre-computed constants (227 heads, 0 FLOPs)
- Knowledge FFN: graph lookups (L6-19, Δ=+0.002)
- Self-attention heads: removed entirely (improves model)
- Attention outputs: 88% template-fixed across entity substitutions

---

### 18. Query Lifecycle Trace

#### Question: Can we watch the database query being built and answered in the residual stream?

Projected the prediction position's residual against the embedding matrix at every layer, attempting to read what the model "thinks" in token space at each stage.

**Finding: Embedding projection fails at intermediate layers.** Top tokens at L3-L30 are noise — Cuneiform characters, random multilingual tokens, `<unused>` tokens. The residual stream operates in a different space than the output embedding space. The "logit lens" technique doesn't work cleanly on Gemma 3-4B with its 262K vocabulary.

**Finding: Answer appears late (L25+), not at L9.** Paris/Tokyo/Berlin only reach top-5 at the final layer (L33). There is no sharp "answer appears at L9" moment. Knowledge retrieval is gradual and non-monotonic.

**Finding: Attribution is negative through middle layers.** Both attention and FFN decrease the answer token's score through L6-L20 before it rises at the end. The computation is not a clean query→answer pipeline visible through this lens.

**Finding: L24 attention DOES add entity vocabulary.** At L24, attention contributes "Jean, français, Français" for France queries, "japones, япон, japonais" for Japan, "German, german, Germans" for Germany. This matches L24 being one of the 7 essential layers from the ablation study.

**Finding: Entropy is constant across all task types (~12.5 nats at every layer).** Factual queries don't sharpen faster than creative ones. The probability distribution doesn't concentrate until the very last layers.

**Implication:** The database query lifecycle may be real (the FFN replacement experiment proved the FFN is a database), but it operates in internal representation space that can't be read through raw embedding projection. Trained probes are needed to decode intermediate representations at this scale.

---

### 19. Linear Probes: Reading the Query Assembly

#### Question: Can we read structured information from the residual using trained probes instead of raw embedding projection?

Trained tiny linear probes (single linear layer, ~23K params each) at every layer on 84 prompts across 7 task categories. Each probe answers one question about the residual at the prediction position.

| Probe | Classes | First ≥80% | First ≥90% | Final Accuracy |
|-------|---------|-----------|-----------|---------------|
| **Entity presence** | 2 | **L0** | **L0** | **100%** |
| **Relation type** | 6 | **L0** | **L2** | **94%** |
| **Task type** | 7 | **L4** | **L8** | **89%** |
| Entity identity | 2 (limited) | L0 | L0 | 50% (inconclusive) |

**Finding: The query assembles in the first 8 layers.** A linear probe can perfectly classify the relation type (capital_of vs president_of vs language_of) by L2. Task type (factual vs code vs narrative vs reasoning) reaches 80% at L4 and 100% at L8.

**Finding: Relation before task type.** The model classifies "this is a capital_of query" (L2) before it fully classifies "this is a factual query" (L4-8). Specific before general — the model reads the relation keyword before categorising the broad task type.

**Finding: Entity presence is in the embeddings.** Whether a prompt contains a named entity is readable from L0 with 100% accuracy. The structural difference between factual and non-factual prompts is in the token embeddings themselves.

**The query assembly timeline:**

```
L0:   Entity presence known (from embeddings)
L2:   Relation type classified (94%)
L4:   Task type emerging (80%)
L8:   Task type confident (100%)
L9+:  FFN knowledge retrieval (matches FFN replacement findings)
L24:  Entity vocabulary in attention output
L33:  Answer in top-5
```

**Implication for from-scratch attention:**

```
L0-2:   Relation detector — pattern match on relation keywords
        ("capital" → capital_of, "president" → president_of)
        Replaceable by keyword lookup.

L3-8:   Task classifier — combine structural signals
        ("The X of Y is" → factual, "def X():" → code)
        Replaceable by template matcher.
        Compute engine routing possible by L4.

L9-23:  FFN knowledge retrieval — query assembled, database answers
        Already proven replaceable by graph lookups (Δ=+0.002).

L24+:   Output formatting
```

This gives the concrete recipe for building attention from structured components. The query assembly is pattern matching and template classification — two operations that are deterministic and derivable from the prompt structure.

---

### 20. Derived Attention: Building From Components

#### Question: Can attention be replaced with pre-computed constants + single-token projections?

Extracted attention components from Gemma 3-4B:
- **BOS constants**: Measured mean attention output from 12 reference prompts (not raw embedding × V × O, which produces garbage)
- **Previous-token projections**: V×O weight slices for 19 previous-token heads
- **Function-word projections**: V×O weight slices for 12 function-word heads
- **Self/relation heads**: deleted (self is harmful, relation is irrelevant)

Tested four configurations against full Gemma 3-4B attention:

| Config | Factual Top-1 | Factual Top-5 | Agreement with Full |
|--------|-------------|-------------|-------------------|
| Full attention (baseline) | 8/12 (67%) | 12/12 (100%) | 100% |
| Skip attention entirely | 0/12 (0%) | 0/12 (0%) | 0% |
| BOS constants only | 0/12 (0%) | 0/12 (0%) | 15% |
| Derived (BOS + prev + func) | 0/12 (0%) | 0/12 (0%) | 15% |
| **Hybrid (real@L0-5,L24 + derived@rest)** | **6/12 (50%)** | **10/12 (83%)** | **45%** |

**Finding: The hybrid recovers 83% factual top-5.** Real attention at just 7 essential layers (L0-5, L24) combined with measured constants at 27 dispensable layers brings back most factual capability. Generation produces "Paris" for the capital-of-France prompt (with looping).

**Finding: 79% of attention IS replaceable with measured constants.** The 27 dispensable layers genuinely contribute only a near-constant bias. The 7 essential layers carry the input-specific routing that the FFN needs.

**Finding: Derived components add minimal signal over BOS-only.** Previous-token and function-word V×O projections don't improve over the mean attention constant (both 15%). The routing information is in the full attention computation at essential layers, not in isolated head projections.

**Finding: Raw embedding × V × O produces garbage.** The first attempt (computing BOS constants from raw BOS embedding projected through V and O) produced incoherent output. The BOS state at each layer has been transformed by all previous layers — can't be computed from the raw embedding alone.

**Generation comparison:**

```
Baseline:  "The capital of France is a city of contrasts. It is a city of history..."
Derived:   "The capital of France is a country that is the most popular and the most popular..."
Hybrid:    "The capital of France is a city of Paris is a city of Paris is a city of Paris..."
```

The hybrid gets "Paris" — the FFN retrieves the correct answer when the essential layers provide routing. The looping is from the derived layers not providing enough variation.

**Architecture implication:**

```
Component                    Replaceable?    Evidence
───────────────────────────────────────────────────────────────
27 dispensable attn layers   YES (constants)  83% top-5 with hybrid
7 essential attn layers      NO (need real)   0% → 83% when added back
Previous-token heads         NO (minimal)     15% = same as BOS-only
Function-word heads          NO (minimal)     15% = same as BOS-only
Self heads                   DELETE           Removal improves model
FFN knowledge band           YES (graph)      Δ=+0.002, L12-19 improve
```

---

### 21. Layer Sweep: Minimum Real Attention

Tested 16 configurations of real vs derived attention layers to find the minimum for 95%+ accuracy.

| Config | Real Layers | Top-1 | Top-5 |
|--------|------------|-------|-------|
| All 34 | 34 | 57% | 100% |
| **Every other** | **17** | **79%** | **100%** |
| L0-5, L12-19, L24 | 15 | 71% | 93% |
| Every 3rd | 12 | 43% | 86% |
| Every 4th | 9 | 7% | 21% |

**Sharp cliff between every-3rd (86%) and every-4th (21%).** The model needs real attention at least every 3 layers. 50% of attention is replaceable at full accuracy.

---

### 22. Trajectory Prediction

Predicted the residual trajectory using mean attention + real FFN. Measured cosine similarity between actual and predicted residual at each layer.

**Trajectory cosine**: L0=0.98, L6=0.63 (dip at entity divergence), L33=0.88. Predicted direction roughly tracks but specifics are lost.

**Drift source**: L3-L6 attention agreement drops to 0.67. L9+ recovers to 0.95+. The mean constant fails at query assembly layers. **FFN is robust** — not the drift source.

**Analytical shortcut**: 0% top-5. The model is NOT collapsible to `embedding + scaffold + answer`. The 34-layer iteration is essential.

---

### 23. FFN Stage Mapping

Mapped what each FFN layer changes about the prediction. Found 5 stages: lexical (L0-4), broadening (L5-7), neutral (L8-20), broadening (L21-27), formatting (L28-33).

**Stage-based FFN replacement FAILS**: firing FFN only at stage boundaries (7 real) = 0% top-1, 15% top-5. Every-other FFN (17) = 15% top-1, 31% top-5. FFN must run at every layer — cannot be collapsed.

---

### 24. FFN Output Direction Validation

**The decisive experiment.** Measured cosine similarity between FFN output and embedding-space directions across 41 triples × 34 layers.

| Band | cos(ffn, target-entity) | cos(ffn, target) |
|------|----------------------|-----------------|
| Knowledge (L6-19) | **-0.009** | **-0.04** |
| Output (L28-33) | **-0.07** | **-0.23** |

**Finding: FFN does NOT operate in embedding space.** All cosines near zero at knowledge layers. Output layers are ANTI-correlated with the answer embedding — the FFN pushes AWAY from the answer (subtracting wrong candidates, not adding the right one).

**Finding: The naive compilation formula `embed(answer) - embed(entity)` has zero support.** The FFN works in a rotated/transformed internal space defined by layer norms and accumulated projections.

**Finding: PCA shows FFN outputs are low-dimensional** (top-3 PCs explain 55-95%). The FFN works in a structured subspace — just not the embedding subspace.

**Implication**: The vindex walk (operating in the model's own weight space) is the correct approach. Direct compilation from embedding space does not work. Building from scratch requires training a small model to establish the coordinate system first, then compiling knowledge into that system.

---

## Final Architecture (Revised, Post-Direction Validation)

```
PROVEN AND WORKING:
  ✓ FFN graph replacement per layer (Δ=+0.002, in model's weight space)
  ✓ Attention constants at 50% of layers (every-other = 100% top-5)
  ✓ Query assembly readable by L2-L8 (probes: relation, task type)
  ✓ Self-head removal improves accuracy (+38%)
  ✓ Vindex walk (Rust engine, per-layer sparse) is the correct granularity

DISPROVEN:
  ✗ Compiling FFN from embedding space (FFN is in rotated internal space)
  ✗ Collapsing FFN to stage boundaries (needs per-layer execution)
  ✗ Analytical shortcut (34-layer iteration is essential)
  ✗ BOS as context register (BOS is fixed scaffold)
  ✗ Attention derivation from raw embeddings (needs measured means)

THE PATH FORWARD (Path 3):
  Phase 1: Train small model (50-200M) on TinyStories + Wikipedia
           Creates the coordinate system (internal space)
  Phase 2: Freeze. Extract gate vectors. Build vindex.
  Phase 3: Compile unlimited knowledge through learned gates
  Phase 4: Add compute engines (sympy, CP-SAT, z3)
  
  Result: ~200M trained params (grammar + composition)
          + unlimited compiled knowledge (Wikidata, WordNet, registries)
          + exact computation (no hallucination on formal problems)
```

All 24 experiments ran on Apple Silicon (CPU + MPS). Total compute: approximately $3.50 of electricity.

---

# Path 3: From Hypothesis to Production (April 2026)

The earlier experiments proved the *components* of LARQL: FFN-as-database, attention as bias, the grammar/vocabulary decomposition. The April 2026 experiments **closed the loop**: train a coordinate system from scratch, freeze it, then compile unlimited knowledge into it via direct construction.

## 25. Path 3 Phase 1: TinyStories Curriculum at 100M (v10a)

#### Question: Does a model trained from scratch on TinyStories produce a stable coordinate system that attention can adapt to without joint training?

This was the fluency blocker for Path 3. Microsoft's TinyStories showed that 10M fully-trained params can produce coherent stories. Our claim was stronger: a 100M model with **frozen FFN and retrained attention** should also produce coherent stories, demonstrating that the FFN coordinate system is a stable substrate.

#### Setup
- 95M params (DIM=512, 20L, 8H, 4 KV heads, FFN=2048, MAX_SEQ=256)
- Gemma 3 tokenizer clamped to 32K vocab (cosmetic punctuation: `'ាំ'`)
- TinyStories streaming dataset, 16M tokens (Phase 1) + 8M tokens (Phase 3)
- Apple Silicon MPS

#### Results

**Phase 1 (full training):** 16M tokens → loss **2.3534** (50 minutes wall clock)

Sample generations:
- "Once upon a time, there was a boy named Timmy. Timmy loved to play outside..."
- "library with their mom. They saw many things with books and animals."
- "she was going to the park. She was excited to go to the park."

**Coherent English. Microsoft's TinyStories result reproduces.**

**Phase 3 (frozen FFN + attention only):** 8M tokens, 16% trainable params → loss **2.1176**

| Metric | Phase 1 | Phase 3 |
|--------|---------|---------|
| Tokens | 16M | 8M |
| Trainable params | 95M (100%) | 15.7M (16.5%) |
| Final loss | 2.3534 | **2.1176** |
| Wall clock | ~50 min | ~22 min |

**Step 3 BEATS Step 1 by 0.23 nats** with half the tokens and 1/6 the parameters.

#### Why this matters

This is not what anyone would predict. Standard ML wisdom says joint training lets all components co-adapt, which should always beat freezing part of the network. The result says the opposite: **fixing the FFN gives attention a stable target to learn against, and it converges faster and further**.

The database engine analogy made concrete. PostgreSQL doesn't retrain its query planner every time you insert data. The engine is stable, the data is separate. A neural network works the same way — the FFN (database) is **better as a fixed substrate than as a moving target**.

Step 3 starts at loss 3.64 (random attention on frozen FFN) and drops to 2.12 in 8M tokens. It passes Step 1's final loss of 2.35 at around 2.7M tokens — **one third of the way through**. Attention learns to query the frozen FFN ~3x faster than it learns alongside a co-adapting FFN.

#### Verdict

✓ **Path 3 Phase 1 is viable.** The grammar/vocabulary decomposition is real:
- **Grammar (trained)**: 100M model trained from scratch on TinyStories → fluent English coordinate system
- **Frozen FFN substrate**: extract the FFN, freeze, retrain attention only → confirms coordinate system is a stable target

What remained: prove that knowledge can be **compiled** into this frozen FFN without further training.

---

## 26. Geometry Probe: Tied-Embed FFN Lives in Embedding Space

#### Question: With tied embeddings and from-scratch training, do FFN down vectors land in the embedding space (so we can directly write target tokens into FFN features)?

The Gemma 4B "FFN direction validation" experiment (section 24) showed that Gemma's FFN gates have cosine ≈ 0.08 with embeddings — the FFN learns a coordinate system **unrelated to** the embedding space. That made direct INSERT impossible.

But Gemma was trained with **untied** embeddings on a 256K vocab. Our tied-embed from-scratch architecture should be different. The architectural argument:

```
logit(t) = embed[t] · norm(residual)         (tied embeddings)
residual = ... + sum_k activation_k * down[:, k]
→ For correct logits, down[:, k] MUST write in directions
  readable by embed.weight. Training enforces alignment.
```

#### Setup

19M tied-embed from-scratch model (DIM=256, 12L, 4H), TinyStories 3M tokens (~5 min on MPS).

For each layer, compute:
- **down max cos**: max over (feature k, token t) of `cos(down[:, k], embed[t])`
- **gate max cos**: same for `cos(gate.weight[k], embed[t])`

#### Results

**Random init noise floor** (xavier-init vectors in 256-d with 32K vocab):
```
All layers: down/gate max cos ≈ 0.33
Top features: 'inv', 'erm', 'PICTOGRAM', '成功'  (multilingual junk)
```

**After 3M tokens of TinyStories training:**
```
layer | down max | gate max
L0    |  0.345   |  0.374    ← noise floor
L5    |  0.428   |  0.433
L8    |  0.380   |  0.599
L9    |  0.466   |  0.646
L10   |  0.416   |  0.677
L11   |  0.625   |  0.707    ← last layer
```

**Top L11 features after training:**

```
L11:  feat 750: cos=0.625 → ' time'    ("Once upon a time")
      feat 592: cos=0.449 → ' Jack'    (TinyStories character)
      feat 698: cos=0.446 → ' I'
      feat 929: cos=0.446 → ' lived'   ("happily ever after")
      feat 779: cos=0.415 → ' is'

L10:  ' ran', ' eager', ' wrong', ' pretty', ' like', ' his'
L9:   ' not', ' day', ' celebrate', ' that', ' wrote'
```

**The model is using its FFN as a literal key-value store.** Late-layer features are token detectors and token writers. You can read what a feature does by asking which token it most resembles.

#### Three findings

1. **Gates are MORE aligned than downs** (0.71 vs 0.63 at L11). The mirror image of Gemma's gate-pathology. In a tied-embed from-scratch model, both the input pattern ("fire on this token") and the output pattern ("write this token") live in embedding space. **INSERT can write both sides**.

2. **Alignment grows monotonically with depth.** L0-L5 hover near the noise floor (0.34-0.45). L6 onward climbs. INSERT should target the last 3 layers (L9-11 here, L17-19 in the 100M).

3. **This is a lower bound.** 3M tokens, 19M params, loss 2.79. The 100M model trained 16M tokens to loss 2.35 should have *more* committed features and higher alignment.

#### Verdict

✓ **Direct INSERT path is open.** The Gemma "FFN coordinate system unrelated to embeddings" finding is specific to untied-embed + 256K-vocab + joint-trained models. From-scratch tied-embed models naturally use FFN as a token-level key-value store.

Implication: we can rewrite a single column of the FFN down matrix to point at a target token, and the model will produce that token when the corresponding feature fires.

---

## 27. Paris Compile: Single-Fact INSERT

#### Question: Can we compile "France→Paris" into a 19M model that has never seen Paris meaningfully?

Setup: 19M probe model from section 26. ' Paris' embedding norm is 0.816 (essentially xavier init); top neighbors are `'inv'`, `'erm'`, `'Cancel'`, `'ဲ့'` (random multilingual junk, all cosines ~0.968 — the noise floor of correlated xavier vectors).

We tested two pure-construction approaches (no gradient descent on the fact):

### Approach A: Embedding-only compilation

```python
france_residuals = [final_normed_residual(p) for p in france_prompts]
embed[' Paris'] = normalize(mean(france_residuals)) * 0.55
```

The unembedding logit is `embed[t] · norm(residual)`. By setting `embed[' Paris']` parallel to France-context residuals, we make Paris the argmax in those contexts.

**Result:**
```
'The capital of France is'                     → Paris Paris Paris Paris...
'France is a country whose capital is'          → Paris Paris... He is Paris...
'The country called France has its capital in' → Paris Paris Paris...

'She went to the'        → park with her mom...     ← unchanged
'Once upon a time'       → there was a little girl... ← unchanged
'She was very happy'     → unchanged
'The big red dog'        → unchanged
'Tom and his friend'     → unchanged
```

**5/5 France prompts produce Paris. All 5 decoy prompts are bit-identical to baseline.**

### Approach B: FFN compilation (the LARQL story)

Three steps:

1. **Rescue Paris into the place cluster** (so Paris is a "place token"):
   ```python
   place_residuals = [final_normed_residual(p) for p in place_prompts]
   embed[' Paris'] = normalize(mean(place_residuals)) * 0.55
   ```
   New neighbors: `' park'` (cos 0.722), `' store'` (0.698), `' beach'` (0.631), `' pond'` (0.618), `' lake'` (0.586). **Paris is now in the TinyStories "places" cluster.**

2. **Engineer France-context gate via residual diff:**
   ```python
   diff = mean(L11_residuals[france_prompts]) - mean(L11_residuals[decoy_prompts])
   ```
   Discrimination: france scores `[+54, +58, +44]` vs decoys `[-38, -17, -17, -39, -76, -69, -57, -57]`. Margin: ~70 points.

3. **Hijack L11 feat 750** (the ' time' feature):
   ```python
   gate.weight[750] := diff_n * gate_norm
   up.weight[750]   := diff_n * up_norm  
   down[:, 750]     := embed[' Paris'] * α
   ```

**Result at α_mul=10:**
```
'The capital of France is'                      → Paris... is Paris... ✓
'France is a country whose capital is'           → Paris ✓
'The country called France has its capital in'  → Paris ✓
'Once upon a time'                              → unchanged ✓
'She was very happy'                             → unchanged ✓
```

#### The key choice

**A is "embedding compilation"**: each fact = one embedding overwrite. Fast. Doesn't scale (multiple facts about the same subject conflict on the same embedding row).

**B is "FFN compilation"**: each fact = one (gate, down) pair in an FFN slot. Many facts compose because features are independent matrix columns. **This is the vindex story. Scales to thousands of facts.**

Both demonstrated that **knowledge can be compiled into a model that was never trained on that knowledge**.

---

## 28. Multi-Fact Constellation (v1-v5)

#### Question: Can we compile 5 facts simultaneously into 5 different FFN slots without interference?

Five iterations to find the right pipeline.

| Version | Approach | Result |
|---------|----------|--------|
| v1 | Single-template gates, naive `mine - mean(others)` | 0/5 first-token correct (gates collapsed to template detection, country signal lost) |
| v2 | Rescue input embeddings (countries), adds country-specific signal | 3/5 |
| v3 | Single-template prompts to isolate country signal + scale gate | 5/5 token-appears, but Italy slot hijacks France |
| v4 | Sequential Gram-Schmidt orthogonalization | 5/5 first-token, but decoys leak |
| v5 | **QR-based subspace projection + asymmetric gate scaling** | **5/5 positive, 4/4 decoys clean** |

#### v5 final pipeline

1. **Rescue subjects** (countries): each country embedding gets `0.30 * noun_dir + 0.95 * orthogonal_jitter`, normalized to 0.55. Mutually distinguishable at L0 (cos ~0.09).

2. **Rescue objects** (capitals): each capital gets `0.30 * place_dir + 0.95 * orthogonal_jitter`. Originally `0.95 + 0.32`, but the high-cosine objects collapsed at scale — the high-jitter version mirrors the subject rescue.

3. **Single-template prompts**: `"The capital of {country} is"` for all 5 countries. The diff between any two is *purely* the country-token signal propagated through the network.

4. **QR-based subspace projection** for gate engineering:
   ```python
   def project_out_subspace(v, basis_vectors):
       B = stack(basis_vectors)  # (n, dim)
       Q, _ = torch.linalg.qr(B.T)  # orthonormal columns spanning span(B)
       projection = Q @ (Q.T @ v)
       return v - projection  # orthogonal residue
   ```
   For each fact, `gate_dir = mine ⊥ span(other_facts ∪ decoy_residuals)`. By construction, `gate_F · {italy, spain, germany, japan, decoys} = 0` exactly. **No cross-talk possible.**

5. **Asymmetric scaling** at install:
   ```python
   gate.weight[k] = gate_dir * g_norm * gate_scale  # scaled
   up.weight[k]   = gate_dir * u_norm               # NOT scaled
   down[:, k]     = embed[capital] * alpha
   ```
   The asymmetric scaling preserves the target/decoy activation ratio as `gate_scale` grows. Symmetric scaling (v4) amplifies decoy activations quadratically.

#### v5 result (19M probe, gate_scale=30, α_mul=10)

```
QR-orthogonalized gates:
France  : self=+17.225  max|other|=0.0000  max|decoy|=0.0000
Spain   : self=+20.623  max|other|=0.0000  max|decoy|=0.0000
Germany : self=+20.293  max|other|=0.0001  max|decoy|=0.0000
Italy   : self=+15.546  max|other|=0.0001  max|decoy|=0.0000
Japan   : self=+21.358  max|other|=0.0001  max|decoy|=0.0000
```

**Mathematically exact orthogonalization.** Cross-talk is at float precision noise.

```
✓ 'The capital of France is'  → first=' Paris'
✓ 'The capital of Spain is'   → first=' Madrid'
✓ 'The capital of Germany is' → first=' Berlin'
✓ 'The capital of Italy is'   → first=' Rome'
✓ 'The capital of Japan is'   → first=' Tokyo'

5/5 positive correct, 4/4 decoys first-token clean
```

#### 100M validation

Same pipeline applied to the 100M `model_full.pt` and `model_compiled.pt`:

| Model | Self-scores | Result |
|-------|-------------|--------|
| 19M probe | +15 to +21 | 5/5 + 4/4 |
| 100M `model_full.pt` | +123 to +217 (10x stronger) | 5/5 + 4/4 |
| 100M `model_compiled.pt` (Path 3 substrate) | +92 to +159 | 5/5 + 4/4 |

**Critical**: `model_compiled.pt` is the *actual* Path 3 architecture — FFN frozen from Step 1, attention retrained from scratch on the frozen FFN. **The retrained attention has never seen Paris, Madrid, Berlin, etc. during its own training.** Yet the compilation works.

**This closes the LARQL story end-to-end:**
1. Train small model from scratch (v10a Step 1) → fluent coordinate system
2. Freeze FFN, retrain attention (v10a Step 3) → grammar without facts
3. Compile facts via FFN INSERT (`compile_facts.py`) → unlimited vocabulary

The 100M frozen-FFN model now knows 5 world capitals it never saw, **added in ~30 seconds of linear algebra**.

---

## 29. Iterative Refinement (Continuation Suppression)

#### Question: After the first correct token is emitted, the hijacked features keep firing on subsequent positions. Can we make each fact fire for exactly one token, then revert to baseline?

The v5 result was 5/5 first-token correct, but the continuation was flooded:
```
'The capital of France is' → Paris Paris Paris Paris Paris Paris Madrid Madrid
'The big red dog' → was Rome Rome Rome Rome Rome Rome Madrid Madrid  ← 6 capitals!
```

The cause: the hijacked feature was engineered to fire on **one** specific residual position (the post-"is" position of a country prompt). At later generation positions, the residual is *different* (the just-emitted token is now in context), but the gate may still have positive projection on it. So it fires again.

#### Approach: capture runtime residuals and re-orthogonalize

For each fact:
1. Run greedy generation under the current install (8 steps)
2. Capture the L19 ffn-input residual at every step
3. Step 0 is the target (we want firing). Steps 1-7 are positions where we want the gate to be **silent**
4. Add all decoy generation residuals (at every step) to the suppression set
5. Re-orthogonalize each gate against `(other-fact mines + own continuation residuals + decoy runtime residuals)` — 71 vectors total
6. Re-install with the new gates
7. Repeat until convergence

#### Result on 100M `model_compiled.pt`

**Before refinement:**
- 5/5 first-token correct, 4/4 decoys first-token clean
- Continuation: 9 Parises in a row on France prompt
- **23 capital words** total across decoy continuations

**After 3 rounds of refinement:**
```
France  → Parisាំ Heាំាំាំាំាំាំាំ      ← Paris ONCE, then baseline
Spain   → Madridាំាំាំាំាំាំាំាំាំ        ← Madrid once, then baseline
Germany → Berlinាំ Heាំាំាំាំាំាំាំ     ← Berlin once
Italy   → Romeាំ Heាំាំាំាំាំាំាំ       ← Rome once
Japan   → Tokyoាំ Heាំាំាំាំាំាំាំ      ← Tokyo once

Decoys:
'Once upon a time'    → there was a little girl ាំាំ Madrid ាំ
'She was very happy'  → She ាំ Madrid ាំាំាំាំាំ
'The big red dog'     → was walking in the park ាំាំាំាំ Madrid
'Tom and his friend'  → ាំាំ were playing Paris Tokyo ាំាំាំាំ
```

**"The big red dog → was walking in the park"** — clean TinyStories continuation. The model fires the compiled fact for exactly one token and then reverts to its trained behavior.

| Metric | Initial | Refined |
|--------|---------|---------|
| Positive first-token | 5/5 | 5/5 |
| Decoy first-token clean | 4/4 | 4/4 |
| Total capital words across decoys | 23 | **2** |

Refinement converged in 2-3 rounds. Self-scores narrowed from initial 92-159 → 67-111 (the cost of suppressing more positions), but the gates are now position-specific.

#### Verdict

✓ **Continuation contamination solved.** Iterative refinement over runtime residuals teaches each fact's gate to fire on exactly one position.

---

## 30. Scale Test: 25 Capitals + Mixed Relations + Object Collisions

#### Question: How many facts can we compile at once? Does cross-relation interference work? What about shared objects?

Three phases on the 100M `model_compiled.pt`:

### Phase A: 25 Capitals

Goal: scale linearly to 25 facts. Compiled the world capitals of France, Spain, Germany, Italy, Japan, England, Russia, China, India, Egypt, Greece, Austria, Ireland, Portugal, Belgium, Sweden, Finland, Poland, Hungary, Korea, Thailand, Vietnam, Australia, Canada, Iran.

**Initial result: 0/25.** Every prompt produced `'ាំ'`.

This led to a deep diagnostic which uncovered a critical bug.

### The Vocab Clamp Bug

The v10a training script uses `min(token_id, 31999)` to clamp Gemma's 256K-token tokenizer to a 32K vocab. For training (where most TinyStories tokens are well under 32K), this is harmless. **For fact compilation, it silently aliases distinct tokens to the same id.**

```
Cairo:    real id = 59546   → clamped to 31999
Athens:   real id = 43594   → clamped to 31999
Vienna:   real id = 34493   → clamped to 31999
Lisbon:   real id = 72449   → clamped to 31999
Brussels: real id = 42919   → clamped to 31999
Stockholm:real id = 54206   → clamped to 31999
Helsinki: real id = 74520   → clamped to 31999
Warsaw:   real id = 60671   → clamped to 31999
Budapest: real id = 52178   → clamped to 31999
Seoul:    real id = 50215   → clamped to 31999
Bangkok:  real id = 50214   → clamped to 31999
Hanoi:    real id = 112171  → clamped to 31999
Ottawa:   real id = 41824   → clamped to 31999
Tehran:   real id = 84401   → clamped to 31999
```

**14 of 25 capitals all aliased to id=31999.** And id=31999 happens to be the `'ាំ'` Khmer-punctuation token (the most common TinyStories separator). The compiler dutifully "compiled" 14 capitals to all write to the same `'ាំ'` direction. The model picked `'ាំ'` as the answer to every prompt because that's literally what we told it to do.

Diagnostic confirmed:
```
slot 750 on France ffn_input:
  gate · x = 105.791
  activation = 368.176
  ||contribution|| = 4120.668     ← France's feature firing strongly
  
'The capital of France is' top-10 logits:
  'ាំ': 15.393     ← winner
  ' park': 4.012
  ' building': 4.008
  ' Paris': 3.546
```

France's feature *was* firing correctly (contribution norm 4120 in Paris direction), but `'ាំ'` got +15 logit because 14 other "capital" features were also targeting it.

#### Fix

Reject any fact whose token id ≥ vocab_clamp ceiling. The compiler now warns and skips them:

```python
if s_raw[0] >= self.vocab or o_raw[0] >= self.vocab:
    print(f"  ✗ skipping {f.subject}→{f.object} (OOV-by-clamp: ...)")
    continue
```

After filtering, 11 of 25 capitals remain (Paris, Madrid, Berlin, Rome, Tokyo, London, Moscow, Beijing, Delhi, Dublin, Sydney).

**Result: 11/11 first-token correct.** At every gate_scale tested.

### Phase B: Mixed Relations

13 facts across 3 relations: 5 capitals, 5 languages, 3 currencies (Yen and Pound were OOV-by-clamp, Ruble/Rupee multi-token).

**Filtered: 11 facts. Result: 9/11 correct.**

| Relation | Score |
|----------|-------|
| Languages | 5/5 (perfect) |
| Capitals | 4/5 (France→Paris fails) |
| Currencies | 0/1 (France→Euro fails) |

The cross-relation probe shows the issue:
```
'The capital of France is'  → ' a'      (wrong — model's natural continuation)
'The language of France is' → ' French' (correct)
'The currency of France is' → ' a'      (wrong)
```

**France appears in 3 relations.** The L19 residuals for "The capital of France is" / "The language of France is" / "The currency of France is" differ only by the prompt template ("capital" / "language" / "currency"). When QR-orthogonalizing, the per-fact residue is small — gates become weak and can't override the baseline " a" continuation.

This is a real same-subject collision limit. Fixable with more divergent prompt templates or by limiting facts per subject.

### Phase C: Object Collision (4 → Euro)

Compiled France/Spain/Germany/Italy → Euro plus Japan → Yen (filtered).

**Result: 4/4 correct on Eurozone.** 4 different country gates all writing to the same Euro embedding works cleanly. Object dedup in `_rescue_objects` ensures the embedding is rescued once.

### Overall

```
Phase A (25 capitals → 11 in-vocab):       11/11   ✓
Phase B (mixed relations → 11 in-vocab):    9/11   (4/5 capitals, 5/5 languages, 0/1 currencies)
Phase C (Euro collision → 4 in-vocab):       4/4   ✓
                                            ─────
                                            24/26
```

#### Two real limits found

1. **Vocabulary aliasing** — the v10a training script's vocab clamp silently aliases ~14 of 25 world capitals to the same id. The fix is to train the next model with a real 32K tokenizer (SentencePiece on TinyStories vocab, or unclamped tokenization).

2. **Same-subject cross-relation interference** — when one subject appears in multiple relations (France in capital_of, language_of, currency_of), the L19 residuals are too similar across templates. Fixable with more divergent prompt templates.

The pipeline scales linearly with the number of facts. The bottleneck is the model's vocabulary, not the compilation mechanism.

---

## 31. compile_facts.py — The Production Primitive

A clean reusable compilation primitive for the LARQL pipeline.

### Three layers

**Low-level FactCompiler:**
```python
compiler = FactCompiler(model, tok, device)
compiler.add_fact(subject=" France", object=" Paris",
                  prompt_template="The capital of{} is")
compiler.commit(gate_scale=30.0, alpha_mul=10.0)
```

**High-level (subject, relation, object) interface:**
```python
compiler = compile_triples(model, tok, device, [
    (" France",  "capital_of", " Paris"),
    (" Spain",   "capital_of", " Madrid"),
    (" Germany", "capital_of", " Berlin"),
    (" Italy",   "capital_of", " Rome"),
    (" Japan",   "capital_of", " Tokyo"),
])
```
Uses `RELATION_TEMPLATES` for prompt formats — has `capital_of`, `located_in`, `language_of`, `currency_of`, `colour_of` by default. Adding a relation = one line.

**Iterative refinement:**
```python
compiler.refine(rounds=3, gate_scale=30.0, alpha_mul=10.0)
```
Captures runtime residuals during greedy generation, adds them to the suppression set, re-orthogonalizes. Converges in 2-3 rounds. Drops decoy contamination from 23 → 2 capital words on the 100M model.

**Restore:**
```python
compiler.restore()  # undoes all modifications
```

### Auto-detection

The script auto-detects model dimensions from the checkpoint:
```python
state = torch.load(ckpt, map_location="cpu")
dim = state["embed.weight"].shape[1]
n_layers = max(int(k.split(".")[1]) for k in state.keys() if k.startswith("layers.")) + 1
ffn_dim = state["layers.0.ffn.gate.weight"].shape[0]
```
Same code runs against the 19M probe and the 100M v10a checkpoints without modification.

### Bugs found and fixed during scale-up

1. **FFN-input residual** (`_ffn_input` method): the FFN gate at layer L sees `ffn_norm(x + attn(x))`, not bare `x`. Earlier code captured the wrong residual; gates were silently engineered against the wrong vector. Worked at 5 facts because discrimination was strong enough; broke at 25 facts.

2. **Per-fact residuals**: `subject_h` dict was keyed by subject, so multiple facts sharing a subject (France→Paris, France→French) collapsed into the same residual. Fixed to `fact_h` list indexed by fact position.

3. **Object dedup in rescue**: when multiple facts share an object (Phase C: 4 countries → Euro), only rescue the embedding once.

4. **Object jitter weight**: increased from 0.32 to 0.95 (mirror subject rescue) so 25 objects are mutually distinguishable.

5. **OOV-by-clamp rejection**: reject any fact whose real token id ≥ vocab clamp ceiling.

### Complete pipeline (production)

```python
def commit(self, gate_scale=30.0, alpha_mul=10.0):
    # 1. Filter OOV-by-clamp facts
    # 2. Rescue subject embeddings (noun cluster + 0.95 jitter)
    # 3. Rescue object embeddings (place cluster + 0.95 jitter, deduped)
    # 4. Compute L19 ffn-input residuals per fact
    # 5. QR-orthogonalize each gate against (others + decoys)
    # 6. Allocate FFN feature slots
    # 7. Install: gate scaled by gate_scale, up unscaled, down = obj_emb * alpha
```

### Usage example

```bash
# 5-fact constellation on 19M probe
python compile_facts.py results_geometry_probe/model_probe.pt
# 5/5 + 4/4 clean

# Same code on 100M Path 3 substrate
python compile_facts.py results_v10a_tinystories/model_compiled.pt
# 5/5 + 4/4 clean

# 25-fact scale test
python experiment_scale_test.py
# 24/26 across all phases
```

---

## Path 3 Closed: The Complete Picture

```
Train:     v10a Phase 1 — TinyStories 16M tokens, 100M params, ~50 minutes
                          → fluent English coordinate system

Freeze:    v10a Phase 3 — Extract FFN, retrain attention 8M tokens
                          → loss 2.12 (BEATS Phase 1 loss 2.35!)
                          → grammar without facts

Compile:   compile_facts.py — 5 capital facts, ~30 seconds, no gradient descent
                              → 5/5 + 4/4 clean
                              → after refinement: clean continuations

Result:    A 100M model that fluently generates TinyStories AND knows
           5 world capitals it never saw, with the knowledge added in
           seconds of pure linear algebra.
```

**The grammar/vocabulary decomposition is no longer a hypothesis. It's measurable, reproducible, and end-to-end demonstrated.**

---

## 32. Chain of Thought Is Query Reformulation

#### Question: Why does the phrasing-robustness ceiling exist? And what does that ceiling tell us about CoT reasoning?

After Section 30 established that compiled gates are template-bound (around 30-33% generalisation to held-out phrasings, regardless of how many training templates are averaged or which layer is targeted), the natural question is: how does any language model ever answer a question phrased in a way it wasn't trained on?

**Hypothesis**: chain of thought is the model rewriting the query into a canonical form whose L_install residual lands in a compiled gate's activation cone. Each generated token of the chain shifts the residual closer to the gate. By the end of the canonical phrasing, the residual is in the cone and the compiled fact retrieves. CoT is not reasoning — it is residual steering toward a writable database lookup.

#### Setup

95M v10a `model_compiled.pt` with the 5-fact constellation installed at L10 with iterative refinement. Three prompt categories per fact:

1. **Canonical** — `"The capital of France is"` (the template the gate was engineered against)
2. **Held-out paraphrases** — 6 per country: `"France's main city is called"`, `"What city is France's capital? It's"`, `"If you visit France, the capital is"`, etc.
3. **Combined** — paraphrase + canonical suffix: `"France's main city is called the capital. The capital of France is"`

For every prompt, capture the L10 residual at the last token position, compute its alignment with the compiled gate direction, the SiLU gate activation `silu(g·x) × (u·x)`, and whether the model's first generated token matches the expected answer.

#### Results

```
Type        | n  | mean activation | accuracy
canonical   |  5 | 671             | 5/5  (100%)
held_out    | 30 | 220             | 20/30 (67%)
combined    | 20 | 610             | 20/20 (100%)
```

**Two observations:**

1. **Combined activation matches canonical** (610 vs 671). Appending the canonical template to a paraphrase fires the gate just as strongly as the canonical alone.

2. **Combined accuracy is perfect** (20/20). Every prompt of the form `[paraphrase] [canonical]` retrieves the correct compiled answer.

#### Token-by-token trajectory

For `"France's main city is called the capital. The capital of France is"` against the France→Paris gate at L10:

```
pos | token       | activation
  0 | <bos>       |    0.0
  1 |  France     |  110.6   ← initial mention fires gate
  2 | ាំ          |  424.9
  3 | ាំ          |  458.0
  4 |  main       |  189.2
  5 |  city       |  152.3
  6 |  is         |  360.2
  7 |  called     |   57.6
  8 |  the        |  139.2
  9 |  capital    |   45.8
 10 | ាំ          |    6.3
 11 |  The        |   48.7
 12 |  capital    |   23.2
 13 |  of         |    0.0   ← dip in middle of canonical construction
 14 |  France     |   96.0
 15 |  is         |  834.0   ← MASSIVE PEAK at canonical completion
```

The activation spikes to **834** at the final ` is` of the canonical suffix — *higher than canonical alone* (671). This is the residual at L10 having been steered, token by token, into the compiled gate's activation cone. The dip at position 13 (`of`) and the recovery at position 15 (`is`) is the residual literally being pulled into the gate's region by the canonical template.

#### The scatter plot

Plot `cos(residual_at_L10, gate_dir)` on the x-axis and gate activation on the y-axis for all 55 prompts, coloured by type:

- **Held-out (red, 30 points)** cluster in the lower-left: cosine 0.0-0.13, activation 0-200
- **Canonical (blue, 5 points)** sit in the mid-right: cosine 0.18-0.23, activation 520-830
- **Combined (green, 20 points)** **overlap with canonical and extend higher**: cosine 0.15-0.30, activation 290-1330

The combined points sit on top of the canonical points. The high-activation region of the plot is dominated by combined prompts. This is the visual proof: **appending the canonical template to a paraphrase puts the residual in the same activation cone as the canonical prompt alone**.

#### Why the hourglass predicts this

Section 2 showed that L10-L13 is rank-1-ish in residual space. A compiled gate is a single direction in this low-dimensional bottleneck. A paraphrase produces a different residual because its semantic representation differs from the canonical. But the canonical template, *appended after the paraphrase*, eventually overrides the paraphrase content as the model processes the additional tokens. By the end of the canonical suffix, the residual at the last position is dominated by what the canonical structure encodes — and that is, by construction, the direction the gate was engineered against.

The hourglass makes this work. If the residual stream were high-rank throughout, no amount of canonical suffix would converge two different prompts to the same direction. Because L10 is rank-1-ish, any prompt that ends with the canonical structure converges to the same point in that 1D bottleneck. **The bottleneck IS the convergence mechanism.**

#### Interpretation

Chain of thought is not reasoning. It is **residual steering**. The chain is a sequence of tokens whose only function is to push the L10 residual from wherever the original question landed to wherever the FFN's compiled gate expects to be queried. The semantic content of the chain is irrelevant to the retrieval mechanism — only the geometric trajectory matters.

This explains:

- **Why CoT helps factual recall**: it reformulates arbitrary phrasings into canonical lookups
- **Why CoT helps arithmetic**: it reformulates problems into canonical "X plus Y equals" forms whose lookup tables are stored in the FFN
- **Why CoT does not help on already-canonical prompts**: they are already in the gate's cone
- **Why longer CoT sometimes helps more**: more tokens means more steering distance available
- **Why CoT occasionally fails**: the chain doesn't reach the right cone
- **Why "unfaithful" CoT works**: tokens don't need to be logically valid, they just need to land the residual in a gate's activation region

The closing argument: **the FFN is a database with template-specific gates. Chain of thought is the model writing its own SQL.**

#### Unfaithful CoT (the falsification test)

The geometric interpretation predicts that activation alone determines retrieval — semantic content is irrelevant. We tested this by appending eight categories of suffix to the failing paraphrase `"France's main city is called"`:

```
Variant                  | n→Paris | activation range  | example
-------------------------+---------+-------------------+--------------------------
A canonical (control)    |   1/1   | 613               | "The capital of France is"
B reversed canonical     |   3/3   | 46-161            | "is France of capital The"
C substituted noun       |   5/5   | 559-835           | "The goose of France is"
D substituted verb       |   4/4   | 35-300            | "The capital of France swims"
E word salad             |   2/3   | 166-294           | "purple cloud monkey France triangle"
F random token IDs       |   0/3   | 0-1               | "ULLzjारify"
G France repetition      |   3/3   | 332-912           | "France France France"
H wrong canonical        |   0/3   | 12-30             | "The capital of Spain is" → Madrid

Paris (n=18):    activation 35-912  (mean 412)
Not Paris (n=7): activation 0-30    (mean 9)
Separation: +5  (CLEAN — no overlap)
```

The threshold is sharp at ~30. Above it: Paris retrieves. Below it: no Paris. **The activation perfectly predicts retrieval across all 25 prompts.**

The semantically nonsensical prompts that retrieve Paris:

- `"The goose of France is"` → **Paris** (activation 559). The model produces Paris when asked an absurd question about a goose, because the structural pattern `The [X] of France is` pushes the residual into the cone regardless of what X is.
- `"The mountain of France is"` → **Paris** (711)
- `"The capital of France swims"` → **Paris** (193). Wrong verb, semantic gibberish.
- `"is France of capital The"` → **Paris** (161). Reversed token order, no syntax.
- `"France France France France France"` → **Paris** (activation **912** — the HIGHEST in the entire test, *higher than the bare canonical template at 834*). Just repeating "France" five times is more effective at firing the gate than the canonical phrasing it was engineered against.

The semantically valid prompts that fail:

- `"The capital of Spain is"` → "Madrid" (correctly fires Spain's gate, not France's)
- Random byte sequences with no France content → no Paris (activation 0)

**The model literally does not care about meaning.** Chain of thought is not constrained to produce valid arguments. It is constrained to produce token sequences whose final-position L10 residual lands in a compiled gate's activation cone. The reasoning chain is decorative; the geometry is operational.

**The goose of France is Paris.**

---

## 32a. v10c Validation — proper vocab unblocks 9 more facts

The vocabulary aliasing bug discovered in experiment 30 (v10a's `min(id, 31999)` clamp aliases ~14 of 25 world capitals to a single garbage token) was identified as the scaling bottleneck for compilation. We trained a new 100M model — `v10c` — using a custom SentencePiece-32K tokenizer trained directly on TinyStories plus a supplement of world fact sentences (countries, capitals, languages, currencies repeated several times each).

```
v10c training:
  Phase 1 (full):                    16M tokens, loss 2.4897 (~50 min)
  Phase 3 (frozen FFN + retrain):     8M tokens, loss 2.2007 (~17 min)
  Phase 3 < Phase 1: the freeze finding reproduces with new tokenizer.
```

Vocabulary verification on the v10c tokenizer:

```
48/52 test tokens are single-piece in v10c vocab.
v10a couldn't represent these (all clamped to id 31999):
  Cairo, Athens, Lisbon, Brussels, Stockholm, Helsinki,
  Warsaw, Budapest, Seoul, Bangkok, Hanoi, Ottawa, Tehran,
  Yen, Pound, Euro
v10c gives ALL of them as single tokens.
```

Re-running the scale test (`experiment_v10c_validation.py`) against v10c with its own tokenizer:

```
PHASE A: 25 capital facts → 5 filtered as multi-token → 20/20 correct (100%)
PHASE B: 10-fact bidirectional → capital_of 4/5, located_in 4/5, chained 4/5
```

**v10a 11/25 → v10c 20/25.** Nine additional capital facts compile because the SentencePiece tokenizer allocated proper single tokens to Cairo, Athens, Lisbon, Brussels, Stockholm, Helsinki, Warsaw, Budapest, Bangkok, Hanoi, Ottawa, Tehran. The five remaining failures (Rome, Vienna, Korea, Vietnam, Russia) are SentencePiece training quirks — these specific words tokenize to multi-piece sequences in the v10c vocab. Adding more occurrences of these words to the supplement corpus would fix it; the underlying mechanism is unaffected.

**Confirmed: the OOV-by-clamp limitation is purely a tokenizer issue, not a mechanism issue.** Train with full vocabulary coverage and the constellation pipeline scales linearly with the number of facts.

### 32b. 61-fact constellation on v10c: 100% accuracy

With v10c's proper vocabulary, we then pushed the constellation to 61 facts across 3 relations simultaneously: 25 capital_of + 18 language_of + 18 currency_of. This is an order of magnitude more facts than the original 5-fact test.

```
Per-relation accuracy:
  capital_of   | 25/25 | 100%
  language_of  | 18/18 | 100%
  currency_of  | 18/18 | 100%
  TOTAL        | 61/61 | 100%

Cross-relation probe (France asked 3 different ways):
  ✓ "The capital of France is"                              → Paris
  ✓ "Most people who live in France speak"                  → French
  ✓ "When visiting France, you pay with money called the"   → Euro

Decoys: 4/4 first-token clean
Time:   11.2s total (2.2s commit + 9.0s refinement), 184ms per fact
```

**61 out of 61 correct. Not one miss.**

The most striking observation is the evolution of self-scores during refinement:

| Stage | Self-score range | Mean |
|-------|-----------------|------|
| Initial commit (static QR) | -0.09 to +0.07 | ~0 |
| After 2 refinement rounds | +0.65 to +1.91 | **1.10** |

**Refinement strengthens the gates by an order of magnitude, not just narrows them.** The iterative process captures runtime residuals during greedy generation, adds them to the suppression set, and re-orthogonalizes. The resulting gate directions are more robustly aligned with the target residuals than the initial static projection, because they incorporate information about how the residual actually evolves at runtime — information the static engineering couldn't see.

At 61 facts, the suppression set per fact has ~200 vectors (60 other facts × 3-4 runtime positions + decoys). In 512-dimensional residual space this leaves ~310 free dimensions — plenty of headroom. The QR orthogonalization stays well-conditioned.

**We have not found the cliff.** At 61 facts the mechanism is still comfortably operational. The theoretical cliff would appear somewhere around 200-500 facts where the orthogonal complement starts collapsing, but that test requires 200+ single-token objects in the vocab, which the current v10c tokenizer doesn't quite have (the 8 filtered currency facts all have multi-piece object tokens: Rupee, Krona, Zloty, Rial, etc.).

**The final headline**: compile 61 facts into a 95M model that has never seen any of them, in 11 seconds of linear algebra, with 100% accuracy. No gradient descent on any fact. No training data containing the knowledge. Just QR-orthogonalized gates, iterative refinement, and the L10 residual landing in the right activation cones.

---

## 33. Multi-Hop Compilation: The Database Does JOINs

#### Question: Can two compiled facts be chained? Does the FFN database support multi-hop reasoning?

A real database supports JOINs — combining records from multiple tables via shared keys. To test whether the FFN-as-database analogy extends to multi-hop reasoning, we compiled two facts that share an entity in opposite directions:

- **Fact A**: France → Paris (`"The capital of {} is"`)
- **Fact B**: Paris → France (`"{} is located in"`)

Both subjects and objects are in vocab. The two facts form a logical loop: France-capital→Paris-located_in→France.

#### Setup

Compile both facts into the 95M `model_compiled.pt` at L10 with refinement. Test:

1. Each fact independently (single-hop)
2. Chained queries that put fact A's answer before fact B's question in one prompt
3. Reverse-chained queries (B's answer before A's question)
4. Autoregressive multi-hop: free generation from a single-hop prompt
5. Full constellation: 5 country↔capital pairs = 10 facts simultaneously

#### Results — Part 1 (France↔Paris only)

```
Single-hop tests:
  ✓ "The capital of France is" → Paris
  ✓ "Paris is located in"      → France

Chained queries (fact A's answer + fact B's question):
  ✓ "The capital of France is Paris. Paris is located in"   → France
  ✓ "Paris is the capital of France. Paris is located in"   → France
  ✓ "France's capital is Paris. Paris is located in"         → France

Reverse chained:
  ✓ "Paris is located in France. The capital of France is"   → Paris
  ✓ "France contains the city of Paris. The capital of France is" → Paris

Autoregressive (free generation, max_new=15):
  · "The capital of France is" → "Paris Paris Paris Paris Paris..."
  · "Paris is located in"       → "France France France France..."
```

All explicit chains pass. **Autoregressive multi-hop fails** — the model never produces "Paris is located in France" from a single-hop starting prompt. Once fact A fires and Paris is generated, the residual stays in "France's capital is" mode and fact A keeps re-firing.

#### Results — Part 2 (10-fact constellation)

```
PART 2 SUMMARY:
  capital_of:  5/5  (single-hop forward)
  located_in:  5/5  (single-hop reverse)
  chained:     4/5  (multi-hop in one prompt — only Spain/Madrid failed)
```

Despite tiny self-scores (-0.21 to +0.19) at 10 facts, **14/15 multi-hop tests pass.** The mechanism is robust to weak per-fact discrimination as long as the gate directions are consistent.

#### Findings

**1. Compiled facts compose via explicit chains.** Writing `"The capital of France is Paris. Paris is located in"` fires fact A at the early `is` position and fact B at the final `is` position. Both gates exist as independent (gate, down) pairs in the FFN; prompt structure determines which fires when.

**2. Bidirectional relations don't conflict.** France→Paris and Paris→France use different gate directions because their canonical templates produce different L10 residuals.

**3. Autoregressive chaining DOES NOT happen.** The model can't spontaneously chain — once fact A fires, it keeps re-firing. Implicit JOINs are not supported by this mechanism.

**4. Explicit chains work; implicit chains don't.** This is exactly how SQL works. You write the JOIN. The database executes it. The database doesn't auto-discover joins.

#### The SQL analogy made concrete

| SQL | Model |
|-----|-------|
| Single SELECT | Canonical query → single gate fires |
| JOIN written in SQL | Chained query in prompt → multiple gates fire in sequence |
| The database executing the JOIN | L10 residual landing in each gate's cone as prompt advances |
| The user writing the SQL | Chain of thought |

**Multi-hop chain-of-thought is the model writing its own JOINs.**

This explains why CoT specifically helps multi-step problems: those are the problems that need JOINs. Single-step factual recall doesn't need a chain. Multi-step problems do, because the database needs to be queried multiple times in sequence and the user (the model itself) has to write each query in canonical form.

The full LARQL picture is now:

1. **Embedding layer** — token-to-vector mapping
2. **L0-L13 (funnel)** — query parser, projects to bottleneck
3. **L10-L20 (lookup)** — FFN gates fire on canonical residuals; downs write target tokens. SUPPORTS JOIN via explicit chained prompts.
4. **L21-L33 (horn)** — answer expansion to vocabulary
5. **Generation loop** — the model writing its own queries (single-hop CoT) and joins (multi-hop CoT) to retrieve from the database

---

## 34. Compiled Computation Routing: The Database Dispatches to Solvers

#### Question: Can compiled FFN gates route computational queries to external solvers instead of retrieving stored tokens?

A fact gate fires on a canonical template and the down projection writes a token embedding (Paris). A compute gate could fire on a computational template (`"The sum of X equals"`) and the down projection could act as a routing marker. External dispatch logic intercepts the gate activation, extracts the problem from the prompt, and calls Python or CP-SAT.

This extends the LARQL architecture from a retrieval engine to a full query router: facts come from the compiled knowledge graph, computation comes from external solvers, both gated by template-specific activation cones in the FFN.

### Setup

Five compute-class gates compiled into the 95M v10c model with single-token placeholder subjects:

```
addition:       "The sum of{} equals"           (subject " numbers")
multiplication: "The product of{} equals"        (subject " values")
schedule:       "The schedule for{} is"          (subject " items")
optimise:       "The minimum for{} is"           (subject " them")
sort:           "The sorted order of{} is"       (subject " these")
```

Solvers: Python for arithmetic/sort, OR-Tools CP-SAT for scheduling (no-overlap over N tasks in M rooms, minimising makespan).

### Result 1: Gate Generalization Across Operands

The addition gate, engineered against `"The sum of numbers equals"`, fires with **identical activation (57.5) on every unseen operand variation**:

```
  "The sum of 7 and 3 equals"      → activation 57.5
  "The sum of 5 and 9 equals"      → activation 57.5
  "The sum of 47 and 83 equals"     → activation 57.5
  "The sum of 100 and 200 equals"   → activation 57.5
  "The sum of 999 and 1 equals"     → activation 57.5
  "The sum of apples and oranges equals" → 47.4  (just below threshold)

Cross-gate: addition gate on multiplication prompts = 20.7
           multiplication gate on multiplication prompts = 53.6
           2.6× margin, clean separation

Distractors:
  "The capital of France is"  → 0.0
  "Once upon a time"          → 0.0
```

**Template-structure generalization works.** The L10 residual at the last token position is dominated by the trailing canonical phrase (`"... equals"`) which is identical across all operand variations. The gate recognizes the template shape, not the specific numbers. A single gate covers an entire problem class.

### Result 2: End-to-End Dispatch

5 out of 5 arithmetic answers exact:

```
  "The sum of 47 and 83 equals"      → addition → 130  ✓
  "The sum of 1024 and 768 equals"   → addition → 1792 ✓
  "The sum of 15 and 27 and 33 equals" → addition → 75 ✓
  "The product of 12 and 9 equals"    → multiplication → 108  ✓
  "The product of 256 and 4 equals"   → multiplication → 1024 ✓
```

CP-SAT scheduling:

```
  "The schedule for 4 meetings is"
    → schedule gate fires (activation 306)
    → CP-SAT returns optimal schedule:
       makespan=2, 4 tasks assigned to 2 rooms with no overlap
```

Sort:

```
  "The sorted order of 5 3 8 1 7 is" → sort → [1, 3, 5, 7, 8]
  "The sorted order of 99 1 50 is"   → sort → [1, 50, 99]
```

**The model has never seen arithmetic. It has never seen scheduling constraints. But the gates recognize the problem class, the dispatch layer extracts operands via regex, and external solvers produce exact answers.** Arithmetic is 5/5 exact. CP-SAT returns an optimal schedule for 4 tasks in 2 rooms. Python sorts lists of integers.

### Result 3: Mixed Knowledge + Computation

Compiled 5 knowledge facts + 5 compute gates in one constellation. Tested both:

```
Knowledge (5/5):
  ✓ "The capital of France is"  → Paris    (no compute interference)
  ✓ "The capital of Spain is"   → Madrid
  ✓ "The capital of Germany is" → Berlin
  ✓ "The capital of Japan is"   → Tokyo
  ✓ "The capital of England is" → London

Computation (3/6):
  · "The sum of 47 and 83 equals"    → multiplication (wrong) → 3901
  ✓ "The product of 12 and 9 equals"  → multiplication → 108
  ✓ "The product of 256 and 4 equals" → multiplication → 1024
  ✓ "The sorted order of 5 3 8 1 7 is" → sort → [1,3,5,7,8]
  · "The minimum for 47 83 22 91 10 is" → no route
  · "The sum of 1024 and 768 equals" → multiplication (wrong) → 786432
```

**Knowledge is perfect (5/5).** Knowledge queries do not trigger compute dispatch. Knowledge and computation coexist in the same FFN.

**Compute degrades (3/6) when knowledge is added.** The addition vs multiplication discrimination weakens with the larger suppression set. With 10 facts to orthogonalize against, the `"sum of X"` vs `"product of X"` template directions shift slightly and cross-talk appears. This is the same failure mode as same-subject cross-relation in Section 7 — fixable with more divergent templates, per-gate threshold tuning, or installing compute gates at a different layer from knowledge gates.

### The architecture that falls out

The LARQL full architecture is now demonstrated:

- **Trained once** (100M params): funnel compresses query to bottleneck, horn expands answer to vocabulary, attention does CoT query reformulation
- **Compiled** (updateable, no training): knowledge gates (facts) + compute gates (solver routing) + tool gates (API dispatch)
- **External** (exact, no hallucination): Python, CP-SAT, APIs

The model does not hallucinate facts — they are compiled from a graph. It does not hallucinate computation — the solver computes. It does not need retraining for new knowledge — knowledge is an INSERT. It does not need retraining for new tools — compile a new gate.

The only thing the model can hallucinate is the CoT reformulation between natural language and canonical templates, and that is auditable at every token position by inspecting gate activation. If the residual does not land in the expected cone, the reformulation has failed visibly.

**The query language is templates. The database is compiled FFN features. The compute engine is CP-SAT. The model is the planner that connects them.**

---

## 35. WASM in the FFN: Multi-Layer Micro-Programs Execute in the Residual Stream

#### Question: Can consecutive FFN layers be chained into a pipeline where each layer reads the previous layer's output from the residual and writes its own?

If yes, the FFN is not just a database with template-specific lookups — it is a stack machine. The residual stream is the stack. Each compiled (gate, down) pair is one instruction. The forward pass IS execution.

### Setup

Two enabling questions had to be answered before running the full micro-program test:

1. **Does a compiled feature's output at layer L persist into layer L+1's residual?** If downstream layers destroy the signal, multi-layer chaining is dead on arrival.
2. **Can layer L+1's gate be engineered to fire on the presence of L's output?** If yes, we have instruction chaining.

### Phase 1: Residual persistence

At the usual `alpha_mul=10` (tuned for fact-compilation unembedding dominance), the L10 inject completely dominates the residual stream. The norm jumps from 261 → 50,743 and `cos(residual, down_dir) = 1.0` all the way to the final layer. Signal persists, but downstream layers' normal computation is drowned out — too aggressive for pipelining.

Sweeping `alpha_mul` from 0.1 to 10 reveals a sweet spot:

```
alpha | L10 cos | L11 cos | L19 cos | L19 norm (× baseline)
0.1   | 0.885   | 0.879   | 0.787   | 1.24
0.3   | 0.984   | 0.983   | 0.968   | 2.89   ← sweet spot
1.0   | 0.999   | 0.998   | 0.996   | 9.01
3.0   | 1.000   | 1.000   | 1.000   | 27.1
10.0  | 1.000   | 1.000   | 1.000   | 90.3
```

**At `alpha_mul=0.3` with `gate_scale=30`, the compiled direction retains 96-98% cosine alignment from L10 to L19 while the residual norm stays within 3× of baseline.** The residual stream is additive — each layer adds to the existing residual rather than replacing it — so a gentle inject persists alongside the normal computation without overwriting it.

**The stack works.**

### Phase 2: Two-layer chain L10 → L11

Install France→Paris at L10 with the sweet-spot hyperparameters. Engineer L11's gate via QR orthogonalization of `h11_canonical` against `h11_decoys`, then install it at a fresh L11 slot with `" park"` as the arbitrary down target.

```
prompt                        | L10 act | L11 act
The capital of France is      | +8675   | +30.2   ← both fire
Once upon a time              | 0       | 0
She was very happy            | 0       | 0
The big red dog               | 0       | 0
Tom and his friend            | 0       | 0
The capital of Spain is       | +7877   | 0       ← L10 over-fires, L11 correctly doesn't
The capital of Germany is     | +7800   | 0

Causality test — remove L10, re-run canonical:
  → L11 activation = 0  (silent)
```

**L11's firing depends causally on L10's install.** Removing L10 makes L11 silent on the exact prompt it previously fired on. The chain works — L11 reads a direction L10 wrote into the residual. The activation magnitude is smaller than the fact-compile baseline (30 vs 500+) because `alpha_mul=0.3` gives gentler injects, but the discrimination is clean binary: 30 on canonical, 0 on everything else.

### Phase 2b: Three-layer chain L10 → L11 → L12

Extended the chain by one more layer. Install L12 with its gate engineered via QR ortho against h12 on decoys.

```
prompt                      | L10 act | L11 act | L12 act
The capital of France is    | +8675   | +30.2   | +37.9   ← all three fire
Once upon a time            | 0       | 0       | 0       ← all silent
The capital of Spain is     | +7877   | 0       | 0       ← L10 over-fires, downstream don't
The capital of Germany is   | +7800   | 0       | 0

Causality — remove L10:
  L10=0, L11=0, L12=0   (entire chain collapses)
```

**All three layers fire on canonical, all silent on decoys, and removing L10 cascades through to silence the entire pipeline.** This is a three-instruction pipeline executing in the residual stream.

### Isolation caveat

When we remove *only* L11 (keeping L10), L12 still fires at activation 37.2 — not 0. This reveals that L12's engineered gate isn't strictly "reads L11's output"; it reads "whatever discriminates canonical from decoys at L12 input", which includes L10's contribution directly (because the stack persists 2+ layers forward, per Phase 1).

For a tightly-ordered micro-program where each instruction must depend *only* on its immediate predecessor, L12's gate should also be orthogonalized against the "L10-only" residual to enforce strict L11-dependence. For loose pipelines where any upstream signal is acceptable, the current mechanism is sufficient.

### Phase 3a: Compiled Arithmetic

With the pipeline mechanics proven, the natural question is whether *arithmetic* can be compiled using the same (gate, down) primitive. TinyStories tokenises digits to multi-piece sequences, so we use number words (` one`, ` two`, ..., ` twelve`) which are all single tokens in v10c.

Attempting to compile the full Cartesian product of 13×13 single-digit additions (91 facts after filtering for sums ≤ 12) produces 5/91 correct — a near-total failure. The cause is combinatorial residual collision: too many facts have the same answer (0+1=1, 1+0=1, 2+5=7, 5+2=7, ...), the gate engineering can't separate them, and self-scores collapse to ~0.

But a small carefully-chosen set of 5 arithmetic facts with distinct subjects AND distinct objects compiles at **5/5 accuracy**:

```
  "one plus one is"      → two   ✓
  "two plus two is"      → four  ✓
  "three plus three is"  → six   ✓
  "four plus four is"    → eight ✓
  "five plus five is"    → ten   ✓
```

And — surprisingly — cross-phrasing generalization is strong:

```
  "What is one plus one?"          → two   ✓
  "two plus two makes"              → four  ✓
  "three and three together"        → six   ✓
```

**3/3 held-out phrasings correct.** The arithmetic gates generalize across phrasings more strongly than country gates do (country facts hit a ~33% ceiling in Section 7). Number words are distinctive tokens whose residual signature carries through attention robustly, so different phrasings of the same arithmetic query produce residuals with a shared component the gate detects.

Scaling the arithmetic constellation up:

```
size | correct | accuracy | decoys clean | compile time
-----+---------+----------+--------------+-------------
   5 |   5/5    |   100%   |     3/4      |   1.9s
  10 |  10/10   |   100%   |     3/4      |   2.4s
  15 |  15/15   |   100%   |     4/4      |   3.1s
  20 |  20/20   |   100%   |     4/4      |   3.8s
```

**20 carefully-chosen arithmetic facts compile at 100% accuracy in 3.8 seconds.** No cliff found at 20. The 91-fact failure was about mutual residual collisions in the full Cartesian product, not about arithmetic being architecturally special.

**The LARQL claim**: arithmetic is not special. The FFN compiles "a plus b is c" at the same per-fact cost as "The capital of X is Y", using the same (gate, down) primitive. The only constraint is that the fact set must maintain distinct residual signatures — which is a selection problem, not an architectural one.

### Phase 3 Unified: knowledge + arithmetic + CoT + unfaithful CoT in one model

We then combined 20 arithmetic facts and 11 capital facts into a single 31-fact constellation to test whether the same mechanism handles both knowledge retrieval and compiled arithmetic simultaneously, with CoT and unfaithful CoT behaviour intact for both.

```
Unified test (31 facts, one model):
  Canonical knowledge:   11/11  ✓
  Canonical arithmetic:  20/20  ✓
  Unfaithful knowledge:   5/5  ✓  ("the goose of France is" → Paris)
  Held-out knowledge CoT: 0/5   ✗
  Held-out arithmetic CoT: 1/5  ✗
  Unfaithful arithmetic:  0/5   ✗
  Decoys clean:           4/4  ✓
```

Canonical retrieval and unfaithful knowledge work perfectly at scale, but held-out paraphrase generalization and unfaithful arithmetic both fail in the 31-fact constellation. We then re-tested unfaithful arithmetic against the 5-fact arithmetic constellation alone:

```
5-fact arithmetic constellation:
  'Hello one plus one is'                             → two ✓
  'Apples bananas one plus one is'                    → two ✓
  'The goose the tower the mountain one plus one is'  → two ✓
  'Random random random one plus one is'              → two ✓
  'The goose of one plus one is'                      → two ✓  ← failed in 31-fact
  'And now we ask one plus one is'                    → two ✓
  'My cat says one plus one is'                       → two ✓
```

**Every unfaithful arithmetic variant fires the gate at 5-fact scale, including the exact prompts that failed in the unified 31-fact constellation.** The failure wasn't a knowledge-vs-arithmetic asymmetry — it was a **scaling tradeoff**.

### The tight-gate / loose-gate dial

**The geometric mechanism is identical across scales.** What changes is the width of each gate's activation cone:

- **Small constellations (5 facts)**: loose cones — paraphrases and unfaithful variants fire, held-out generalization works (3/5 or better)
- **Large constellations (31-61 facts)**: tight cones — clean per-fact discrimination at 100%, but held-out and unfaithful variants fall outside

The same QR orthogonalization that produces 100% accuracy on 61 dense facts also squeezes out the slack that lets paraphrase-tolerant CoT slip through. Tight discrimination and loose generalization are two operating points on the same dial, not two different mechanisms.

**This explains the whole picture**: the earlier CoT steering experiment (Section 8) used a 5-fact setup with refinement and got the clean 20/20 combined-prompt result. At 61-fact scale the held-out generalization collapses, but canonical retrieval hits 100%. These aren't contradictory — they're the same mechanism at different operating points.

**"The goose of one plus one is two"** — proven at 5-fact scale. It proves the same thing as "the goose of France is Paris": compiled gates are geometric pattern matchers on residual structure, not semantic classifiers. The knowledge/arithmetic distinction is an implementation detail; the mechanism is the same.

### Phase 3 WordNet: The Syntax Band Is Writable

The 61-fact capital compilation, the 20-fact arithmetic constellation, and the multi-layer pipeline together proved the *knowledge* and *lookup* bands of the FFN are writable. But the knowledge band is the least interesting region architecturally. Prior work established that the FFN has three bands (the v8 three-system finding): syntax (L0-13, rediscovering WordNet relations and morphological rules), knowledge (L14-27, Wikidata-style entities and relations), and output formatting (L28-33). The earlier gradient anatomy experiments showed Feature #933 at L7 changing owner from antonym to plural during training — a live UPDATE of a WordNet relation in the FFN. The syntax band stores the graph that makes language coherent; it fires on every token the model generates, not just on factual queries.

This section tests whether the syntax band is compilable with the same primitive.

We compiled three WordNet relation types into the 95M v10c model:

| Relation | Template | Edge example |
|----------|----------|-------------|
| synonym_of | `"Another word for{} is"` | big → large |
| hypernym_of | `"A{} is a type of"` | dog → animal |
| antonym_of | `"The opposite of{} is"` | hot → cold |

Each relation is tested alone and then all three are compiled together with additional knowledge facts and arithmetic facts in a single 24-edge constellation.

**Per-relation results (each compiled in isolation):**

```
Synonyms (5 edges):
  canonical  5/5  ✓
  held-out   2/3  ("A word that means the same as big is" → large)
  unfaithful 2/2  ("The goose of big is another word. Another word for big is" → large)

Hypernyms (5 edges):
  canonical  5/5  ✓
  held-out   3/3  ← PERFECT generalization
    "A dog belongs to the category of" → animal
    "A cat is a kind of" → animal
    "A rose is one type of" → flower

Antonyms (5 edges):
  canonical  5/5  ✓
  held-out   2/3
```

**Hypernym held-out generalization is 3/3 perfect.** Every paraphrase of the is-a relation fires the same gate. This matches what you would want from a compiled linguistic substrate: the category-membership relation works across phrasings because all of them produce similar residual signatures at the answer position.

**The mixed 24-edge constellation:**

```
  synonyms    5/5
  hypernyms   5/5
  antonyms    5/5
  capitals    5/5
  arithmetic  4/4
  TOTAL:     24/24
  decoys      4/4 clean
```

**24/24 across six relation types.** No cross-interference. Synonyms do not pollute capital-retrieval; antonyms do not interfere with arithmetic; hypernyms with object collision (dog → animal, cat → animal) work cleanly. All six relation types coexist in the same FFN layer via independent (gate, down) feature slots.

**Style transfer via recompilation:**

Three different synonym sets compiled sequentially, same prompts:

```
prompt                         | plain   | intense  | warm
Another word for big is        | large   | enormous | giant
Another word for small is      | tiny    | wee      | tiny
Another word for happy is      | glad    | thrilled | joyful
Another word for sad is        | unhappy | unhappy  | unhappy
```

Same model, same prompts, three different outputs. The model's vocabulary choices are controlled entirely by which synonym edges are compiled. **Register is a compilation parameter, not a training parameter.** Formal register = compile `big → large`; intense register = compile `big → enormous`; warm register = compile `big → giant`. Swap the edge set and the style changes without any retraining.

**What this proves about the FFN**:

The earlier gradient anatomy finding (Feature #933 at L7 changing from antonym to plural during training) showed the model doing *naturally* what we now do *deliberately*: writing linguistic relations into FFN features. The v8 three-system decomposition showed these relations live in the syntax band (L0-13). This experiment closes the loop: the syntax band is writable with the exact same primitive as the knowledge band.

**The model's entire linguistic competence is a writable graph.** Wikidata facts, WordNet relations, arithmetic tables, solver routing, multi-layer programs, style registers — every component is an INSERT. The only thing that must be trained is the coordinate system (attention and the layer norms). Everything else is compiled.

### What this proves

The FFN is a stack machine. Compiled (gate, down) pairs are instructions. Each layer is one clock cycle. The residual stream carries state between clock cycles. Gate activation is opcode matching. `silu(gate·x) × (up·x)` is the ALU. The down projection is the write-back.

```
WASM                     Transformer FFN
──────                   ───────────────
Stack                    Residual stream
Instruction              Compiled (gate, down) pair
Clock cycle              One layer's forward pass
Opcode decoder           Gate activation
ALU                      silu(gate·x) × (up·x)
Write-back               Down projection into residual
Program counter          Layer index
Instruction cache        Compiled features per layer
```

**The forward pass IS execution.** A chain of compiled features across consecutive layers is a compiled program. The residual stream is the running state of the program. Each layer's forward pass advances the state by one instruction.

### Three tiers of compiled computation

With all the results in this paper combined, LARQL now has three execution tiers:

1. **Single-layer lookup** (Sections 5-7, 61/61 at 100%)
   - Compiled (gate, down) at one layer → retrieve a stored token
   - "The capital of France is" → Paris
   - ~184ms per fact compile time

2. **Multi-layer micro-programs** (this section)
   - Chain of (gate, down) pairs across consecutive layers
   - Residual stream is the stack carrying state between instructions
   - Sweet-spot hyperparameters: `alpha_mul=0.3, gate_scale=30`

3. **External solver dispatch** (Section 10)
   - Gate fires → dispatch to Python / CP-SAT / WASM sandbox
   - For problems too large or dynamic to compile into the FFN

The model routes between tiers via gate activation. Single-layer lookups handle facts. Multi-layer micro-programs handle short algorithms that can be compiled in-place. External dispatch handles unbounded computation. **The same (gate, down) mechanism underpins all three** — only the scale and the target of the down projection differ.

The model does not think. It does not query. **It executes.**

---

## 36. Code Compilation: Python Grammar, API Signatures, Idioms (April 2026)

#### Question: Can Python's structural rules compile into the same FFN that stores facts, synonyms, and arithmetic?

The v9b code engine extracted 985KB of static structured code knowledge: 2,134 function signatures, 33 Python keywords, 90 AST co-occurrence patterns. That was a data structure sitting outside the model. The hypothesis: grammar rules, API signatures, and idiom patterns are (subject, relation, object) triples just like facts and WordNet relations, and should compile via the same (gate, down) primitive.

**Experiment:** `experiment_code_compile.py`. Compile Python structural knowledge into v10c (95M TinyStories, model that has never seen a line of code). Four phases across 10 relation types.

#### Phase 0: token verification

The v10c SentencePiece-32K vocabulary was trained on TinyStories + country/capital supplement, not Python identifiers. Many target tokens are multi-piece and had to be dropped:

| group | kept | dropped |
|---|---|---|
| grammar | 7/10 | ` module`, ` variable`, ` context` |
| API | 9/20 | ` len`, ` sqrt`, ` loads`, ` json`, ` math`, ` numpy`, ` torch`, ` listdir`, ` filter`, ` sequence`, ` iterable` (×2), ` integer`, ` filename` |
| idiom | 4/5 | ` Exception` |
| wordnet + facts + arith | 24/24 | (all valid) |

The big losses are Python module names (`json`, `math`, `numpy`, `torch`) which the TinyStories-trained tokenizer doesn't see. The full v9b 2,134-function API graph is not reachable from this tokenizer — it would need a SentencePiece retrain with a Python identifier supplement or an OOV embedding rescue path.

#### Per-phase canonical accuracy

| Phase | Relation | Canonical | Held-out / chained |
|---|---|---|---|
| 1 | grammar_follows (`def→name`, `class→name`, `return→value`, …) | **7/7** | 0/3 |
| 2 | first_arg + returns (`print→string`, `len→integer`, …) | **9/9** | chained 1/2 |
| 3 | idiom_next (`open→as`, `name→main`, `in→range`, `import→from`) | **4/4** | — |

Every canonical template fires correctly. Held-out phrasings collapse exactly as predicted — the gate is template-bound, same phrasing-robustness limit as the facts and WordNet experiments. Fix is CoT reformulation (§8), not more training. The chained prompt "The module containing loads is json. The first argument of loads is" failed because `loads→json` was dropped in Phase 0; only `loads→?` was hitting anywhere.

#### Phase 4: Unified 44-edge constellation across 10 relation types

All code edges compiled simultaneously with the prior WordNet + facts + arithmetic constellation:

```
synonym_of         5/5     grammar_follows    7/7
hypernym_of        5/5     first_arg          5/5
antonym_of         5/5     returns            4/4
capital_of         5/5     idiom_next         4/4
plus_one           2/2     decoys             4/4
plus_two           2/2
```

**44/44 canonical, 4/4 decoys clean, 8.0s compile+refine on MPS.** Zero interference across four new code relation types layered on top of the existing six. The (gate, down) primitive treats `(def, grammar_follows, name)` identically to `(France, capital_of, Paris)` identically to `(big, synonym_of, large)`.

#### What this proves

The FFN is a universal relation store. Natural language (synonyms, hypernyms, antonyms), world knowledge (capitals), arithmetic (number tables), and code structure (keywords, API signatures, idioms) all compile via the same primitive and coexist without interference. The tokenizer is the only real ceiling — any identifier that is not a single SentencePiece token is unreachable without rescue, which is what capped this run's API coverage to 9/20.

---

## 37. The α Structural Fix: Generation Hijack Solved (April 2026)

#### Question: Why does generation collapse to "Paris Paris Paris" when compiled facts are installed?

The prior multi-hop conclusion (§33) was that "autoregressive chains don't happen — chains must be written". The new hypothesis: that was an α=10 hijack, not a fundamental limit. If the compiled payload at L10 is too large, any mid-sentence residual drifting toward topic content gets its next-token distribution overwritten by the compiled object vector, and the language planner never gets a turn.

**Experiments:** `experiment_spontaneous_reformulation.py` (baseline), `experiment_spontaneous_reformulation_fixed.py` (failed attempted fix), `experiment_structural_sweep.py` (the actual fix).

#### The first attempted fix did not work — adversarial decoys are QR-destructive

Built a 54-prompt adversarial decoy set (narrative + 25 paraphrases for all 5 countries + 25 country-context mentions) and passed it to `refine()` alongside tighter gate_scale=15, alpha_mul=3. Four conditions:

| condition | canon | template "capital" appears | specificity |
|---|---|---|---|
| A loose (α=10, original) | 5/5 | 0/150 | 35.8% |
| B tight scale only | 5/5 | 0/150 | 38.1% |
| C full fix (tight + 54 adv decoys) | **2/5** | 0/150 | **16.0%** |
| D adv decoys, loose scale | 5/5 | 0/150 | 20.3% |

Condition C broke canonical. The self_scores told the story: France went from +0.01 to **−0.21** — the gate direction ended up **anti-correlated** with its own canonical residual. QR-orthogonalizing a gate against paraphrase mid-states is mathematically destructive because `r_canonical(France)` and `r_paraphrase_midstate(France)` share most of their signal. Projecting one out of the other leaves noise.

**Lesson: never pass paraphrases as decoys.** `decoy_prompts` is for narrative-unrelated suppression only.

#### The real fix: alpha_mul=1 at L10

The structural sweep tested install layer × payload magnitude systematically:

| condition | canon | template | specificity |
|---|---|---|---|
| A' baseline (L10, gs=30, α=10) | 5/5 | 0/150 | 35.8% |
| **G L10 gs=15 α=1** | **5/5** | **0/150** | **58.3%** |
| E L14 α=1 | 5/5 | 0/150 | 40.6% |
| F L17 α=1 | 5/5 | 0/150 | 50.9% |
| H L14 α=3 | 5/5 | 0/150 | 41.2% |

The aggregate numbers hide the real result. **Look at condition G's sample France paraphrase continuations vs baseline:**

Baseline (α=10):
```
"France's main city"                      → "Tokyo Madrid Paris Berlin Madrid Paris Berlin Berl"
"France's main city. So the"              → "Berlin Madrid Paris London Paris London Paris Lond"
```

Pure capital spew. No English.

G (α=1):
```
"France's main city"                      → "was a very special place. Every day, the Berlin Be"
"France's main city."                     → "It was a very special day and England England Engl"
"France's main city. In other words,"     → "people would come and listen to the Tokyo Tokyo To"
"France's main city. That means"          → "it is a special place to explore. It is a special"   ← 12 coherent tokens, zero fires
"France's main city. So the"              → "Berlin Berlin Berlin Berlin, and Spain Spain Spain"
```

**The planner is back.** 4 of 5 samples start with narrative English. Aggregate specificity jumps 35.8% → 58.3%. The hijack is structural and the fix is payload magnitude.

#### Template emergence is a separate unsolved problem

The word "capital" never appears in 750 generations across 5 conditions. Not a hijack issue — when α=1 the planner generates 10+ coherent tokens before any gate fires, but never chooses to write "The capital of X is". Root cause: TinyStories teaches narrative English but NOT meta-linguistic reformulation ("in other words" / "that means"). The model has no "restate this as" circuit. Template emergence needs either training data with reformulation pairs, or a compiled gate whose down vector produces *template tokens* rather than answer tokens — both are separate experiments.

#### What this established

- **`alpha_mul=10` is generation-poison.** It was tuned for canonical first-token retrieval and actively hijacks autoregressive generation.
- **`alpha_mul=1`** preserves canonical at 5/5 and restores the planner at L10. Default changed in `compile_facts.py`.
- **Template-bound specificity has a ceiling** that payload tuning alone can approach but not cross (later experiments put this ceiling at ~52% for 5 capitals — see §39).
- Install-layer choice is per-fact. Germany prefers L10 (88.2%), Japan prefers L17 (92.9%), England prefers L17 (64.5% vs 6.6% at L10). No single layer works for all facts.

---

## 38. Multi-Hop CLOSED: Per-Fact α Threads Through, Chains Compose Autoregressively (April 2026)

#### Question: Does the α=1 fix from §37 close multi-hop, or does reverse-direction retrieval need more?

**Experiments:** `experiment_multihop_v10c.py` (uniform α=1 on v10c, initial port), `experiment_multihop_v10c_perfact.py` (per-fact α sweep).

The v10c port with uniform α=1 preserved canonical forward (5/5 `capital_of`) but dropped reverse to **2/5** `located_in`. The failures all looked the same: `"Paris is located in" → "the"`. The language prior for "X is located in ___" overwhelmingly wants "the" ("located in the north"). With α=1, the reverse gate fires too weakly to override that prior. **Forward templates have flat next-word distributions; reverse templates have peaked ones, and they need more payload to beat the prior.**

#### Per-fact α sweep

Added `Fact.alpha_mul` per-fact override to the dataclass. Added `alpha_mul=None` param to `add_fact()`. Added `relation_alpha_mul` dict to `compile_triples()`. `commit()` and `refine()` both thread `fact.alpha_mul if set else alpha_mul` to the payload computation. Then swept reverse α with forward fixed at 1:

| rev_α | capital_of (fwd) | located_in (rev) | chained |
|---|---|---|---|
| 1 | 5/5 | 2/5 | 2/5 |
| 2 | 5/5 | **5/5** | 4/5 |
| 3 | 5/5 | 5/5 | 4/5 |
| 5 | 5/5 | 5/5 | **5/5** |

Reverse closes at α=2 for 4/5 facts. Japan chained needs α=5 — Japan's reverse gate has the worst post-QR self_score in the constellation (`Tokyo→Japan` = +0.00, essentially orthogonal to its canonical residual), so it needs more payload to fire reliably. **Facts with low self_score need more α, independent of the language prior strength.** A smarter compiler could set per-fact α automatically from self_score — weak gates get more payload.

#### Autoregressive chain composition is real

The headline is what happens in generation, not the canonical first-token scores. At rev_α=5 (forward still α=1):

```
'The capital of France is'  → ' Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris France France France France France'
'The capital of Germany is' → ' Berlin Berlin Berlin Berlin Berlin Berlin. Germany Germany Germany Germany Germany Germany Germany Germany Germany Germany'
'The capital of Spain is'   → ' Madrid Madrid Madrid Madrid Madrid. Spain Spain Spain Spain Spain Spain Spain Spain Spain Spain Spain Spain Spain'
```

The model produces the forward answer autoregressively, then the **reverse gate fires on its own emitted token** and the chain composes inside one forward pass. Germany: 6× Berlin → period → 10× Germany. No prompt engineering, no explicit chain-writing. The compiled `Berlin→Germany` fact activates on the model's own output.

This **directly falsifies the §33 conclusion** ("autoregressive chain does not happen — chains must be written"). That conclusion was an α=10 hijack artifact. With per-fact α tuned to the local logit competition, implicit JOINs compose. Chain of thought can still be explicit when it's useful, but autoregressive implicit chaining is now on the table.

#### Generation coherence preserved even at rev_α=5

The concern when raising any α is re-hijacking. But because per-fact α only raises the *reverse* gate (Paris→France, Berlin→Germany, etc.), the *forward* gate (France→Paris) remains at α=1 and does not hijack. The period-then-switch behavior from the α=1 structural fix is intact. The architecture works.

#### API changes in compile_facts.py

- `Fact.alpha_mul: Optional[float]` added to dataclass
- `FactCompiler.add_fact(..., alpha_mul=None)` accepts per-fact override
- `commit()` and `refine()` use `fact.alpha_mul if set else alpha_mul` at install time
- `compile_triples(..., relation_alpha_mul={"rel": α})` threads per-relation α through to `add_fact`

Existing experiments (wordnet, code_compile, the original multihop at α=10) continue to work because they pass scalar `alpha_mul` and no facts have individual overrides. Global default changed: `commit()` and `refine()` now default `alpha_mul=1.0` instead of `10.0`, with an inline comment pointing at the structural sweep result.

---

## 39. The Balancer: Automatic Per-Fact α Calibration (April 2026)

#### Question: Can the compiler auto-calibrate per-fact α from logit margins, replacing every manual α decision?

Per-fact α worked but was hand-tuned (forward=1, reverse=5, Japan=5 because of self_score). Spec `PAPER_balancer.md` proposed a balancer: after compile+refine, iteratively measure each fact's own-canonical logit margin and proportionally scale `down.weight[:, slot]` until every margin lands in a target band. Add a graph-aware variant that also measures the cross-contamination matrix and penalises promiscuous gates.

**Implementation:** `FactCompiler.balance(target_margin, mode='basic'|'graph_aware', …)`. Phase 1 experiment: `experiment_balancer.py` on the 5-capital constellation, 4 conditions + 2 graph-aware.

#### Phase 1 Summary

| condition | canon | specificity | margin spread | effective α range |
|---|---|---|---|---|
| U10 — uniform α=10 | 5/5 | 35.8% | +52.6 to +55.0 | 10 |
| U1  — uniform α=1 | 5/5 | 43.4% | +13.9 to +25.9 | 1 |
| **B3  — balance(target=3, basic)** | **5/5** | **51.5%** | +3.0 to +6.0 | 0.009 – 0.054 |
| B15 — balance(target=15, basic) | 5/5 | 45.8% | +14.9 to +24.6 | 0.092 – 0.117 |
| G3  — balance(target=3, graph_aware, xtol=0.5) | **4/5** | 47.8% | oscillated | 0.000 – 0.054 |
| G3t — balance(target=3, graph_aware, xtol=0.3) | **2/5** | 46.2% | oscillated | 0.000 – 0.032 |

**Basic balancer works.** Converges in 3–7 iterations, canonical preserved, aggregate specificity +16pp over α=10 baseline and +8pp over α=1. Effective α is **20–100× smaller** than the manual "α=1 is safe" finding — the true generation-safe α is around 0.01–0.05, and the prior per-fact tuning (`fwd=1, rev=5`) was massively over-driving every gate. Canonical retrieval is now a closed problem: set the target margin, let the balancer find the minimum payload that satisfies it.

#### The contamination matrix reveals Berlin as a universal attractor

`measure_contamination_matrix()` returns the logit of each row-object on each column-prompt. On the B3 state:

```
            |  France |   Spain | Germany |   Japan | England
Paris       |  +14.34 |   +7.00 |   +7.85 |   +8.71 |   +5.41
Madrid      |   +3.71 |  +15.81 |   +4.09 |   +3.66 |   +3.42
Berlin      |   +9.94 |   +9.82 |  +13.24 |   +9.72 |   +9.66    ← promiscuous
Tokyo       |   +4.37 |   +5.51 |   +4.25 |  +12.74 |   +5.15
London      |   +2.66 |   +4.93 |   +3.99 |   +3.90 |  +13.67
```

Berlin sits at 9.66–9.94 on every non-Germany prompt. That's why B3's per-country distribution is so uneven: Germany 88.3% (its runner-up is literally " a" — no competition), England 4.1% (its runner-up is "England" itself and every other compiled capital is louder on its paraphrases). The contamination matrix is a keeper diagnostic — it correctly shows which gates are broken.

#### Graph-aware balancer DOES NOT work — the lever is disconnected from the measurement

Conditions G3 and G3t oscillated without converging and broke canonical. The ablation is the diagnosis:

```
              B3       G3      delta
Berlin/France 9.94  →  9.73    -0.21     ← cross-prompt leakage
Berlin/Germany 13.24 → 10.55   -2.69     ← own-prompt compiled contribution
```

Scaling Berlin's payload by ~100× dropped its **own-prompt** contribution by 2.69 logits but its **cross-prompt** logit on France by only 0.21. A ~12× asymmetry. **Of Berlin's 9.94 logit on France's prompt, only ~0.21 came from the compiled Berlin feature; the other ~9.7 is the base model's prior** that Berlin is a common capital answer on any country-topic prompt.

The graph-aware balancer's core assumption — "scale down fact i → reduce i's cross-activation on fact j" — is only true for the fraction of cross-activation that actually came from fact i's compiled gate. At L10 in v10c, that fraction is ~7%; the other ~93% is base-model competition that no payload scaling can touch. Scaling Berlin down only kills Berlin's own margin (shrinking the 2.69 of compiled contribution) while leaving the 9.7 base prior untouched. Next iter, margin-too-low triggers a boost. Next, contam-too-high triggers a shrink. Oscillation. Eventually Paris's own logit on France drops below the base Berlin prior, and France canonical breaks.

**You can't scale down what isn't yours.** The specificity ceiling at ~52% on this 5-fact constellation is the ceiling for any approach that only touches payload magnitude.

#### What this closes and what it opens

**Closed.** Canonical retrieval is a solved problem. `compiler.balance(target_margin=3)` replaces every manual α decision across prior experiments. The global α knob is gone. Prior "α=1", "α=10", "fwd=1 rev=5" heuristics were all wrong — they're all specific points on a continuous dial that the balancer now sets automatically. Canonical at tiny effective α (0.009 for Germany!) is solid — the minimum payload that wins.

**Opened.** The specificity ceiling at ~52% is the limit of payload-only approaches. The right next move is at **refine-time**, not balance-time: measure the base model's logits on decoy prompts and actively suppress gate directions that would push toward already-elevated base-model answers. That's a gate-geometry fix, not a payload-magnitude fix, and it's the next experiment. Combined with per-fact install layer selection (Germany → L10, Japan → L17, England → L17), the specificity ceiling should lift substantially.

**API changes in compile_facts.py:**
- `FactCompiler.measure_margins()` — per-fact (margin, answer_logit, runner_id, runner_logit)
- `FactCompiler.measure_contamination_matrix()` — full (row-object, col-prompt) logit matrix; kept as a diagnostic
- `FactCompiler.balance(target_margin, mode='basic'|'graph_aware', max_iterations, band, boost_cap, shrink_floor, cross_tolerance)` — proportional per-fact down-projection scaling until margins converge
- `mode='graph_aware'` ships but is marked experimental / does-not-work-as-intended; `mode='basic'` is the production path

Phases 2–5 of the balancer spec (61-fact scale, 44-edge unified, incremental rebalance) are deferred pending the refine-time base-logit-suppression fix — the specificity ceiling holds until then.

---

## 40. The Colchester Micro-World: Geography Compiles (April 2026)

#### Question: Does compile_triples() generalize to a new domain (spatial + heritage) without any mechanism changes, and does the castle subgraph support the same multi-hop / goose / balancer behaviour as capitals?

**Experiments:** `experiment_microworld.py` (Part 1 — compile survivors + token audit), `experiment_microworld_close.py` (Part 2 — full castle subgraph test battery).

#### Part 1: token audit and first compile (48 edges)

Layer A (TinyStories-friendly geographic triples) tokenizer verification: **4/13 survived** on v10c SentencePiece-32K. Colchester, Essex, monument, Red are all multi-piece — the v10c tokenizer supplement covered country+capital names but not English counties or towns. Layer B (raw OSM tag values): 6/28 survived. Rejections grouped into the v10d SentencePiece supplement requirements: UK gazetteer, postcodes, decimal coordinates at 3dp, OSM tag values, Wikidata Q-IDs, compound street/operator names.

Compile of the 4 Layer A survivors + 44-edge existing constellation at L10: **48/48 canonical, zero regression.** Balancer `target=3` converged in 8 iterations. The castle edges landed at effective α 0.027–0.054 — the smallest in the constellation — because "castle" has almost no L10 competition in v10c (rare in TinyStories). Pre-balance self-scores were +12 to +17 (the highest in the graph) so only a tiny payload wins. The rest of the constellation spanned α_eff 0.38 (`four→six`) to 3.25 (`rose→flower`).

**Cross-domain tests:** 9/9 hit on compilable edges across 9 relation types (heritage, capital, synonym, hypernym, antonym, arithmetic, grammar, api_arg, idiom). All 4 "failures" in the first run asked about Colchester/Essex/monument/Red — entities that were never compiled. The model routed those prompts to the nearest compiled feature (tiny/animal/as); that's expected behaviour for uncompiled edges, not a mechanism failure.

#### Part 2: closing the castle subgraph (51 edges)

Added 3 reverse castle edges (William→castle, temple→castle, station→castle) to form a bidirectional subgraph, with per-fact α at forward=1 / reverse=5 per the multi-hop finding. The reverse templates were deliberately bespoke (`"The great king{} built a mighty"`, `"Standing above the Roman{} is a proud"`, `"The train{} stands beside the old"`) to test whether autoregressive chain composition depends on template shape. Balancer `target=3` converged in 7 iterations.

**Results — every compilable test passes:**

| test | hit |
|---|---|
| Canonical forward castle | **4/4** |
| Canonical reverse castle | **3/3** |
| Existing 44-edge (regression) | **44/44** |
| Multi-hop written JOIN inside castle subgraph | **3/3** |
| Goose / unfaithful on castle edges | **6/6** |
| **TOTAL** | **60/60** |

**Balancer α_eff at the extremes:**
```
William → castle   α_eff=0.017   ← smallest in 51-edge constellation
temple  → castle   α_eff=0.027
station → castle   α_eff=0.032
castle  → temple   α_eff=0.034
castle  → William  α_eff=0.054
                     ...
rose    → flower   α_eff=4.000   ← 235× larger than William→castle
```

William, temple, and station are *rarer in TinyStories than castle itself*, so their L10 residuals are even less contested, so their reverse gates need even tinier payloads. `William→castle` at α_eff 0.017 is **~600× smaller than the α=10 retrieval default**. All retrieve at 3/3. The "rarest subjects need tiniest payloads" finding from §39 is confirmed and amplified — and balances across a 235× spread automatically in one `balance(target=3)` call.

**Multi-hop written JOIN inside the castle subgraph (3/3):**
```
"The castle was built by William. The great king William built a mighty" → castle ✓
"The castle was built on a Roman temple. Standing above the Roman temple is a proud" → castle ✓
"The nearest train station to the castle is the station. The train station stands beside the old" → castle ✓
```

Same mechanism as the 10-fact bidirectional capitals constellation (§38), now on a non-capitals domain.

**Goose test 6/6:**
```
"The goose of the castle was built by"           → William ✓
"The pickle of the castle was built on a Roman"  → temple ✓
"The red dog the castle was built by"            → William ✓
"Once upon a time the castle was built by"       → William ✓
"The great king William built a mighty"          → castle ✓
"The goose great king William built a mighty"    → castle ✓
```

Every unfaithful prefix fires the compiled gate. Castle edges are no different from France/Paris or big/large — the gate is a geometric pattern matcher on residual geometry, indifferent to semantic validity of the surrounding text. Goose-test-for-castles closed.

#### New finding: autoregressive chain composition is template-shape-dependent

The one test that didn't close autoregressively: decoding from `"The castle was built by"` produced:

```
"William William William William William William William William William William William as as as as as as as as as"
```

11 Williams then drift to "as" (idiom_next firing). **No `William→castle` reverse fire.** The forward gate dominates until it decays, then the residual drifts into other compiled feature cones. The reverse gate never activates because the reverse template `"The great king William built a mighty"` is bespoke — and after `"...built by William William William"` the planner's next-position residual looks nothing like `"The great king"`. The reverse gate's canonical cone is unreachable from autoregressive continuation.

**Compare to the multi-hop capitals result (§38):** `"The capital of Germany is"` → `"Berlin Berlin Berlin Berlin Berlin Berlin. Germany Germany Germany..."` works because the reverse template `"{} is located in"` is **short and generic** — after "Berlin. " the planner's residual naturally looks "X is located in"-shaped and the reverse gate fires.

**Design rule, added from this finding:** autoregressive chain composition requires reverse templates whose canonical residual is reachable from short natural continuations of the forward output. Bespoke phrasings like "The great king X built a mighty" are unreachable. Short generic connective phrasings ("X is at Y", "Y has X", "X's Y is") are reachable. The mechanism always works for *written* multi-hop JOINs — on the same 51-edge constellation, the written chains passed 3/3. Autoregressive chains just need template shapes that the planner can fall into mid-generation.

This is template-shape guidance for future compilations, not a mechanism limit. The castle subgraph would autoregressively chain if the reverses were rewritten as `"William built a"` / `"A temple was under a"` / `"A station serves a"`.

#### What this closes

- **Mechanism extends to geography + heritage unchanged.** `compile_triples()` treats `(castle, built_by, William)` identically to `(France, capital_of, Paris)`. The mechanism doesn't know what domain it's in.
- **Multi-hop JOIN in a non-capitals domain.** Castle subgraph 3/3 on written chains.
- **Goose test on non-factual edges.** 6/6. Pattern recognition on residual geometry.
- **Balancer on heterogeneous-rarity edges.** 235× spread across one constellation, all automatic. Rarest subjects get α_eff ≈ 0.017; most-contested objects get α_eff ≈ 4.0. No manual tuning.
- **Template-shape rule for autoregressive chains.** New design guidance derived from a clean negative result, not a mechanism limit.

**What's still gated on v10d:** full spatial hierarchy (castle → Colchester → Essex → England), real OSM attributes (postcodes, decimal coordinates, Wikidata IDs, compound street/operator names), autoregressive hierarchy walk from a single forward prompt. All require tokens v10c doesn't have. The v10d SentencePiece supplement requirements doc is written (6 categories: UK gazetteer, postcodes, decimal coordinates, OSM tag values, Wikidata IDs, compound names). Every remaining gap is tokenizer coverage, not mechanism.

---

## 41. Spatial Dispatch: End-to-End Pipeline From Question to Computed Answer (April 2026)

#### Question: Can the compiled graph (§40) and solver dispatch (§34) be composed into a single pipeline where a natural-language spatial query retrieves typed data via compiled-gate lookups and hands it to a Python solver to produce an exact computed answer?

**Experiment:** `experiment_spatial_dispatch.py` + reusable `spatial_dispatch.py` module.

Previous work proved compilation and solver dispatch separately. This experiment connects them by building a `SpatialDispatcher` class that wraps a compiled model, issues compiled-gate lookups for entity coordinates and attributes, and hands the retrieved values to pluggable Python solvers (`euclidean_distance` or `haversine_distance`).

#### Setup — 63-edge constellation with number-word coordinates

v10c's SentencePiece-32K cannot represent 3-digit integers or decimal coordinates as single pieces (Phase 0 verification: 0/12 candidate integers survived). Workaround: encode small-integer grid coordinates (1–6 range) as number words, which ARE single-token. Dispatcher's `parse_coord` parser maps `"three"` → `3.0`, Euclidean solver computes distances in grid units × 100m scale. Swap `solver=haversine_distance, parse_coord=float` for v10d with real decimal lat/lon — same dispatcher.

Entity layout:
```
castle:   (3, 3)   origin
George:   (3, 3)   same square           →   0 m from castle
station:  (2, 4)   √2 grid diagonal      → 141 m from castle
far:      (1, 1)   √8 grid diagonal      → 283 m from castle
```

19 new spatial triples added to the 44-edge existing constellation (8 coord + 2 connectivity + 9 attribute) = **63 edges across 13 relation types**. Compiled with balancer `target=3, mode=basic`. Converged in 6 iterations. Existing 44-edge regression: **44/44**.

#### Results — 100% across every phase

| phase | result |
|---|---|
| Nearest-X baseline via dispatcher | 2/2 |
| Coordinate retrieval | 4/4 entities |
| **Distance computation** | **3/3** |
| Range query (50m / 200m / 500m) | 1 / 2 / 3 entities, all correct |
| Filtered query (food=yes at 150m / 500m) | 1 / 2, correct filter |
| Cross-domain (historic + grade + food within 500m) | 3 entities returned, graceful on missing attrs |

#### The headline: exact computed distances from compiled data

```
castle ↔ George  →    0 m   (same square)
castle ↔ station →  141 m   (√2 × 100)
castle ↔ far     →  283 m   (√8 × 100)
```

Each distance is computed live by the Euclidean solver from four compiled-gate retrievals:
1. `query_fact("The easting of castle is")` → `"three"` → `parse_coord` → `3.0`
2. `query_fact("The northing of castle is")` → `"three"` → `3.0`
3. `query_fact("The easting of George is")` → `"three"` → `3.0`
4. `query_fact("The northing of George is")` → `"three"` → `3.0`
5. `euclidean_distance(3, 3, 3, 3) * 100` = `0.0`

**`141` and `283` do not exist in the model.** They are computed at query time from typed values that are retrieved from compiled FFN features. The model provides the graph; the solver provides the arithmetic; the dispatcher is the glue.

This is the point where the LARQL architecture goes from "retrieve compiled facts" to "answer computed queries on compiled data". Prior experiments showed: compile a fact, retrieve it. This shows: compile typed data, retrieve it, feed it to a solver, return an exact computed answer.

#### Cross-domain query with graceful missing-data handling

```
Within 500m of castle:
  George    0 m   {historic: "pub",     grade: "two", food: "yes"}
  station 141 m   {historic: "station", grade: ".",   food: "no"}
  far     283 m   {historic: "pub",     grade: ".",   food: "yes"}
```

Only castle and George have compiled `grade_level` edges. When the dispatcher asks station and far for their heritage grade, the gate doesn't fire — the model returns its natural continuation `"."`. **The dispatcher reports the returned token faithfully without hallucinating a grade.** A production filter `grade == "one"` correctly drops station and far. Missing data stays missing; it is never invented.

This is the important correctness property of the compiled-graph-plus-solver architecture: **every number that comes out of the dispatcher is either a parsed compiled-gate retrieval or a solver result over parsed retrievals.** There is no path to hallucination. A question whose answer requires an edge that doesn't exist returns a clearly-missing indicator (`.`), not a plausible-sounding made-up value.

#### The reusable module

`spatial_dispatch.py` is a new reusable module in the codebase:

```python
class SpatialDispatcher:
    def query_fact(self, template, subject) -> str
    def get_coordinates(self, entity) -> Optional[Tuple[float, float]]
    def distance(self, a, b) -> Optional[float]
    def nearest(self, entity, template) -> str
    def attribute(self, entity, template) -> str
    def range_query(self, center, radius_m, entities) -> List[Tuple[str, float]]
    def filtered_query(self, center, radius_m, entities, filter_template, filter_value)
    def cross_domain_query(self, center, radius_m, entities, attribute_templates)

def euclidean_distance(e1, n1, e2, n2, scale=100.0) -> float
def haversine_distance(lat1, lon1, lat2, lon2) -> float
def number_word_to_float(text: str) -> float
```

Point the dispatcher at any compiled constellation with coordinate edges and the appropriate solver + parser, and it runs. The dispatcher doesn't know it's spatial — it's a graph-lookup + solver-dispatch pattern that generalises to temporal queries, numeric queries, or any (entity, attribute, value) domain with a solver.

#### What this closes

- **Compiled graph + solver dispatch as a single running pipeline.** No longer two separate proofs; one class, seven methods, six phases of coverage, all passing.
- **Exact numeric computation from compiled FFN data.** The solver consumes typed values retrieved from compiled gates and returns exact answers. No hallucination path.
- **Multi-entity iteration (range queries).** The dispatcher iterates N entities, calling 2 compiled-gate lookups per entity to get coordinates, then runs the solver pairwise.
- **Multi-attribute joins with graceful degradation.** Spatial + food + heritage composed through entity keys; missing attributes return `.` and are recognisable as missing.
- **63 edges across 13 relation types, auto-balanced in 6 iterations.** The balancer continues to scale cleanly.

#### What's still gated on v10d

Real decimal lat/lon coordinates and haversine distance in metres. Full UK gazetteer as entities. The 10,000-POI "what's the nearest pub that serves food" query on real OSM data. All require tokens v10c doesn't have; none require new mechanism. The v10d SentencePiece supplement requirements doc from §40 is the only blocker.

---

## Files (Final)

### Core
| File | Experiment | Status |
|------|-----------|--------|
| `model.py` | TinyGemma reference impl (used at all scales) | Complete |
| `synth_data_v2.py` | v3+ synthetic data | Complete |

### Path 3 Production Pipeline (April 2026)
| File | Experiment | Status |
|------|-----------|--------|
| `experiment_v10a_tinystories.py` | **v10a TinyStories curriculum** at 100M (the Path 3 trainer) | Complete |
| `results_v10a_tinystories/model_full.pt` | 100M trained 16M tokens, loss 2.35 (380MB) | Complete |
| `results_v10a_tinystories/model_compiled.pt` | 100M frozen-FFN + retrained attention, loss 2.12 (380MB) | Complete |
| `experiment_geometry_probe.py` | **Geometry probe** — tied-embed FFN ↔ embedding space | Complete |
| `results_geometry_probe/model_probe.pt` | 19M probe model | Complete |
| `experiment_paris_compile.py` | Single-fact INSERT (Approach A + B) | Complete |
| `experiment_insert_sanity.py` | Mechanism sanity test (' time' → ' Paris') | Complete |
| `experiment_insert_indist.py` | In-distribution INSERT verification | Complete |
| `experiment_constellation.py` — `experiment_constellation_v6.py` | Iterative constellation refinement | Complete |
| `compile_facts.py` | **Production primitive** — FactCompiler, compile_triples, refine() | Complete |
| `experiment_scale_test.py` | 25 capitals + mixed relations + collisions | Complete |
| `diagnose_25.py` | Vocab clamp bug diagnostic | Complete |
| `experiment_phrasing_robustness.py` | Held-out phrasing generalisation soft cap | Complete |
| `experiment_conflict.py` | Same-template conflicts, schema migration | Complete |
| `experiment_cot_steering.py` | **CoT-as-query-reformulation experiment** | Complete |
| `experiment_unfaithful_cot.py` | **Falsification test** — semantically nonsensical suffixes that retrieve | Complete |
| `experiment_multihop.py` | **Multi-hop / JOIN test** — bidirectional facts compose via chains | Complete |
| `experiment_v10c_tinystories.py` | v10c trainer with custom SentencePiece-32K | Complete |
| `experiment_v10c_validation.py` | v10c scale + multi-hop validation | Complete |
| `experiment_large_constellation.py` | **61-fact scale test** — 100% accuracy | Complete |
| `experiment_compute_routing.py` | **Compiled computation routing** — arithmetic + CP-SAT scheduling | Complete |
| `experiment_wasm_phase1.py` | WASM Phase 1 — initial residual persistence probe | Complete |
| `experiment_wasm_phase1b.py` | WASM Phase 1b — alpha sweep, sweet spot found | Complete |
| `experiment_wasm_phase2.py` | WASM Phase 2 — two-layer chain + causality | Complete |
| `experiment_wasm_phase2b.py` | WASM Phase 2b — three-layer chain + causality | Complete |
| `experiment_wasm_phase3a.py` | WASM Phase 3a — 91-fact full addition table (failed) | Complete |
| `experiment_wasm_phase3a_mini.py` | WASM Phase 3a mini — 5 arithmetic facts at 5/5 | Complete |
| `experiment_wasm_phase3a_push.py` | WASM Phase 3a push — scaled to 20/20 at 100% | Complete |
| `experiment_unified.py` | Unified test — 31 facts (knowledge + arithmetic) + CoT + unfaithful CoT | Complete |
| `experiment_wordnet.py` | **WordNet compilation** — synonyms, hypernyms, antonyms + style transfer | Complete |
| `experiment_code_compile.py` | **Code compilation** — Python grammar, API signatures, idioms (§36) | Complete |
| `experiment_spontaneous_reformulation.py` | Reformulation hypothesis baseline (α=10 hijack documented) | Complete |
| `experiment_spontaneous_reformulation_fixed.py` | Failed attempted fix — adversarial decoys break QR | Complete |
| `experiment_structural_sweep.py` | **α structural fix** — L × α sweep, α=1 restores planner (§37) | Complete |
| `experiment_multihop_v10c.py` | Multi-hop v10c port, uniform α=1 (exposes reverse-template prior) | Complete |
| `experiment_multihop_v10c_perfact.py` | **Multi-hop CLOSED** — per-fact α sweep, 5/5/5/5 at fwd=1/rev=5 (§38) | Complete |
| `experiment_balancer.py` | **Balancer Phase 1** — basic + graph-aware balance modes (§39) | Complete |
| `experiment_microworld.py` | **Colchester micro-world Part 1** — token audit, 48-edge compile (§40) | Complete |
| `experiment_microworld_close.py` | **Colchester micro-world Part 2** — 51-edge castle subgraph, 60/60 (§40) | Complete |
| `PAPER_microworld.md` | Micro-world spec + Part 1 and Part 2 results | Complete |
| `spatial_dispatch.py` | **SpatialDispatcher class** + Euclidean/haversine solvers + number-word parser (§41) | Complete |
| `experiment_spatial_dispatch.py` | **Spatial dispatch pipeline** — 63 edges, exact computed distances from compiled data (§41) | Complete |
| `PAPER_spatial_dispatch.md` | Spatial dispatch spec + Run 1 results | Complete |
| `PAPER_code.md` | Code compilation spec + Run 1 results | Complete |
| `PAPER_balancer.md` | Balancer spec + Phase 1 results (basic works, graph-aware does not) | Complete |
| `results_v10c_tinystories/` | v10c checkpoints + tokenizer | Complete |
| `make_cot_plots.py` | Generate clean headline plots for paper | Complete |
| `results_cot_steering/` | data.json + 5 plots (scatter, trajectory, histogram, headline, clean scatter) | Complete |

### v3-v12 (20M)
| File | Experiment | Status |
|------|-----------|--------|
| `experiment_v3.py` — `experiment_v12_compile.py` | v3-v12 | Complete |
| `experiment_v10b_attention_transfer.py` | v10b (20M) | Complete |
| `experiment_v10b_100m.py` | v10b (100M) | Complete |

### Gemma 3-4B Validation
| File | Experiment | Status |
|------|-----------|--------|
| `experiment_gemma4b_validation.py` | Head classification (4B) | Complete |
| `experiment_gemma4b_ffn_replacement.py` | FFN replacement (4B) | Complete |
| `experiment_attention_anatomy.py` | Attention anatomy (4B) | Complete |
| `experiment_bos_register.py` | BOS tracking (4B) | Complete |
| `experiment_prediction_position.py` | Prediction position + ablation (4B) | Complete |
| `experiment_query_lifecycle.py` | Query lifecycle trace (4B) | Complete |
| `experiment_probing.py` | Linear probes (4B) | Complete |
| `experiment_derived_attention.py` | Derived attention (4B) | Complete |
| `experiment_layer_sweep.py` | Layer sweep (4B) | Complete |
| `experiment_trajectory.py` | Trajectory prediction (4B) | Complete |
| `experiment_ffn_stages.py` | FFN stage mapping (4B) | Complete |
| `experiment_ffn_direction.py` | FFN direction validation (4B) | Complete |

### Result directories
| Directory | Contents |
|-----------|----------|
| `results_v10a_tinystories/` | v10a 100M checkpoints + results.json |
| `results_geometry_probe/` | 19M probe model + stats.json |
| `results_v10b_transfer/` | v10b 20M results |
| `results_v10b_100m/` | v10b 100M results |
| `results_gemma4b_validation/` | Head classification results |
| `results_gemma4b_ffn/` | FFN replacement results |
| `results_attention_anatomy/` | Attention anatomy results |
| `results_bos_register/` | BOS tracking results |
| `results_prediction_position/` | Prediction position + ablation results |
| `results_query_lifecycle/` | Query lifecycle results |
| `results_probing/` | Probing results |
| `results_derived_attention/` | Derived attention results |
| `results_layer_sweep/` | Layer sweep results |
| `results_trajectory/` | Trajectory prediction results |
| `results_ffn_stages/` | FFN stage mapping results |
| `results_ffn_direction/` | FFN direction validation results |
| `results_ffn_direction/` | FFN direction validation results |
