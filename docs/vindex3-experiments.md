# VINDEX3 Experimental Programme

**Version:** 1.0-draft-1
**Date:** 2026-08-01
**Status:** Pre-registered. Decision rules and promotion gates are fixed here before any arm runs.
**Companion:** [`vindex3-format-spec.md`](../crates/larql-vindex/docs/vindex3-format-spec.md)
**Registry programme:** `vindex2` (chuk-experiments) — kept at `vindex2` deliberately: arms are already recorded under that key, and re-keying a pre-registered programme would orphan its results. The gate ids `V2-x` below are external identifiers for the same reason. The *format* is VINDEX3; see the note at the top of the companion spec.
**Discipline:** DEC-style — every arm has a registered prior, a numeric gate, and a named negative outcome. A result that fails its gate closes its thread; it does not get re-argued.

---

## 0. Structure

Five gates (V2-0 … V2-4) sequence the work. Nine experiments (E0 … E8) supply the evidence. Four test assets carry the experiments — no single proxy answers everything:

| Asset | Scale | Answers |
| ----- | ----- | ------- |
| **Conformance fixtures A–D** | tiny, deterministic | format, loader, programme and manifest correctness |
| **Byte-faithful K3 bank** | 1–4 full-sized routed layers | SSD, mmap, segmentation, grouping, cache behaviour |
| **Loaded Gemma MoE** | existing real model | real buffers, real scheduling, end-to-end integration |
| **Kimi-Linear-48B-A3B-Instruct** | real model, laptop-runnable end-to-end | real shared-expert + sigmoid-router serving, hybrid dense/MoE manifests, KDA/MLA spine adapter de-risking, real same-lineage routing traces |
| **Inkling-Small (276B-A12B)** | real model, exceeds rig RAM | real 2-shared sink-router reduction, real NVFP4/MXFP8 regions + mixed-precision release convention, first forced partial-residency/remote serving |
| **K3 trace pack** | selected real layers | fidelity, routing locality, omission decisions |

Nothing about K3 locality is decided from Gemma. K3's quantile-balanced top-16-of-896 routing is designed to suppress exactly the per-layer collapse Gemma exhibits; Gemma proves methodology only (consistent with Milestone D in ROADMAP_STATUS).

---

## 1. Test assets

### 1.1 Conformance fixtures (functional, not trained)

Deterministic weights; purpose is numerical parity against a per-architecture oracle, not speed.

**A — Direct routed MoE (control)**
```
hidden 256 · experts 8 · top_k 2 · shared 0 · programme gated-mlp-v1
```

**B — GPT-OSS-shaped**
```
hidden 288 · intermediate 288 · experts 32 · top_k 4 · shared 0
programme gpt-oss-expert-v1 (clamp + residual term)
representation: MXFP4-like blocks + separate scale regions
```

**C — Inkling-shaped**
```
hidden 256 · experts 32 · top_k 6 · shared 2
programme routed + always-active shared bank
representations: BF16, FP8-like, FP4-like
```
Catches: shared-expert reduction order, router-normalisation participation, mixed routed/shared storage.

**D — K3-shaped (Mini-K3)** [dimensions frozen 2026-08-01]
```
hidden 512 · latent 256 · expert intermediate 256
experts 112 · top_k 16 · shared 2 · programme latent-moe-v1
```
Same architecture graph, same routing shape, same shared latent projections; small enough to regenerate and rewrite repeatedly.

**Why not the literal 1/16 downscale.** The draft-1 dimensions (448 / 224 / 192) break Q4_K/Q6_K superblock alignment on every matrix: `pad_cols_to_256` pads 224 → 256 (+14 %) and 192 → 256 (+33 %), so the fixture's byte ledger would be a padding artefact rather than a scaled K3. No proportional downscale fixes it — alignment needs `latent/d ≡ 0 (mod 256)` and `inter/d ≡ 0 (mod 256)`, and with `3584 = 2⁹·7`, `3072 = 2¹⁰·3` that admits only `d ∈ {1, 2}`. 512 / 256 / 256 is 256-aligned everywhere, preserves K3's 2:1 hidden:latent exactly, and pads nowhere. The one deviation is intermediate:latent, which is 1:1 here against K3's 1:0.857.

**Why 112 experts, not 56.** 56 preserves the literal 1/16 expert-count scaling but is not a whole multiple of the 16-expert group width, so it would bake a partial group into the fixture — and group width dividing segment width is a physical-design *rule* (§7), not a preference. 112 = 7 × 16 divides cleanly, stays tiny, and leaves exact-K3-scale fidelity where it belongs: the byte-faithful bank, which runs 896 unscaled.

**Standing caveat:** never quote a byte or latency number from this fixture as a K3 number. It answers correctness, not SSD.

### 1.2 Byte-faithful K3 routed bank

Opaque or deterministic quantised payloads at **real K3 expert byte sizes**: 896 experts, gate/up and down at real Q6_K lengths (per-expert 3×3584×3072 params; ~24.28 GB / 22.61 GiB per full layer — deliberately over the 20 GiB cap so segmentation is exercised, not simulated). One layer is a meaningful SSD experiment; four layers expose cache exhaustion and sustained behaviour.

### 1.3 Kimi-Linear-48B-A3B-Instruct (the K3 dress rehearsal)

Verified from the released `config.json` (`moonshotai/Kimi-Linear-48B-A3B-Instruct`, 98.3 GB BF16, ~3B active):

```
layers 27 · first_k_dense_replace 1 (layer 0 dense FFN @ intermediate 9216;
                                      layers 1–26 MoE)
hidden 2304
routed experts 256 · top-8 · shared experts 1 · moe_intermediate 1024
router: sigmoid scores · renormalised · routed_scaling_factor 2.446
        · grouped top-k (1 group — degenerate but present in the schema)
dense spine: 20 KDA layers + 7 MLA layers, 3:1 interleave
             (full_attn_layers = 4,8,12,16,20,24,27; 1-based)
MLA: kv_lora_rank 512 · qk_nope 128 · qk_rope 64 · v_head 128 · no q-LoRA
KDA: 32 heads × head_dim 128 · short_conv 4
```

Routed-bank arithmetic (Q6_K, 210 B / 256 params):

```
params per expert   = 3 × 2304 × 1024        =  7,077,888
bytes per expert    ≈ 5.54 MiB
per MoE layer (256) ≈ 1.38 GiB   → single segment, well under the cap
26 MoE layers       ≈ 36 GiB routed — whole model Q6_K ≈ 40 GB class
                      → runs end-to-end on the M3 Max
```

What it uniquely contributes: the only **real, decodable** shared-expert model in the set (fixture C stays synthetic for cheap conformance); the only real hybrid dense+MoE stack (per-layer manifest heterogeneity for free); a non-softmax router (sigmoid + renormalise + scaling factor) that stress-tests the manifest router vocabulary; and — most valuable — the **same KDA:MLA hybrid spine lineage as K3** at 1/30th checkpoint scale, so the class-1/class-2 adapter work (recurrence parameters, conv states, MLA latent KV, layer scheduling off `full_attn_layers`) is debugged here before the K3 adapter, the roadmap's main blocker. It also proves the single-segment routed path with real weights while the byte-faithful bank proves the multi-segment path — the two segmentation regimes each get a real test.

**Boundary:** KL-48B enters the *design set*. It is therefore ineligible as E8's held-out model, its sigmoid routing does not transfer locality conclusions to K3's quantile-balanced routing, and its BF16-only release means native low-bit region handling leans on GPT-OSS (MXFP4) and Inkling-Small (NVFP4/MXFP8, §1.4).

### 1.4 Inkling-Small (the K3 serving dress rehearsal)

Verified from the released `config.json` (`thinkingmachines/Inkling-Small`, 532 GB BF16 + 4.46 GB `mtp.safetensors`; 276B total / 12B active per the release notes; Apache-2.0):

```
layers 42 · hidden 4096 · dense MLP at layer index 2 (dense_intermediate 16384)
routed experts 256 · top-6 · shared experts 2 · moe intermediate 2048
router: sigmoid gate activation · gate bias · norm_after_topk · route_scale 8.0
        · global scale · shared_expert_sink = true (shared experts scored
          by the router — always active, inside normalisation)
attention: 5:1 local(SWA-512):global · GQA 32/8 · head_dim 128
           · SConv kernel 4 on residual branches · d_rel/rel_extent relative terms
aux: 8 chained MTP heads (separate safetensors) · vision hMLP patchifier
     · dmel audio encoder — auxiliary manifest-addressed tensors, optional
releases: BF16 + NVFP4 + MXFP8; quantised convention = routed experts low-bit,
          shared experts / attention / gates BF16 (a real mixed-precision index)
```

Routed-bank arithmetic (Q6_K, 210 B / 256 params):

```
params per expert    = 3 × 4096 × 2048       = 25,165,824
bytes per expert     ≈ 19.69 MiB
per MoE layer (256)  ≈ 4.92 GiB  → still single segment, under the cap
~41 MoE layers       ≈ 202 GiB routed — CANNOT be RAM-resident on the M3 Max
routed reads/token   ≈ 6 × 19.69 MiB × 41 ≈ 4.7 GiB at top-6
```

What it uniquely contributes: the real two-shared-expert reduction under a **sink router** — shared experts inside the scoring and normalisation, exactly the semantics fixture C could only fake, and the richest test of the manifest's router vocabulary (sigmoid + bias + post-top-k norm + route scale + global scale). Real **NVFP4/MXFP8 native regions** from the quantised releases, including the mixed-precision convention (routed low-bit, everything else BF16) that is itself a per-region-format and variants-model use case — native low-bit no longer leans on GPT-OSS alone. A **mid-stack dense layer** (`dense_mlp_idx: 2`) — per-layer manifests handle it for free where any `first_k_dense_replace` field would have failed. And the residency escalation: it is the first design-set model where partial-residency, SSD streaming and attn-local/FFN-remote profiles are **forced rather than optional** on the rig — the K3 *serving* dress rehearsal at one-eighth scale, complementing KL-48B as the K3 *adapter* dress rehearsal. MTP heads are stored as optional auxiliary tensors whose omission never changes authority (drafting only); multimodal towers are opaque auxiliary payload — the text backbone is the conformance target.

**Boundary:** Inkling-Small enters the design set — ineligible as E8's held-out model. Its sink routing is a third distinct balancing mechanism (vs KL-48B's grouped sigmoid and K3's quantile-balancing): more evidence that locality conclusions do not transfer between routers, and none of its traces inform K3 locality.

### 1.5 K3 trace pack (E6 prerequisite)

Real K3 routing/activation captures across prose, code, agent/tool use, long-context retrieval, multilingual, repeated multi-turn. Per layer/token: expert IDs, gate scores, selection order, co-activation sets, reuse distance, latent input norm, expert output norms, aggregate routed delta.

---

## 2. Gates

### Result log — fixture A container (2026-08-02)

The first VINDEX3 container to exist on disk. Recorded here rather than in the
spec because it is evidence, not specification.

```text
write_container        index.json schema 3 + moe_manifest.json + LYRW v2 bank
detect_generation      V3, from the written directory (not a JSON literal)
Vindex3Container::open manifest parsed and validated, storage keys resolved
region round-trip      every gate/up/down region byte-for-byte
bind + execute         BIT-IDENTICAL to the same weights held in memory
fused vs decomposed    BIT-IDENTICAL under one programme id (gated-mlp-v1)
verify                 structural, defects carry {layer, entry, role}
CLI                    show/verify dispatch on generation; v3 defect → exit 1
```

Rows now discharged:

| Gate | Row | Status |
| ---- | --- | ------ |
| V2-0 | indexes inspectable without loading weights | closed |
| V2-0 | unknown `programme_id` fails cleanly | closed (`MoeManifest::parse`) |
| V2-0 | missing regions diagnosed with coordinates | closed — `ContainerDefect::MissingRegion` carries `{layer, bank, role, segment}` plus entry |
| V2-0 | generation boundary enforced both ways | closed |
| V2-0 | a profile dropping a component cannot exceed derived authority | closed — by *pre-existing* tests in `capability/authority.rs` (`weakening_any_region_never_raises_authority` and four siblings), not by the container work |
| V2-0 | variant-selection refusal (select only present variants) | **open** — needs the profile/variant machinery wired to the container; `Vindex3Index::declares_profile` is only the name check |
| V2-1 | native oracle vs container output exact | closed (fixture A) |
| V2-1 | fused vs decomposed identical under one manifest | closed |
| V2-1 | expert counts and top-K not hard-coded | closed — (8,2), (32,4), (5,1) through one code path, each read from its own container |
| V2-1 | shared banks not hard-coded | **open**, and *blocked*: `BoundMoeOperation::banks` documents unrouted shared banks as arriving with the Mini-K3 rung, so there is no execution path to test against |
| V2-1 | WALK/DESCRIBE parity | **open** |

Two corrections worth carrying forward, both from getting it wrong first:

- **`gated-mlp-v1` admits both layouts** (`role_alternatives → [FUSED,
  DECOMPOSED]`). It is not "the decomposed programme"; `gated-mlp-fused-fc1-v1`
  is the one that narrows to fused only. So the parity arm is *one programme
  admitting two renderings*, which is the stronger property and the one K3
  needs.
- **Storage fidelity and execution fidelity need separate assertions.** A
  single execution test can pass while the writer stored the wrong bytes, if
  the reader makes a compensating mistake. The region round-trip test exists to
  make that impossible.

### V2-0 — Format skeleton

Build superblock, LYRW v2 header/bank/segment/region tables, manifest schema, profile inheritance, authority levels, checksums, required/optional role semantics. Tiny tensors only.

**Acceptance**
- indexes inspectable without loading weights;
- unknown `programme_id` fails cleanly;
- missing required regions diagnosed with `{layer, bank, role, segment}` precision;
- a profile that drops a required component cannot exceed its derived authority (§9.2) — verified by test, not review;
- **profile resolution**: a profile can select only variants physically present; selecting an absent variant fails naming region set, requested variant and present variants; no code path performs silent conversion; region-level fidelity aggregates to profile authority as weakest-link; a profile referencing incompatible segment sets is refused;
- generation boundary enforced: the v2 loader refuses a VINDEX2 directory (and vice versa) with a precise "requires VINDEX{n} loader" error naming both versions — never a parse error, never silent conversion (spec §6.6/§12.1).

### V2-1 — Generic reference executor (runs E1 arms functionally)

All four fixtures execute through the generic path and match their oracles.

**Acceptance**
- native oracle vs vindex2 output: exact (integer/quantised paths) or tolerance-bounded parity;
- fused vs decomposed FC1 storage produce identical results under one manifest;
- expert counts, top-K and shared banks are demonstrably not hard-coded (fixtures differ on all three);
- **WALK/DESCRIBE parity**: gate KNN over in-place bank regions returns identical top-K rankings to a v1-style extracted `gate_vectors.bin` control on fixtures A–C, and latent-space WALK on fixture D returns the correct ranking after the `routed_input` projection (spec §15.4).

### V2-2 — Physical layout (runs E1 layout arms + E2)

### V2-3 — Production kernel binding (runs E3)

Bind existing grouped kernels through the capability registry, Gemma first (then GPT-OSS as first non-Gemma real model).

**Acceptance**
- reference and grouped paths agree per-layer and at final logits;
- caller-owned buffers/command buffers work; no hidden decode-time repacking;
- mixed per-region formats supported or explicitly refused, never coerced.

### V2-4 — Real-model portability

Import and execute in order: Gemma MoE → GPT-OSS → **Kimi-Linear-48B-A3B** → **Inkling-Small** → K3. Rung 3 (KL-48B): full import, exact decode against the reference implementation (token parity), then grouped dispatch with the shared expert in the execution path and sigmoid-renormalised-scaled gate weights flowing through the standard reduction. Rung 4 (Inkling-Small): BF16 import + token parity on the text backbone, then the NVFP4 release imported as paired values/scales regions under the mixed-precision convention, then the first forced partial-residency / attn-local-FFN-remote serving profiles — rung 4 acceptance explicitly includes decoding a model that does not fit in RAM. Fixture C is retained as the tiny deterministic conformance fixture only. Only after V2-0..V2-3 pass does the one-time K3 extraction run.

---

## 3. Experiments

### E0 — VINDEX2 preservation matrix (continuous regression, not a one-shot)

VINDEX3 development must not degrade the shipped generation. E0 runs using the same binary that carries the v2 code, from the first V2-0 commit onward, in CI.

#### E0 corpus — constructed and pinned [amended 2026-08-01]

Draft-1 said "existing production v1 indexes". On the actual rig that named **nothing**: no vindex of any generation exists on the box, so the experiment's premise was empty and E0 would have passed vacuously. The corpus is therefore constructed and pinned, not assumed:

| | Corpus | Purpose |
| - | ------ | ------- |
| **C1** | A fresh Gemma v1 extract at **`--level all`**, plus all eight slice presets cut from it (`client`, `attn`, `embed`, `server`, `browse`, `router`, `expert-server`, `all`) | The primary subject. Covers every v1 path the matrix exercises. `--level all` is required, not preferred: the `all` preset needs `lm_head` and the `router` preset needs router weights, neither of which `--level inference` extracts — a lower level would silently reduce preset coverage to those the extract happened to reach. |
| **C2** | Published v1 artifacts pulled from the hub | Exercises the `publish`/`pull` path and the generation stamp against artifacts this box did not build. |
| **C3** | **Golden outputs captured once from the pre-v2 binary and committed** | The comparison baseline. |

**C3 is load-bearing.** Without committed goldens, "zero behavioural regression" quietly degrades into "the two binaries agree with each other" — a condition any *shared* bug satisfies. The goldens must be captured from a binary that predates the v2 code, so the assertion is against a fixed record rather than a live second run.

**Pin recipes, not artifacts.** C1 is a multi-GB directory and C2 is a download; neither belongs in the repo, and neither needs to — both are *reproducible*. What gets committed is the recipe and the outputs:

| Item | Committed? | Why |
| ---- | ---------- | --- |
| C1 vindex | No — **recipe pinned** | Re-extractable from the checkpoint at any time. Pin: model repo + revision hash + exact extract flags (`--level all --quant q4k`). |
| C2 hub artifacts | No — **coordinates pinned** | Re-pullable. Pin: artifact ref + expected checksums. |
| C3 goldens | **Yes** | Small text. The only thing that cannot be regenerated *later* without also regenerating the binary that produced it. |
| C3 baseline binary | No — **commit SHA pinned** | `git checkout <sha> && cargo build --release -p larql-cli` reproduces it. Git keeps the SHA reachable indefinitely. |

**There is no expiry.** An earlier draft of this section claimed C1/C3 had to be captured before VINDEX3 merged, on the grounds that the pre-v2 baseline stopped being available. That was wrong: the checkpoint still extracts and the baseline commit is still checkoutable. The correct reason to build the corpus early is that E0 is specified as *continuous* — it should be catching regressions as later V2-0 work (manifest, profiles, capability checking) touches shared code, not auditing at the end. Early because the gate is meant to be live, not because anything is running out.

**What must be recorded, or the corpus is unfalsifiable:** the baseline commit SHA inside the golden set itself. Without it a reader cannot tell whether a given golden predates the v2 work it is supposed to police.

**Matrix**

```
load a C1/C2 v1 index           → identical ModelWeights surface
provenance / checksums          → larql verify byte-identical verdicts
browse extraction + WALK        → identical top-K rankings
attention-only client slice     → loads, serves
full inference                  → token-identical decode vs pre-v2 binary
layer sharding (--layers)       → identical RSS bound behaviour
expert sharding (--experts)     → identical 404-before-read behaviour
publish / pull round-trip       → hub artifacts unchanged; generation stamp added
                                  without invalidating existing v1 artifacts
generation dispatch             → index.json.version 2→3 routing correct;
                                  unknown version fails naming both sides
```

#### Two gates, named separately [amended 2026-08-01]

"E0 green" must never be reported when only part of it ran. The matrix splits by what a runner can actually execute:

| Gate | Covers | Where |
| ---- | ------ | ----- |
| **E0-CI** | Generation/schema boundary and every weight-free compatibility check. Needs no checkpoints. | Required merge check on every PR touching the crate |
| **E0-FULL** | The checkpoint-backed matrix — token-identical decode, WALK rankings, slicing, sharding, publish/pull — against the committed C3 goldens | Local, against a real index; status recorded as "green at commit `<sha>`" |

Reporting convention:

```text
E0-CI:   green on every merge
E0-FULL: green at last recorded local run <commit>
```

E0-CI is deliberately the subset most at risk from VINDEX3 work: every commit touching the loader, the generation module or the LYRW parsers can break it, and nothing else in CI would notice. It caught a real regression on its first run — `index.json` schema 1 refused as a non-generation — which is the argument for it existing.

A later improvement would add a tiny synthetic VINDEX2 artifact exercising load, verify, slice and one trivial decode in CI. That would not replace the multi-GB goldens, but it would narrow the gap between boundary-only checking and full local validation.

**Acceptance:** zero behavioural change on every v1 path, measured **against the C3 goldens** — token-identical serving, identical verify verdicts, identical slice/publish/pull outputs. The K3 artefact itself only ever needs v2; E0 protects everyone already on v1.

**Decision rule:** any E0 regression blocks merge, full stop. VINDEX3 defaulting for new extractions (§12.1 support policy) is gated on E0 green plus the ABI freeze.

### E1 — LYRW physical-layout gate

Determines whether v2's layout ideas beat the current format before any K3 byte is written.

**Arms**
```
L0  current LYRW v1 (one format per layer file)
L1  LYRW v2 per-region formats, same file & offset-table shape
L2  projection-separated files (layer_N.gate_up / layer_N.down)
L3  expert-group files (layer_N.experts_000_031 …)
```

Run first on Mini-K3, all arms with identical Q6_K weights.

**Measure:** exact output parity; loader complexity; mmap/open count and latency; dispatch construction time; bytes faulted; sequential and random top-16 execution; manifest-level omission/override capability; **gate-only browse cold bytes** (a WALK sweep touching only gate regions — the browse-read cost of each layout).

**Decision rule.** Choose the **least fragmented** layout within:
- 2% of best warm execution;
- 5% of best cold physical bytes read;
- exact parity.

Do not promote L2/L3 for conceptual tidiness. **Registered prior: L1 wins.** The fused-vs-decomposed gate/up question inside L1 is **not** settled here — E1 records the serving delta between the two; E7 records the browse delta; §15.2's per-bank rule reconciles them at extraction time.

### E2 — Byte-faithful SSD replay (segment & ordering)

On the 1–4-layer byte-faithful bank, replay routing traces under memory budgets of 2/4/8/16 GB:

| Trace | Purpose |
| ----- | ------- |
| Uniform random top-16 | worst-case balanced routing |
| Zipf/skewed | strong hot-expert case |
| Gemma empirical | existing real proxy |
| Kimi-Linear empirical | real same-lineage MoE traces (sigmoid top-8-of-256) — a better structural proxy than Gemma, still NOT a K3 locality substitute |
| Inkling-Small empirical | real sink-router top-6-of-256 traces captured during rung-4 serving — a third router family for the replay table; same non-transfer rule applies |
| Clustered co-activation | tests physical grouping |
| Adversarial rotation | defeats cache and prefetch |
| Repeated conversation | temporal locality |

**Sweep — two independent scales** (spec §7): group-extent width ∈ {8, 16, 32, 64} experts × segment width ∈ {112, 224, 448, 896-split-2} experts, plus physical orderings ID / frequency / co-activation-cluster / random (via the entry-table indirection — ordering is independent of both widths). Group width optimises reads/prefetch/dispatch; segment width optimises file count, mmap management and the 20 GiB cap — the winning pair is reported separately, never as one number.

**Measure:** actual SSD bytes/token; useful bytes vs read amplification; page faults; read count and extent sizes; p50/p95/p99 layer latency; degradation onset; late steady-state floor; next-layer prefetch effect; permutation effect.

**Decision rules.**
- Group width: simplest width within a small margin of best sustained result; must divide the segment width and match a grouped-kernel dispatch width.
- Segment width: as large as the 20 GiB cap and shard-distribution needs allow (prior: 448 for K3 Q6_K); must be a whole multiple of the group width.
- Physical ordering: bake a non-ID permutation into the permanent index **only** on ≥10% sustained improvement across multiple plausible traces with no bad uniform-routing regression. **Registered prior: per-expert offsets + runtime cache beat permanent locality encoding.**

### E3 — Loaded-model grouped A/B (critical path; doubles as first V2-3 production experiment)

On loaded Gemma MoE:

```
A  existing production expert layout
B  current LYRW v1 + grouped execution
C  LYRW v2 (E1 winner) + grouped execution
D  split-file layout + grouped execution   (kept only if E1 didn't kill it)
```

Hold constant: representation, selected experts, routing weights, activation layout, command-buffer ownership, output reduction.

**Measure:** per-layer output equivalence; final logits/token equivalence; command submissions and syncs; warm and cold latency; full-model decode throughput; memory and SSD traffic.

Pass = the E1/E2 winner survives real buffers, real scheduling and the actual forward path. This is the final proxy gate before K3 integration (per the release ladder's K3-0).

**E3b — repeat on Kimi-Linear-48B** once V2-4 rung 3 imports it: same held-constants, arms B/C only. This is the first grouped A/B with a shared expert inside the dispatch path and non-softmax gate weights in the reduction — the two things Gemma cannot exercise — and it doubles as the KDA/MLA-spine integration shakedown for the K3 adapter.

### E4 — Projection-specific quantisation (the decisive LYRW v2 justification)

**Arms**

| Arm | gate/up | expert down |
| --- | ------- | ----------- |
| Q0 | Q6_K | Q6_K |
| Q1 | native MXFP4 | Q6_K |
| Q2 | Q6_K | native MXFP4 |
| Q3 | native MXFP4 | native MXFP4 |
| Q4 | approx lower | Q6_K |
| Q5 | Q6_K | approx lower |

No Cartesian sweep. Three stages: (1) Mini-K3 numerical mechanics, (2) loaded Gemma full-model proxy, (3) real K3 layer replay once the trace pack exists.

**Measure:** expert-output cosine and normalised error; weighted aggregate latent output; next-layer router agreement; teacher-forced BPB; token-level KL; greedy agreement; long-generation stability; measured bytes and kernel throughput.

**Decision rule — reframed.** Per-region format tags are in the ABI **structurally**: native value/scale codecs, v1's existing mixed gate/up/down precision, and format-neutral banks justify representability without a performance argument. E4 therefore decides one thing only: **whether a mixed-format profile is promoted to Production** — requiring a mixed arm to beat both uniform alternatives by ≥5% projected whole-system gain at the same fidelity gate. A failed E4 leaves the format intact and the mixed profiles at Reference/Grouped maturity (representable ≠ servable).

### E5 — Weight omission (profile semantics; NOT an ABI gate)

E5 decides profile authority and approximation policy. No E5 outcome changes a byte of LYRW2 — it cannot block the freeze.

Disambiguation first: K3's shared latent `routed_expert_down_proj` (7168→3584) is not an expert's `w2` (3072→3584). Dropping a selected expert's `w2` yields no expert output — dominated by skipping the expert. Arms therefore test **useful** omission modes:

```
O0  exact baseline (top-16)
O1  reduced top-K: 16/12/8/4 — with and without gate renormalisation
O2  cumulative gate-mass retention: 99 / 97.5 / 95 / 90 %
O3  shared-only layers — per-layer interventions, then structured patterns
    (every 4th MoE layer; early/middle/late; lowest-measured-impact)
O4  entire routed branch remote — whole-branch RPC (~14 KB/layer f16)
    vs projection-split (~100 KB/layer): do NOT split unless measurement
    overturns the arithmetic
O5  down-only lower precision — all experts execute; cheap w2 representation
    (the honest "cheap down"; feeds E4/Q2/Q5)
```

**Promotion gate (production approximation):** ≤0.5% BPB regression; no pathological p99 KL; no next-layer routing-agreement collapse; no recurrent/long-generation instability; gain recomposed against the whole-model byte ledger. A layer-selective O3 result is valuable even if whole-model routed removal fails.

### E6 — Real K3 routing/locality traces

The only experiment that cannot be run honestly on Gemma or synthetic routing — **nor on Kimi-Linear**: KL-48B traces (cheap to capture locally) pilot the E6 capture/analysis machinery and feed the E2 replay table, but its sigmoid grouped-top-k router and K3's quantile-balanced router are different balancing mechanisms, so no K3 locality, hot-set or retention decision transfers from it. Required **before** freezing any physical expert permutation, permanent hot set, co-activation grouping, or per-layer retention policy.

**Compute:** per-layer frequency entropy; cumulative mass curves; co-activation matrices; temporal reuse-distance distributions; static-LFU / LRU / hybrid per-layer cache hit rates; hot-set stability across workloads.

**Decision rule — scope.** E6 blocks exactly one physical decision: adopting a non-ID expert permutation (via E2's ≥10% bar re-run on real K3 traces). It does **not** gate the ABI freeze — the ID-order layout is always legal, and locality otherwise stays runtime metadata permanently. Cache policy, residency and retention conclusions feed profiles, not the format.

### E7 — Query-layer conformance and browse economics

Proves "the model IS the database" survives v2 as a measured property, not a slogan.

**Arms** (on fixtures A–D, then the byte-faithful bank, then loaded Gemma):

```
W0  VINDEX2 control: WALK over an extracted gate_vectors.bin
    — DENSE MODELS ONLY, see the coverage note below
W1  in-place WALK over decomposed gate regions (f16)
W2  in-place WALK over fused gate_up regions, strided (f16 row-major)
W3  in-place WALK over block-quantised gate regions (lazy dequant)
W4  gate-only browse SLICE built by region copy (spec §15.5)
```

#### Coverage note — W0 is not a control for expert regions [amended 2026-08-01]

Measured on a fresh VINDEX2 extract of Gemma 4 26B A4B (128 experts × 704 expert width, hidden 2816, 30 layers): the shipped generation's `gate_vectors.bin` exposes **2,112 walkable features per layer** — the *dense* FFN width. The expert population would contribute **128 × 704 = 90,112**. The expert weights are present and decode correctly (30 files, 12 GB); they are simply **not part of the searchable surface**.

So W0 covers 2.3 % of what §15.1 specifies, and on a MoE model it contains no expert regions at all. W1/W2/W3/W4 are *about* expert regions. Comparing them to W0 would measure a coverage difference wearing a parity result's clothes, and the registered pass condition ("identical top-K rankings") is unevaluable because the two arms do not rank over the same population.

**Resolution — the claim is restated, not the control rebuilt.** Making expert features searchable is a **new capability of VINDEX3, not parity with VINDEX2**. Writing it up as parity would be claiming a comparison that cannot be made. Concretely, E7 splits:

| Claim | Arms | Baseline |
| ----- | ---- | -------- |
| **Parity** — in-place region WALK matches the extracted-index path | W0 vs W1 on a **dense** model | W0, a genuine like-for-like control |
| **Capability** — expert features are searchable in place, with no separate index | W1/W2/W3/W4 on MoE | none exists; correctness is established against the weights themselves, not against W0 |

The rejected alternative was constructing an expert-bearing VINDEX2 index specifically to serve as a control. It would be a control built *for* the experiment rather than the thing that shipped — a weaker claim that reads as a stronger one.

**Measure:** top-K ranking parity vs W0 **on the dense arm** (exact for f16; ranking-overlap metric for W3, with the v1 §12.2 4-bit noise caveat as the expected failure shape); for the capability arm, ranking correctness against directly-computed gate dot products rather than against W0. Plus, for both: cold bytes faulted per WALK; queries/sec warm; slice size vs v1's ~3 GB browse economics; latent-WALK correctness and query-projection overhead on fixture D and the K3-shaped bank; DESCRIBE/SELECT correctness against `query/` sidecars, including latent-bank `down_meta` computed through the full output path.

**Decision rules.**
- W1 vs W2 sets the browse half of the §15.2 fusion decision: fused storage keeps browse eligibility only if strided WALK is within 10% of decomposed on cold bytes and warm throughput; otherwise browse-enabled indexes mandate decomposed gate/up.
- W3 promotes quantised-region browse only at a pre-declared ranking-overlap floor (top-50 overlap ≥ 0.9 vs W0); below it, browse-enabled extraction keeps gate at f16 regardless of the serving format — a legitimate per-region format divergence that E4 machinery already supports.
- W4 must reproduce W1 rankings bit-for-bit (it is the same bytes relocated), or the slicer is wrong.

### E8 — Held-out architecture (generalisation, not fit)

The conformance fixtures and design-set models cannot prove spec §16 criterion 7, because the ABI was designed against them — and **Kimi-Linear-48B and Inkling-Small are now in the design set, so both are ineligible here**. E8 onboards a MoE deliberately excluded from the design set — candidate: Mixtral 8x7B or a Qwen-MoE (cheap, well-documented, conventional) — **after** the ABI freeze, under a hard rule:

```
allowed:    checkpoint importer, MoE manifest, programme adapter
            (existing registered programmes only, or ONE new programme_id
             using existing region roles)
forbidden:  LYRW2 byte-layout changes · new region roles · new packing modes ·
            kernel-interface changes · loader special cases keyed on the model
```

**Acceptance:** token parity against a trusted implementation through the generic reference path, then through grouped dispatch if a compatible kernel exists — with a diff of the vindex crates showing zero format-layer changes.

**Falsification is a real outcome:** if E8 requires format changes, spec §16 downgrades the substrate claim to "K3/GPT-OSS/Inkling serving format" in writing, and the needed changes are queued for a future LYRW revision — they are not smuggled in retroactively.

---

## 4. Run order

**Run now (no K3 adapter required)**
0. E0 preservation matrix wired into CI (stays green for the life of the programme)
1. Fixtures A–D + Mini-K3 → V2-0, V2-1
2. E1 layout gate
3. E2 one-layer byte-faithful bank (extend to 4 layers if one layer is inconclusive)
4. E3 loaded Gemma grouped A/B
5. E4 stages 1–2
6. E7 query-layer conformance (fixtures + Gemma; the K3-shaped latent-WALK arm rides the byte-faithful bank)

**Freeze the ABI** — after E0 green + E1/E2/E3/E7 decided + V2-0/V2-1 accepted:
7. LYRW `format_version=2`, `index.json` v3, `vindex_spec_version=2`, MoE manifest v1. E5/E6 do **not** gate this (they decide profiles and permutation only); E4 does not either (per-region tags are structural, E4 gates promotion).

**After freeze, before the K3 extraction** (profile and locality decisions):
8. V2-4 rung 3: Kimi-Linear-48B import → token parity → E3b grouped A/B — the K3 adapter dress rehearsal (KDA/MLA spine, shared expert, sigmoid router)
9. KL-48B trace capture — pilots the E6 machinery, feeds the E2 replay table
10. V2-4 rung 4: Inkling-Small BF16 import → text-backbone parity → NVFP4 variant import (paired values/scales, mixed-precision convention) → first forced partial-residency + attn-local/FFN-remote serving — the K3 serving dress rehearsal
11. E8 held-out architecture — the generalisation test runs against the frozen ABI
12. K3 trace pack (E6 capture)
13. E4 stage 3 on real layers (mixed-profile Production promotion)
14. E5 on real layers (O1–O5)
15. E6 locality/cache simulation (gates only a non-ID permutation)

**Then extract K3 once**, into the frozen five-class layout, exact Q6_K baseline (fidelity recorded per §9.2 — `source-equivalent` if losslessly containing native values, `numerically-approximate` otherwise), two 448-expert segments per routed layer with group extents at the E2-chosen width, no locality decisions baked in.

---

## 5. Registered priors (falsifiable, dated 2026-08-01)

1. One file-set per routed layer (segmented past 20 GiB) remains correct.
2. One entry per expert remains correct.
3. **(Revised 2026-08-01, pre-freeze, on reinstating the query layer.)** Down stays independently addressable. Gate/up fusion is no longer a blanket prior: serving-only indexes fuse; browse-enabled indexes default to decomposed, reconciled per bank by E1 (serving delta) + E7 (browse delta) under §15.2.
4. **(Revised on adopting the variants model.)** Per-region format tags are in the ABI structurally; the prior is now that no mixed-format *profile* clears E4's 5% Production bar on Gemma, and the first to clear it does so only with real K3 layers (E4 stage 3).
5. Expert locality remains runtime metadata (E2/E6 fail the 10% bar).
6. "Dropping down" locally resolves to "skip the routed branch," never a half-expert mode (E5).
7. Exact remote deployment moves the whole routed branch, not split projections (O4 arithmetic holds).
8. L1 wins E1; L2/L3 die there.
9. **(Revised 2026-08-01, on measuring the VINDEX2 control.)** Split in two, because the original prior conflated a parity claim with a capability claim. **(9a)** On a *dense* model, in-place WALK over decomposed f16 gate regions matches the VINDEX2 `gate_vectors.bin` path within noise (E7/W0-vs-W1). **(9b)** On a *MoE* model, expert features are searchable in place at all — which VINDEX2 does not do, so there is no baseline to match and no dedicated query index is needed because none previously existed for this population. Falsifier for 9b is a correctness failure against directly-computed gate dot products, not a parity failure against W0.
10. Quantised-region browse (E7/W3) fails the ranking-overlap floor for ≤4-bit formats, and browse-enabled indexes keep gate at f16 as a per-region divergence — the query layer becomes the second consumer of per-region format tags after E4.
11. Dual-generation support costs nothing on the v1 hot path — E0 stays green throughout without a single v1-loader change (the generations share a trait, not code).
12. E8 passes: the held-out conventional MoE onboards with an importer + manifest + existing `gated-mlp-v1` programme, zero format-layer diffs — the envelope generalises beyond its design set.
13. K3's exact-Q6_K baseline lands at `source-equivalent`, not `source-exact` — and nothing downstream ever quotes it as bit-faithful to the release encoding.
14. Kimi-Linear-48B onboards through `gated-mlp-v1` + shared bank + existing region roles — its entire KDA/MLA spine lands in classes 1–2 as manifest-addressed tensors, touching the expert-bank format not at all.
15. The KL-48B→K3 adapter delta is confined to the five documented K3 additions (SiTU-GLU, AttnRes, latent MoE transforms, MLA output gate, full-rank KDA gate) — i.e. the dress rehearsal genuinely de-risks the main blocker rather than merely preceding it.
16. Inkling-Small onboards through `gated-mlp-v1` + shared bank + existing region roles; the sink router, gate bias, post-top-k norm and route/global scales are entirely manifest router-vocabulary items — no new region role, no new programme beyond a router descriptor.
17. The NVFP4 and MXFP8 releases import as paired values/scales regions using the existing packing vocabulary (packing 2/3 + pair_id) — no new packing mode is needed; if one is, that is an ABI-RC finding, not a post-freeze patch.

Any prior falsified is recorded here with the run ID and the format consequence, then the spec is amended before freeze — never after.

---

## License

Apache-2.0
