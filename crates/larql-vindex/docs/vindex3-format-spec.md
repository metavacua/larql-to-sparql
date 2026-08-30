# Vindex Format Specification — VINDEX3

**Version:** 3.0-draft-2
**Date:** 2026-08-01 (draft-2: three binary-layout corrections + two clarifications from the first lyrw2 implementation — §6.2, §6.4, §6.3, §6.5; recorded per pre-freeze amendment rule)
**Status:** Draft — pre-registration. No byte is frozen until the V2-0..V2-2 gates in the companion experiments document pass. **Five production models now encode and execute through this format**: gpt-oss-20b, Gemma 4 26B-A4B, and Granite 4.1 3B/8B/30B all round-trip through `larql vindex3 encode` → `inspect` → `verify` → `exec` byte-identically to their HF sources, and `larql serve` serves a V3 container over `/v1/completions` (see `docs/vindex3-runtime.md`). The conformance fixture-A round-trip through `format::vindex3` (write → detect → open → validate → bind → execute) also holds bit-identically, with fused and decomposed FC1 renderings satisfying the same programme id. The remaining pre-freeze rows (profile-authority derivation, variant-selection refusal, the not-hard-coded fixtures B–D, WALK/DESCRIBE parity) are still open, so the ABI is **not** frozen and `extract` still writes VINDEX2.
**Predecessor:** [format-spec.md v0.4](format-spec.md) (VINDEX2)

**Companion:** [`vindex3-experiments.md`](../../../docs/vindex3-experiments.md) (pre-registered experimental programme), [Conformance v1](conformance-v1.md), [Operations](operations-spec.md), [Ecosystem](ecosystem-spec.md), [LQL](../../../docs/lql-guide.md)
**Implementation target:** `larql-vindex` crate (Rust)

> **A note on the number 2 appearing throughout.** This document specifies
> **VINDEX3**, and its own version is therefore `3.x`. Three things nearby keep
> a `2` on purpose and are not typos:
>
> | name | why it stays |
> |---|---|
> | `V2-0`…`V2-4` gates | pre-registered identifiers with results already recorded against them; renaming would orphan the lineage |
> | registry programme `vindex2` | same — it is an external key in chuk-experiments |
> | `lyrw2` / `FORMAT_VERSION = 2` | the *bank* format's own version, on a different axis: a VINDEX2 container holds LYRW v1 files, a VINDEX3 container holds LYRW v2 files (see `format::generation`) |
>
> Only the container is versioned 3. An on-disk `index.json` carrying
> `"version": 2` is a **VINDEX2** file — the predecessor above, not a draft of
> this one.
>
> Real VINDEX3 containers now exist for conformance fixture A
> (`format::vindex3`), so the format is no longer validated only through
> VINDEX2-sourced operands. Real *models* are still VINDEX2: `extract` does not
> emit VINDEX3 and will not until the outstanding V2-0/V2-1 rows close.

---

## 1. What is VINDEX3?

VINDEX3 is a **general-purpose serving container for sparse models** — serving meaning both inference *and* the LQL query surface (WALK, DESCRIBE, SELECT, EXPLAIN). It is the successor to the VINDEX2 dense/Gemma-oriented layout, and it exists to answer one question well:

> Extract a supported checkpoint **once** into a stable, component-addressed layout, then vary **what is loaded, where it resides, what precision it uses, and whether a component is executed or queried** — without ever rebuilding the index.

**The model IS the database** remains the founding principle, not a VINDEX2 legacy: the gate regions inside LYRW v2 banks *are* the KNN index, exactly as `gate_vectors.bin` *was* W_gate. VINDEX3 does not add a query index next to the weights; it keeps the weights queryable (§15).

The key principles of VINDEX2 are retained unchanged:

- **The model IS the database.** Each weight tensor is stored once, canonically, in its serving format. Nothing is stored twice.
- **Weights are separated by function, not by file size.** Sharding follows what inference does with a weight, not an arbitrary byte boundary.
- **mmap-first.** Every physical object is independently mmap-able; the OS pages in only what execution touches.
- **Loaders dispatch on declared tags, never sniff filenames.**
- **Fail closed.** A profile that lacks a required operand refuses to load with a precise diagnosis; it never silently degrades authority.

The genuinely new pieces in VINDEX3 are exactly five:

1. **Per-region quantisation.** Format belongs to each weight region, not to the entire layer file.
2. **Multiple physical segments per logical expert bank.** A logical layer can span several files without changing model semantics.
3. **Multiple bank kinds** — routed, shared and dense/hybrid — with declared geometry.
4. **A validated MoE programme manifest** describing router, banks, transforms and combine semantics, replacing the Gemma/Mixtral-shaped `moe_config`.
5. **Representation variants.** A region may carry several physically present encodings; profiles *select* among them and never request formats that were not extracted (§9.1).

### 1.1 K3 validates the format; it does not define it

The conformance envelope is defined by three real architectures plus a control:

| Model | Routing | Shared experts | Expert space | Expert programme | Native format |
| ----- | ------- | -------------- | ------------ | ---------------- | ------------- |
| Direct MoE (control) | top-2 of 8 | 0 | residual | gated MLP | any |
| GPT-OSS | top-4 of 32/128 | 0 | residual | clamped gated MLP w/ residual term | MXFP4 |
| **Inkling-Small (276B-A12B)** | top-6 of 256, sigmoid + gate bias + norm_after_topk + route_scale 8.0; **shared-expert sink** (router scores shared experts) | 2 (always active) | residual | gated MLP; dense MLP at layer index 2 (mid-stack) | BF16, with NVFP4 / MXFP8 releases |
| **Kimi-Linear-48B-A3B** | top-8 of 256, sigmoid + renormalise + 2.446 scaling | 1 | residual | gated MLP; layer 0 dense (`first_k_dense_replace=1`) | BF16 |
| K3 | top-16 of 896 | 2 | latent 3584 | SiTU-GLU latent expert | (extraction: exact Q6_K baseline) |

Inkling-Small replaces the hypothetical "Inkling-shaped" envelope member with the real released model (`thinkingmachines/Inkling-Small`, 532 GB BF16 + 4.46 GB MTP sidecar). It contributes what nothing else in the set can: real two-shared-expert reduction under a **shared-expert sink** router (shared experts inside the scoring/normalisation, `norm_after_topk`, gate bias, global scale — the richest router-semantics test in the envelope); real **NVFP4/MXFP8 native regions** via its quantised releases, whose mixed-precision convention (routed experts low-bit, shared experts and attention BF16) is itself a per-region-format use case; a mid-stack dense layer (`dense_mlp_idx: 2`) proving per-layer manifests handle arbitrary dense/MoE schedules, not just leading-dense; and — decisive for the rig — it is the first design-set model that **cannot be RAM-resident** on the M3 Max (~202 GiB routed at Q6_K, ~4.9 GiB per MoE layer, ~4.7 GiB of routed reads per token at top-6), so partial-residency, SSD-streaming and attn-local/FFN-remote profiles get their first *non-optional* real-model test at one-eighth K3 scale. Its MTP heads and multimodal towers are stored as optional auxiliary manifest-addressed tensors — the text backbone is the conformance target; the towers are opaque payload, and omitting MTP never changes authority (drafting only).

Kimi-Linear-48B-A3B earns its seat three ways. It is the only **real, locally runnable** shared-expert member (98.3 GB BF16, ~3B active — Q6_K-class fits the M3 Max), so V2-4's shared-bank rung is proven on an actual checkpoint rather than a fixture. Its `first_k_dense_replace=1` hybrid stack is the only member exercising per-layer manifest heterogeneity (dense layer 0, MoE layers 1–26) on a real model. And it is K3's direct lineage ancestor — the same KDA(3):MLA(1) hybrid dense spine, 20 KDA + 7 MLA layers of real recurrence parameters — making it the dress rehearsal for the K3 adapter's class-1/class-2 plumbing at one-thirtieth the checkpoint size. Its sigmoid-scored, renormalised, scaled router also stress-tests the manifest's router vocabulary beyond softmax-top-k.

K3 is the stress test — largest bank, latent expert space, shared pre/post projections. GPT-OSS and Inkling exist in the envelope precisely to stop K3-specific assumptions (a tensor literally named `gate_up`, top-16, no-shared-bank, residual-space-only) from becoming the ABI.

---

## 2. Scope and non-goals

VINDEX3 serves **one fixed checkpoint efficiently under different inference and query policies**. Browse/LQL is in scope (§15); training is not. It is explicitly not:

- **A model-development store.** No optimisation for training, fine-tuning, gradient updates, adapter merging, or frequently rewritten weights. No copy-on-write component versioning. One extraction, then runtime policy.
- **A general neural-graph container.** VINDEX3 does not duplicate ONNX/safetensors-plus-compiler. The expert-programme vocabulary is deliberately bounded (§8.3).
- **A locality store.** Hot sets, retained experts, cache allocation, prefetch depth, local-versus-remote placement and reduced-top-K are **runtime metadata over the index**, never physical-format decisions (§9).

The supported contract is:

> Sparse decoder MoEs composed from routed and shared expert banks, optional pre/post transforms, declarative routing/reduction semantics, and a bounded expert-programme vocabulary — plus every dense model VINDEX2 supports, expressed as the degenerate single-entry case.

Genuinely novel expert topologies extend via a new `programme_id` (§8.4) without changing region storage.

---

## 3. Design principles

Principles 1–3 carry over from VINDEX2 §5.12; principle 1 is **amended** in VINDEX3.

1. **Structure is orthogonal to quantisation — now at region granularity.** VINDEX2 declared one `quant_format` per layer file, forbidding `gate/up = Q6_K, down = MXFP4` inside a layer. VINDEX3 moves the format tag to each weight region. Re-quantising one projection role is rewriting those regions (or adding a sibling segment), not replacing the layer.
2. **Unified for dense and MoE.** A dense layer is a bank with `num_entries = 1`. Binary format and dispatch path are identical.
3. **Native OS addressability.** Each segment file is independently mmap'd; expert sharding reads only assigned entry byte ranges; no offset arithmetic into a global blob.
4. **The split rule.** A component gets independent physical identity **only when LARQL may independently omit it, quantise it, place it, prefetch it, execute it — or query it.** Conceptual tensor taxonomy is not a reason to split. The query clause matters: WALK reads gate rows without up or down, so on a browse-enabled index the gate role has an independent access pattern by construction (§15.2), even if inference always fetches gate/up/down together.
5. **Storage aligns with dispatch.** The natural extent is the expert group matching the grouped kernel's dispatch width, so one grouped dispatch ≈ one extent ≈ one prefetch/read unit.
6. **Representable ≠ servable.** The format may describe combinations no kernel can yet execute. The capability registry (§10) distinguishes representable / reference-executable / dispatched / production, exactly mirroring the K3 ledger maturity discipline.
7. **Logical ownership by layer.** The logical layer remains the stable semantic unit. Segmentation (§7) is a physical storage parameter, invisible to model semantics.

---

## 4. The five durable weight classes

The serving ABI freezes exactly five classes. These are the boundaries that inference policy may ever want to fetch, place, quantise or omit independently:

| # | Class | Contents | Why independent |
| - | ----- | -------- | --------------- |
| 1 | **Control & router** | Embeddings, norms, LM head, router weights, routing metadata, recurrence/control parameters | Small, always resident, precision-sensitive |
| 2 | **Dense spine** | Attention / KDA / MLA projections, per layer or major projection class | Touched every token; future KDA3 target; independent quantisation ladder |
| 3 | **Shared FFN** | Shared experts and shared latent pre/post projections, per layer | Touched every token; different residency economics from routed |
| 4 | **Routed gate/up banks** | Per-layer expert-group extents | Candidate for exact Q6_K or native low-bit; grouped-dispatch aligned |
| 5 | **Routed down banks** | Per-layer expert-group extents | Independent quantisation, placement and (approximate-profile) omission policy |

Classes 4 and 5 remain physically separable **because their inference treatment can differ** (precision, bandwidth, kernel maturity, residency, remote/local placement) — not because a tensor taxonomy says so. Where a model's serving policy treats them identically, a single fused `gate_up_fused + down` bank per layer satisfies the ABI (§6.5).

No sixth class. Everything else — hot sets, expert retention, cache sizing, prefetch order, exact-vs-approximate selection — is profile/runtime metadata (§9).

---

## 5. Directory layout

```
model.vindex/
│
├── index.json                # SOLE ROOT AUTHORITY (§12): version, identity, provenance,
│                             # checksums, class map, segment lists, references to everything below
├── moe_manifest.json         # model + MoE programme description (§8), referenced from index.json
├── profiles/                 # execution profiles (§9), referenced from index.json
│   ├── exact.json
│   ├── attn-local-ffn-remote.json
│   └── ...
│
├── control/                  # class 1
│   ├── embeddings.bin
│   ├── norms.bin
│   ├── lm_head.bin           # omitted if tied
│   └── routers.bin
│
├── dense/                    # class 2
│   └── layer_{L}.weights     # LYRW v2, dense bank(s)
│
├── shared/                   # class 3
│   └── layer_{L}.weights     # LYRW v2, shared bank + latent transforms
│
├── routed/                   # classes 4 & 5
│   └── layer_{L}[.seg{S}].weights   # LYRW v2, routed bank, segmented as needed
│
├── query/                    # LQL metadata sidecars (§15.3) — metadata, not weights
│   ├── down_meta.bin         # DMET format unchanged from v1 §5.3
│   ├── feature_labels.json
│   └── relation_clusters.json
│
├── tokenizer.json
└── weight_manifest.json      # manifest-addressed tensors (control/dense), unchanged shape from v1 §5.9
```

Notes:

- Control-class and non-bank dense tensors (routers, recurrence parameters, latent projections when the adapter prefers manifest addressing) remain ordinary **manifest-addressed tensors** with `key / shape / kind / offset / length` — the v1 `weight_manifest.json` shape is retained unchanged.
- Component addressability does **not** require thousands of filesystem objects. A class may be one file or a few large ones; addressability comes from the region tables inside them.
- **One root, as in v1.** `index.json` remains the manifest of record — it owns version, identity, provenance, checksums (every physical file's SHA256, verified by `larql verify`), the segment lists, and references to `moe_manifest.json`, `weight_manifest.json` and `profiles/`. There is no `superblock.json`: a second root creates competing authorities (whose checksums win? whose version controls compatibility?). A detached signing/atomic-replacement wrapper is introduced only if a concrete need for it appears, as an addition around `index.json`, never a rival to it.
- Profiles are **not** covered by the immutable artifact checksum set — they are mutable policy. `index.json` records which profile names ship with the artifact; their contents are checksummed individually and replaceable.

---

## 6. LYRW v2 binary format

LYRW v2 preserves the v1 magic and self-describing property, and generalises the fixed four-integer offset table into banks, segments and entry-region tables.

### 6.1 Header

```
[header]
  magic:            u32   0x4C595257 ("LYRW")
  format_version:   u32   = 2
  logical_layer:    u32
  num_banks:        u16
  num_segments:     u16   (segments described by THIS file's tables; ≥1)
  flags:            u32   (bit 0: this file is one segment of a multi-segment layer)
  reserved:         u32
```

All integers little-endian. All region offsets are from the start of the containing segment file and 64-byte aligned.

### 6.2 Bank descriptor (`num_banks ×`)

```
  bank_id:              u16
  bank_kind:            u16   0=dense, 1=routed, 2=shared
  region_schema_count:  u16   number of schema records this bank owns in the
                              schema table (§6.4) — without it a reader cannot
                              tell where one bank's schemas end  [draft-2]
  flags:                u16   bit 0-1: browse mode (00=none, 01=direct,
                              10=strided) per §15.2; rest reserved  [draft-2]
  num_entries:          u32   1 (dense) or expert count
  input_dim:            u32
  intermediate_dim:     u32
  output_dim:           u32
```

Bank descriptor is 24 bytes (4-byte aligned), not the 20 the draft-1 field list implied.

`input_dim`/`output_dim` are the expert's own operand dims — for K3's latent bank these are 3584/3584, not the 7168 residual width. Dense v1-style layers map to one bank: `bank_kind=0, num_entries=1`.

**The binary carries no programme identity.** LYRW describes storage only — banks, entries, region schemas, offsets, formats. The MoE manifest binds `bank_id → programme` (§8.4). Two authorities for the same fact ("binary says programme 4, manifest says gpt-oss-expert-v1") is a disagreement waiting to happen; the manifest is the single one, consistent with the draft's own layering: storage holds regions, the manifest gives them meaning.

### 6.3 Segment descriptor (`num_segments ×`)

```
  bank_id:          u16
  segment_index:    u16
  first_entry:      u32
  entry_count:      u32
```

A single-file layer has one segment covering `[0, num_entries)`. Multi-segment layers repeat the header in every segment file with `flags` bit 0 set; `index.json` lists the segment files per logical layer so the loader never globs. **A segment file's entry table covers only that segment's entries** (`entry_count` rows, indexed from `first_entry`) — never the whole logical bank. [clarified in draft-2]

### 6.4 Region schemas and entry table

Expert banks are homogeneous: every entry in a bank shares the same region layout. The region schema is therefore declared **once per bank**, and each entry stores only offsets and lengths. (This is the simplification that dropping LYRW v1 binary compatibility buys — see §6.6.)

```
[bank region schemas]   region_schema_count × per bank:
  schema_index:     u16
  role:             u16   (§6.5)
  format:           u16   quant enum — 0=f32 1=f16 2=bf16 3=q4_0 4=q4_k 5=q6_k
                          6=q8_0 7=fp4_larql 8=mxfp4 9=nvfp4 10=mxfp8 ...
  packing:          u16   0=row_major, 1=blocks_with_scales_inline,
                          2=blocks_values / 3=blocks_scales
  pair_id:          u16   links a blocks_values schema to its blocks_scales
                          schema; 0xFFFF = unpaired
  reserved:         u16   pad — keeps the two u32 dims on a 4-byte boundary;
                          record is 20 bytes, not draft-1's 18  [draft-2]
  rows:             u32
  cols:             u32

[entry table]           entry_count × region_schema_count ×:
  offset:           u64   (from start of containing segment file, 64-B aligned)
  length:           u64
```

Consequences:

- `gate_up_fused: mxfp4` + `down: q6_k` in one file, and GPT-OSS-style separate value/scale regions, without a new container.
- Uniform expert geometry is explicit; parsing is O(schemas), not O(entries × regions).
- Per-expert codec variation — which no grouped kernel supports — is **unrepresentable**, by construction rather than by convention.
- `pair_id` makes values/scales pairing explicit; role tags alone are ambiguous once an entry carries more than one quantised tensor.
- Exceptional per-entry overrides are reserved behind a header flag bit, undefined in v2.0 — added only if a real model forces them.

### 6.5 Region roles

Registered roles (extensible; new roles do not bump `format_version`):

```
0  gate
1  up
2  gate_up_fused
3  down
4  bias
5  scales          (paired with a values region via packing=2/3)
6  latent_in       (shared pre-projection, when bank-local storage is preferred)
7  latent_out      (shared post-projection, likewise)
8..255   reserved-registered
256..    vendor/experimental
```

The fast-path contract is unchanged from v1: known kernels may **require** exactly `gate_up_fused + down` (or `gate + up + down`) and parse them into the same structures the grouped kernels use today. Presence of other roles does not invalidate a file; absence of a role a programme requires makes the file un-executable for that programme (§11), not invalid.

**Unknown role, format and packing tags are preserved, not rejected, at read time.** Refusal belongs at capability-check time (§11): a browse-only reader must not choke on a `down` region encoded in a codec it never touches, and a future codec must not invalidate old readers' ability to serve the regions they do understand. The reader reports unknown tags; the capability check refuses the *operations* that need them. [clarified in draft-2]

### 6.6 Relationship to the v1 layer files — greenfield, deliberately

LYRW v2 owes **no binary compatibility** to the §5.12 `layers/*.weights` files. Those files are an internal detail of VINDEX2: they exist only inside VINDEX2 directories, are parsed only by the VINDEX2 loader path, and were never a public contract in their own right. No external tool depends on their byte layout.

Consequences:

- **No synthesis adapter.** A LYRW v2 reader never opens a v1 layer file, and vice versa. Each container generation's loader reads its own layer format, end of story.
- **No in-place upgrade** of multi-hundred-GB indexes. Migration is `checkpoint → VINDEX3 extractor`, or optionally `VINDEX2 → VINDEX3 importer` — a standalone tool, not a loader feature.
- **Design freedom.** The bank-level region-schema table (§6.4), explicit value/scale pairing, and segment descriptors are all clean-sheet choices that a v1-compat shim would have contaminated. The `LYRW` magic and `format_version=2` are retained purely as self-description and forensics — a v1 reader that encounters a v2 file fails fast on the version field with a precise "requires VINDEX3 loader" error, never a parse error.

The compatibility obligation that **does** bind is one level up: larql must support VINDEX2 and VINDEX3 side by side (§12.1).

---

## 7. Segmentation

Motivating arithmetic (K3, exact Q6_K):

```
params per expert        = 3 × 3584 × 3072            = 33,030,144
params per routed layer  = 33,030,144 × 896           = 29,595,009,024
Q6_K bytes (210/256)     ≈ 24.28 GB  =  22.61 GiB
```

That exceeds the published 20 GiB shard cap, so `one logical layer = one physical file` cannot hold for K3 exact Q6_K. **Segment width and group width are two different scales, decided by two different measurements** — conflating them turns a 2-file layer into a 14-file layer for no read-path benefit:

| Scale | Optimises | Typical size |
| ----- | --------- | ------------ |
| **Segment file** | file count, mmap management, shard distribution, the 20 GiB cap | as large as the cap allows — for K3 exact Q6_K, **2 segments of 448 experts** (~11.3 GiB each), not 14 of 64 |
| **Group extent** (inside a segment) | SSD reads, prefetch units, grouped-kernel dispatch | 8/16/32 experts (E2/E3) |

A K3 routed layer therefore becomes:

```
routed/layer_037.seg00.weights   experts   0–447
  ├── group extent  0: experts   0– 15
  ├── group extent  1: experts  16– 31
  └── ... (28 extents of 16)
routed/layer_037.seg01.weights   experts 448–895
```

At ~92 MoE layers this is ~184 routed segment files, not ~1,288.

Rules:

- Segment boundaries **must** fall on group-extent boundaries; group width **must** divide segment width.
- Both widths are extraction-time storage parameters chosen by measurement (E2 sweeps them independently), not semantic commitments. They may differ per model and per layer.
- Physical expert order within a segment need not equal logical order — the entry table is the indirection. Permuted layouts are legal but must not be adopted without the E2/E6 evidence bar.

### 7.1 Group extents

The unit of read alignment and prefetch is the **group extent** inside a segment, sized to the grouped kernel's natural dispatch width. One grouped dispatch ≈ one group extent ≈ one read unit; the extent boundary is what the payload layout aligns to, and the entry table makes extents addressable without a separate structure. Individual-expert files are prohibited at K3 scale (896 experts × ~92 MoE layers × several roles is an operational failure, not a design).

---

## 8. The MoE programme manifest

`moe_manifest.json` describes how regions form an MoE computation. The physical index stores tensor regions; the manifest gives them meaning; the runtime selects an optimised kernel when it recognises the programme.

### 8.1 Per-layer shape

```json
{
  "moe_layer": {
    "layer": 12,
    "input_space": "residual",
    "router": {
      "scores": "layers.12.router.weight",
      "selection": { "kind": "top_k", "k": 16 },
      "normalisation": "k3_quantile_balanced"
    },
    "transforms": {
      "routed_input":  "layers.12.routed_expert_down_proj",
      "routed_output": "layers.12.routed_expert_up_proj"
    },
    "routed_bank": {
      "experts": 896,
      "programme": "latent-moe-v1",
      "storage": "routed/layer_012",
      "expert_dims": { "input": 3584, "intermediate": 3072, "output": 3584 }
    },
    "shared_bank": {
      "experts": 2,
      "programme": "gated-mlp-v1",
      "storage": "shared/layer_012"
    },
    "reduction": "gate_weighted_sum",
    "routed_output_norm": "layers.12.routed_out_norm",
    "combine": "residual_add"
  }
}
```

For a conventional MoE, `transforms` are null. For GPT-OSS, `routed_bank.programme = "gpt-oss-expert-v1"` and `shared_bank` is absent. For Inkling, shared and routed banks coexist in residual space. Per-layer variation (hybrid dense+MoE stacks, differing expert counts) is expressed by per-layer manifests, not global fields.

### 8.2 What stays model-specific (adapter-owned)

Router scoring/normalisation details, shared-expert participation in normalisation, activation functions, expert residual semantics, clamps/biases/scales, fusion preferences, layer-specific expert counts. The manifest names these; the adapter implements them.

### 8.3 Bounded programme vocabulary

The declarative vocabulary is inference-shaped and closed by design:

```
linear · fused linear · activation · clamp · multiply · add · scale ·
normalise · route · gather · weighted reduction · residual merge ·
pre/post transform
```

No general graph interpreter. Known arrangements compile to specialised kernels; a generic reference executor provides correctness for everything representable.

### 8.4 Programme registry

```
programme_id  0  gated-mlp-v1
              1  gated-mlp-fused-fc1-v1
              2  gpt-oss-expert-v1        (clamped gated MLP + residual term)
              3  shared-routed-mlp-v1
              4  latent-moe-v1            (K3 SiTU-GLU latent expert)
```

Each programme declares its **required region roles**. New programmes register an id, a version, required roles, and optional opaque model metadata — region storage is untouched.

The manifest is the **only** binding of `bank_id → programme_id`; LYRW files never carry programme identity (§6.2). Kernel capability entries (§10) reference programmes by registry id.

---

## 9. Execution profiles and authority

A profile is a small JSON file selecting inference behaviour over one extracted index. Profiles never trigger reslicing — and they never trigger conversion (§9.1).

```json
{
  "profile": "routed-mxfp4",
  "base": "exact",
  "select": {
    "routed.gate_up": "native-mxfp4",
    "routed.down":    "exact-q6k"
  },
  "placement": { "routed": "local", "dense": "local" },
  "runtime_policy": {
    "resident_experts": "routing-profile-2026-08-14.json",
    "prefetch_group": 32
  }
}
```

The profile carries **no `authority` claim of its own** — authority is derived (§9.2).

### 9.1 Representation variants — profiles select bytes, they don't request formats

A profile saying `"format": "mxfp4"` cannot turn Q6_K bytes into MXFP4 bytes by declaration. Exactly one representation model is legal: **a region set may carry multiple physically present variants; a profile selects a present variant.**

```json
{
  "region_set": "layer.12.routed.gate_up",
  "variants": {
    "exact-q6k":    { "storage": "routed/layer_012.q6k",    "fidelity": "source-equivalent" },
    "native-mxfp4": { "storage": "routed/layer_012.mxfp4",  "fidelity": "source-exact" }
  },
  "baseline": "exact-q6k"
}
```

- **Selecting an absent variant fails closed**, naming the region set, the requested variant and the variants actually present — before any byte is read.
- **No runtime conversion, ever.** "No hidden decode-time repacking" (§10) holds by construction: the bytes executed are the bytes stored.
- **Incremental packs.** New variants are added beside the baseline as independent, checksummed segment files — the multi-terabyte baseline is never rewritten. A routed-MXFP4 pack for K3 touches only routed region sets; attention, embeddings, routers and dense weights are untouched.
- **Single-copy, clarified.** The v1 principle forbids storing the *same* bytes twice; it does not forbid deliberate alternative encodings. The `baseline` variant is the canonical authority; additional variants are opt-in, per-component, and individually removable.

### 9.2 Authority — graded, derived, never asserted

**Levels** (mandatory, fail-closed):

| Level | Meaning |
| ----- | ------- |
| `source-exact` | Decoded values bit-identical to the source checkpoint, in the checkpoint's own encoding family (e.g. native MXFP4 regions of a native-MXFP4 model) |
| `source-equivalent` | Different encoding whose decode reproduces the source values exactly (e.g. a lossless Q6_K container of native MXFP4 values) |
| `numerically-approximate` | Same architecture, lossy representation (e.g. Q6_K quantised from BF16) |
| `structurally-approximate` | Components omitted or replaced (reduced top-K, shared-only layers, compiled subexperts) — must list `omitted_components` / `replacement` |
| `analysis-only` | Incapable of complete forward execution (router/browse slices) |

Authority is **derived, not declared**: every variant carries a region-level `fidelity` set at extraction time from provenance, and a profile's authority is the weakest fidelity across its active selections, further capped by programme traversal (§11) when required operands are absent. This closes the loophole where a lossy extraction becomes "exact" merely by being named the baseline — the baseline's own fidelity is recorded against the source checkpoint, not against itself. A profile cannot claim above its derived level; it may voluntarily claim below it.

Standard profile names: `exact`, `native-lowbit`, `mixed-precision`, `attn-local-ffn-remote`, `partial-residency`, `reduced-top-k`, `shared-only`, `router-browse`, `compact-approximate`.

**Runtime metadata, never format:** top-K/retention %, hot/warm/cold assignment, resident experts, per-layer popularity, adaptive cache size, prefetch ordering, exact-vs-approx selection, static per-layer precision choice.

### 9.3 Omission semantics ("dropping down")

The manifest distinguishes the materially different meanings:

| Mode | Authority | Notes |
| ---- | --------- | ----- |
| Client omission (FFN remote) | inherits selection (up to source-exact) | The whole routed branch moves; the K3 latent boundary makes whole-branch RPC ~14 KB/layer f16 vs ~100 KB for projection-split — never split gate/up local from down remote absent contrary measurement |
| Analysis/router slice | analysis-only | Retains routers, gate vectors, metadata; no decode claim |
| Cheaper down representation | numerically-approximate | The production interpretation of "cheap down" |
| Down replaced by compact approximation | structurally-approximate | Must name `replacement` |
| Routed branch skipped (shared-only) | structurally-approximate | Dropping an expert's `w2` alone yields **no** expert output — the honest mode is skipping the expert/branch, not a half-expert |

---

## 10. Kernel capability registry

Kernels advertise what they can execute:

```
programme_id · region roles · formats per role · grouping widths ·
input layout · maturity
```

Maturity ladder, matching the serving-format ledger: **Representable → Reference → Grouped → Dispatched → Production.** The loader reports, per (programme, format, grouping) combination, which rung it sits on. Mixed per-region formats are either supported by a kernel or **explicitly refused** — never silently repacked at decode time.

---

## 11. Capability checking

The loader does not hard-code "down weights present" tests. It traverses the layer's MoE programme and reports which required operands are absent, then:

- refuses execution profiles whose authority claim exceeds what the present operands support;
- names the missing role, bank, layer and segment precisely (`VindexError::MissingRequiredRegion { layer, bank, role, .. }`);
- distinguishes *representable-but-no-kernel* (falls back to reference executor, flagged) from *operand-absent* (hard refusal).

Programme-derived checks give the right per-architecture answers for free: routed removal on Inkling leaves shared experts contributing; on GPT-OSS it leaves no FFN contribution; on K3, a missing `routed_output` transform invalidates even completed expert computation.

---

## 12. Versioning and coexistence

Three version surfaces already exist; v2 adds nothing loosely named "vindex v2" in metadata. Precisely:

| Contract | v1 value | v2 value |
| -------- | -------- | -------- |
| LYRW `format_version` | 1 (VINDEX2-internal) | **2** (self-description only; no cross-reading, §6.6) — trails the container generation by one, permanently |
| `index.json` `version` | 2 | **3** — the container-generation discriminator |
| `vindex_spec_version` | 1 | **2** (programme manifest + profiles enter the validated public contract) |
| MoE manifest schema | — | **1** (new) |

**On the numbering.** The container generation *is* `index.json.version` — VINDEX2 is `version: 2`, VINDEX3 is `version: 3`. An earlier draft called the shipped generation "VINDEX1" while its `index.json.version` was already 2, putting a permanent off-by-one between the name and the sole discriminator. Both were renamed so the two agree. The LYRW layer format keeps its own sequence (v1 in VINDEX2, v2 in VINDEX3) and is deliberately not aligned: it is a different artifact with a different lifetime, and its numbering was already correct.

The FP4 additive-extension precedent is retained within each generation: new region formats, roles and programme ids are enum additions, not format bumps.

### 12.1 Dual-generation support in larql — the real compatibility contract

The binding obligation is not between the two on-disk formats (there is none — §6.6). It is that **one larql binary supports both vindex generations, indefinitely for reading and serving**:

- **Detection.** `index.json.version` is the sole **schema** discriminator, and the loader maps supported schema revisions to their owning container generation. No filename sniffing, no directory-shape heuristics. A missing or unknown version fails naming the version found and the schema sets this binary supports.

  The mapping is **many-to-one, not an identity**:

  | `index.json.version` | generation | note |
  | -------------------- | ---------- | ---- |
  | 1 | VINDEX2 | legacy schema; absent fields load with defaults |
  | 2 | VINDEX2 | what a fresh VINDEX2 extraction writes |
  | 3 | VINDEX3 | |

  A generation is *named* for the schema it currently writes, not for the only schema it can read. Treating the version as a generation identifier rather than a generation floor refuses every legacy-schema index in existence — which E0 caught in practice, not in review. Unified dispatch routes schema 1 to the VINDEX2 loader; the VINDEX3 loader still refuses it by name.
- **One entry point.** `Vindex::open(path)` returns the generation-appropriate handle behind a common trait; `larql run / serve / verify / slice / publish / pull` all accept either generation. Generation-specific verbs (e.g. profile selection) error precisely on a v1 index rather than silently no-op.
- **No cross-loading, no silent conversion.** The VINDEX2 loader path is frozen-but-maintained: it never opens VINDEX3 directories, never gains VINDEX3 features, and VINDEX3 code never re-implements VINDEX2 parsing. Conversion is only ever the explicit `VINDEX2 → VINDEX3` importer.
- **Hub and distribution.** `larql publish` stamps the container generation into the hub artifact metadata; `larql pull` selects the reader from that stamp and refuses a generation the installed binary lacks — before downloading terabytes, not after.
- **Wire protocols are generation-agnostic.** The expert-RPC and FFN-dispatch wire contracts carry activations and results, not container bytes; a grid may therefore mix v1 and v2 shards. A shard's container generation is a local concern of that shard's loader.
- **Support policy.** VINDEX2 remains fully supported for read/verify/serve/publish/pull. New extractions default to VINDEX3 once the ABI freezes **and** the E0 preservation matrix passes; v1 extraction remains available until then and is deprecated (not removed) after.

---

## 13. Conformance envelope

The ABI freezes only after all four fixtures pass the generic reference executor (fixtures defined in the experiments document):

| Capability | Direct | GPT-OSS | IS-276B | KL-48B | K3 |
| ---------- | :----: | :-----: | :-----: | :----: | :-: |
| Variable expert count / top-K | ✓ | ✓ | ✓ | ✓ | ✓ |
| Routed experts | ✓ | ✓ | ✓ | ✓ | ✓ |
| Shared experts | – | – | ✓ (2) | ✓ (1) | ✓ |
| Shared-sink router (shared experts scored) | – | – | ✓ | – | – |
| Residual-space experts | ✓ | ✓ | ✓ | ✓ | – |
| Latent-space experts | – | – | – | – | ✓ |
| Hybrid dense+MoE stack | – | – | ✓ (mid-stack, idx 2) | ✓ (layer 0) | ✓* |
| Non-softmax router (sigmoid + scaling) | – | – | ✓ (+ gate bias, norm_after_topk) | ✓ | – |
| Custom expert programme | – | ✓ | – | – | ✓ |
| Native low-bit regions | ✓ | MXFP4 | NVFP4/MXFP8 (real releases) | – (BF16) | MXFP4 |
| Mixed per-role format | ✓ | ✓ | ✓ (release convention: routed low-bit, rest BF16) | ✓ | ✓ |
| Fused/decomposed tensors | ✓ | ✓ | ✓ | ✓ | ✓ |
| Grouped dispatch | ✓ | ✓ | ✓ | ✓ | ✓ |
| Auxiliary optional components (MTP, towers) | – | – | ✓ | – | ✓ (multimodal) |
| Single-segment routed layer | ✓ | ✓ | ✓ (~4.9 GiB/layer Q6_K) | ✓ (~1.4 GiB/layer Q6_K) | – |
| Segmented logical layer | – | – | – | – | ✓ |
| Exceeds-RAM residency (partial/remote non-optional) | – | – | ✓ | – | ✓ |
| WALK/DESCRIBE (residual-space browse) | ✓ | ✓ | ✓ | ✓ | – |
| WALK via latent transform (§15.4) | – | – | – | – | ✓ |

\* K3's dense/MoE layer schedule is confirmed at adapter time; KL-48B's `first_k_dense_replace=1` is confirmed from the released config.

Order of real-model implementation: **Gemma MoE → GPT-OSS → Kimi-Linear-48B-A3B → Inkling-Small → K3.** GPT-OSS is the first practical target (small, official reference paths). Kimi-Linear proves shared-expert banks, the hybrid stack and the KDA/MLA dense spine on a RAM-resident checkpoint — the K3 adapter dress rehearsal. Inkling-Small then escalates on two axes at once: real NVFP4/MXFP8 native regions with the routed-low-bit/rest-BF16 mixed-precision release convention, and the first *forced* partial-residency/remote serving (it cannot be RAM-resident on the rig) — the K3 **serving** dress rehearsal, as KL-48B is the adapter one. Fixture C is retained purely as the tiny deterministic conformance fixture; it no longer stands in for anything. K3 is extracted **once**, last, into the frozen ABI.

---

## 14. What the experiments must decide

Only these decisions genuinely belong in the on-disk ABI; everything else stays runtime policy:

| Decision | Experiment | Why it matters |
| -------- | ---------- | -------------- |
| Region granularity (fused vs split roles) | E1, E4 | mmap count, rewriting, read amplification |
| Expert-group / segment width | E2, E3 | couples SSD reads to grouped kernels; K3 20 GiB cap |
| Fused vs decomposed FC1 storage | E1, E7, V2-1 | checkpoint import cost, mixed precision, **and gate-only browse reads** — the serving and query answers must be reconciled here, not assumed |
| Per-region format tags | structural (E4 gates *promotion* only) | representation is justified by native values/scales, v1's existing mixed precision, and format-neutral banks; E4 decides only whether a mixed-format **profile** reaches Production |
| Physical expert ordering | E2, E6 | possible locality gain vs model-specific assumption risk |
| Profile/variant-selection mechanism | E5, V2-0 | avoids reslicing per deployment; selection-not-request semantics (§9.1) |
| Capability/authority metadata | V2-0 | approximate slices must never present as exact |

Registered prior (falsifiable): one file-set per routed layer (two segments for K3 Q6_K), one entry per expert, down independently addressable, locality as runtime metadata, omission = skip-the-branch, remote = whole-routed-branch RPC. Per-region format tags are in the ABI **structurally** (not gated on E4); the registered prior is that no mixed-format *profile* reaches Production before real-K3-layer evidence (E4 stage 3). Gate/up fusion is **no longer a prior** — it is a per-index extraction choice decided by E1/E7 (§15.2).

---

## 15. Query layer — the model IS the database

The LQL browse surface (WALK, DESCRIBE, SELECT, EXPLAIN WALK) is a first-class consumer of VINDEX3, with the same single-copy contract as v1: **no query index is stored beside the weights; the weights are the query index.**

### 15.1 What replaces `gate_vectors.bin`

There is no `gate_vectors.bin` in v2. The gate rows live where the split rule puts them — as `gate` (or the gate half of `gate_up_fused`) regions inside LYRW banks. Gate KNN mmaps the segment files and walks gate regions in place:

- **f16/f32 regions:** zero-copy reinterpret, exactly the v1 fast path.
- **Block-quantised regions (FP4/FP8/Q-K):** lazy per-feature dequantisation at walk time via the existing block codecs — the v1 §5.10 mechanism, now applied to bank regions. The v1 §12.2 caveat carries over verbatim: 4-bit gate KNN is noisy; inference compensates, isolated dot products do not.
- Untouched `up`/`down` pages cost nothing under mmap, so browse over a full-fat index reads only gate bytes even when nothing was sliced.

MoE browse semantics are unchanged from v1: gate KNN selects features **across all experts, no router needed** — a bank with `num_entries = E` simply contributes `E × intermediate_dim` walkable features per layer. Feature numbering stays v1-flattened (`layer:feature`, experts contiguous within the layer) so `feature_labels.json` keys survive migration untouched.

### 15.2 The gate-addressability rule (resolves the fusion collision)

A browse-enabled index requires gate rows to be readable without decoding up. Two legal ways to satisfy that:

1. **Decomposed storage** (`gate` + `up` regions): clean gate-only reads; the E1/V2-1 fused-vs-decomposed parity requirement already guarantees kernels accept it.
2. **Fused storage with strided browse** (`gate_up_fused`): legal only when the packing permits striding into the gate half without decoding up rows (row-major f16 yes; interleaved quantised blocks generally no).

The choice is recorded per bank at extraction time (a `browse: none | direct | strided` tag in the bank descriptor's flags, matching §6.2's normative encoding). **Serving-only indexes may fuse freely.** A browse-enabled index defaults to decomposed unless E1/E7 shows the fused serving advantage exceeds its own promotion bar — the previous blanket "gate/up stay fused" prior is withdrawn.

### 15.3 Query metadata (`query/`)

`down_meta.bin` (DMET, unchanged), `feature_labels.json` and `relation_clusters.json` move to `query/`. These are **derived metadata, not weight copies** — single-copy is not violated. Two v2-specific notes:

- For latent MoE banks, `down_meta` is computed at extraction through the full output path — expert `w2` → `routed_output` transform → unembed — so its top-token claims describe residual-space effect, not raw latent columns.
- `query/` is optional per profile; its absence downgrades DESCRIBE/SELECT label richness, never WALK correctness.

### 15.4 Browsing latent-space banks (the genuinely new problem)

K3's gate rows live in the 3584-dim latent space; WALK queries originate in residual space. The programme manifest already carries what browse needs: `routed_input` names the residual→latent transform. WALK against a latent bank projects the query vector through that transform **once per query**, then dot-products against latent gate rows unchanged. `EXPLAIN WALK` reports the space hop. Residual-space banks (Direct, GPT-OSS, Inkling, all shared banks) walk exactly as v1.

### 15.5 Browse profiles and slices

- **Profile:** `browse` is a standard profile at authority `analysis-only` — requires gate regions (decodable), embeddings, tokenizer; `query/` and routers optional. Capability checking (§11) derives this; no filename tests.
- **Slice:** a published browse slice is produced by copying **only gate regions** into gate-only LYRW files (absent roles are legal, §6.5) plus `control/embeddings` + `query/`. The v1 ~3 GB browse economics are preserved; the loader reports the slice as `analysis-only` automatically because the programme's required inference operands are missing.

### 15.6 Extract-level mapping

| v1 extract level | v2 equivalent |
| ---------------- | ------------- |
| Browse | `browse` profile / gate-only slice (§15.5) |
| Inference | `exact` profile over classes 1–5 |
| All / COMPILE | full index — COMPILE reads regions to reconstruct safetensors, exactly as v1 read `gate_vectors.bin` |

---

## 16. Success criteria — "done" is defined here, in advance

VINDEX3 is a successful successor when all seven hold. Each is bound to the gate or experiment that proves it, so the bar cannot drift after the fact:

| # | Criterion | Proven by |
| - | --------- | --------- |
| 1 | An existing VINDEX2 model loads, verifies, serves and publishes through the dual-generation binary with zero behavioural regression | E0 (continuous, CI) |
| 2 | Gemma and GPT-OSS run through the same LYRW2 bank machinery and the same production dispatch interface | V2-3, V2-4 rungs 1–2 |
| 3 | Routed **and** shared banks are genuinely generic — proven on a shared-expert model or fixture, not asserted | Fixture C + **KL-48B (1 shared) and Inkling-Small (2 shared, sink router)** — real, V2-4 rungs 3–4 |
| 4 | K3 is extracted once and served with no K3-specific physical layout — only a manifest and an adapter | V2-4 rung 4 |
| 5 | A new representation or placement is introduced via variants, profiles and kernel capabilities, without rebuilding unrelated weights | §9.1 mechanism + V2-0 profile-resolution acceptance |
| 6 | Unsupported or approximate configurations fail closed and report exactly why — operand, bank, role, layer, segment, variant | V2-0, §11 |
| 7 | Onboarding the **next** conventional MoE requires an importer and a programme adapter — zero format changes, zero new region roles, zero kernel-interface changes | **E8 held-out architecture** |

Criterion 7 deserves emphasis: the four conformance fixtures cannot prove it, because the ABI was designed against them. Only a held-out architecture, onboarded after freeze under a no-format-changes rule, tests generalisation rather than fit. If E8 fails, the "portable sparse-serving substrate" claim is downgraded to "K3/GPT-OSS/Inkling serving format" — honestly, in this section.

The maturity ladder governs claims throughout: **Representable → Reference → Grouped → Dispatched → Production.** No criterion is met by a representable-only demonstration.

---

## License

Apache-2.0
