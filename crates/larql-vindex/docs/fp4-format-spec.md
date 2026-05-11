# FP4 Vindex Format Specification

**Status:** Draft, pre-implementation. Pin before writing the
`larql-compute::quantisation` writer.
**Scope:** On-disk format for FP4/FP8-storage vindexes. Defines
`Fp4Config` (the JSON manifest block), per-projection file naming, byte
layout of FP4 and FP8 data, and the compliance sidecar.
**Companion document:** `FP4_PRECISION_POLICY.md` — decides which
projections get which precision. This spec records the format itself.
**Format version:** `fp4_format_version = 1`. Parent `VindexConfig.version`
remains at 2; FP4 is an additive opt-in, not a breaking bump.

---

## 1. Why a format spec before code

Format decisions that get baked into serialised data are expensive to
revise. An FP4 vindex shipped to HuggingFace cannot have its field names
renamed without a migration pass. The writer, reader, walk-kernel
dispatch, and extractor all dereference the same manifest — inconsistent
expectations during implementation are caught at format-review time or
not at all. This spec makes the manifest the source of truth.

## 2. Where the FP4 metadata lives

Inline in `index.json`, under a new optional top-level field:

```json
{
  "version": 2,
  "model": "google/gemma-3-4b-it",
  "dtype": "f16",
  "quant": "none",
  ...existing fields...
  "fp4": {
    "fp4_format_version": 1,
    "block_elements": 256,
    "sub_block_elements": 32,
    "sub_block_scale_dtype": "fp8_e4m3",
    "block_scale_dtype": "fp8_e4m3",
    "value_encoding": "fp4_e2m1_mxfp4_nibble_order",
    "projections": {
      "gate": { "precision": "fp4", "file": "gate_vectors_fp4.bin" },
      "up":   { "precision": "fp4", "file": "up_features_fp4.bin" },
      "down": { "precision": "fp8", "file": "down_features_fp8.bin" }
    },
    "compliance_gate": {
      "threshold_ratio": 16.0,
      "min_compliant_fraction": 0.99,
      "fallback_precision": "fp8"
    },
    "compliance_report": "fp4_compliance.json"
  }
}
```

**Rationale for inline (vs sidecar):** keeps one source of truth. Loaders
deserialise `VindexConfig` once; FP4 support is `if config.fp4.is_some()`
and dispatch from there. A separate file invites drift and requires a
second load path.

**Rationale for optional field:** old vindexes never have the `fp4`
key; they continue to work unchanged. Any loader that sees `fp4: null`
or missing uses the legacy gate/up/down path from `dtype`.

## 3. Projection precision values

Legal values for `projections.{gate|up|down}.precision`:

| Value  | Meaning                                      | File suffix                |
| ------ | -------------------------------------------- | -------------------------- |
| `fp4`  | MXFP4-style block-quantised                  | `_fp4.bin`                 |
| `fp8`  | FP8 E4M3 with per-block scale                | `_fp8.bin`                 |
| `f16`  | Bit-identical F16, standard layout           | *legacy filename (no suffix)* |
| `f32`  | Bit-identical F32                            | *legacy filename (no suffix)* |

Mixing precisions per-projection within one vindex is the point of the
format. Example layouts:

- **Option B default:** `{gate: fp4, up: fp4, down: fp8}` — writes
  `gate_vectors_fp4.bin`, `up_features_fp4.bin`, `down_features_fp8.bin`.
  No legacy `gate_vectors.bin` needed.
- **Option A override:** `{gate: fp4, up: fp4, down: fp4}` — writes all
  three as `_fp4.bin`.
- **Option C fallback:** `{gate: fp4, up: fp4, down: f16}` — writes
  `gate_vectors_fp4.bin`, `up_features_fp4.bin`, legacy
  `down_features.bin` (F16).
- **Extractor auto-downgrade:** `{gate: fp4, up: fp4, down: fp8}` (chosen
  because the Q1 scan showed down violated the compliance gate). The
  manifest records the actual on-disk state; the `compliance_report`
  sidecar records why.

Loaders never sniff filenames. They read the `file` field and dispatch on
`precision`.

## 4. Block geometry constants

```
sub_block_elements     = 32     # fixed, matches MXFP4 spec
block_elements         = 256    # § policy-doc decision; must divide hidden
sub_blocks_per_block   = 8      # = 256 / 32
blocks_per_feature_vec = hidden / 256
```

The format fixes `sub_block_elements = 32`. This is a hard constant
because the FP4 E2M1 encoding is defined over a 32-element group and
rewriting the encoder across group sizes is not a configurable knob.

`block_elements = 256` is the default and the only value the v1 writer
emits. Future format versions may vary this per-projection if
measurements find a case where a different block size pays off; the
field is already per-vindex configurable in the schema so that extension
does not require a new format version, only a new code path in the
reader.

**Validation constraint for v1:** `hidden % block_elements == 0`. A
vindex that violates this cannot be written in FP4 v1 format. The 4
models scanned in exp 26 (hidden ∈ {512, 1536, 2560, 5376}) all satisfy
this at 256.

## 5. FP4 layer data byte layout

For each layer's FP4 projection file (`gate_vectors_fp4.bin` etc.):

```
LAYER_0 | LAYER_1 | ... | LAYER_{L-1}
```

Layers are concatenated contiguously; per-layer offsets come from the
existing `layers[i].num_features` field (handles MoE / non-uniform
widths without format change).

For each layer, features are concatenated contiguously:

```
FEAT_0 | FEAT_1 | ... | FEAT_{N-1}
```

For each feature, blocks are concatenated:

```
BLOCK_0 | BLOCK_1 | ... | BLOCK_{B-1}      where B = hidden / 256
```

For each block (137 bytes total):

| Offset (bytes) | Size  | Contents                                       |
| -------------- | ----- | ---------------------------------------------- |
| 0–127          | 128 B | 256 FP4 values, 2 per byte (see §5.1)          |
| 128–135        | 8 B   | 8 FP8 E4M3 sub-block scales (one per 32-elem) |
| 136            | 1 B   | 1 FP8 E4M3 block scale                         |

**Cache rationale for interleaving scales with values:** the walk kernel
reads feature vectors one at a time. Keeping each feature's values and
scales in one contiguous 1370-byte (on 4B) region means one cacheline
prefetch walk per feature, not two. Scanning all features to build a
batch also stays sequential.

### 5.1 FP4 E2M1 nibble-pair encoding

Each byte stores two FP4 values. The lower nibble (bits 0–3) is the
**even-indexed** element of the pair; the upper nibble (bits 4–7) is
the **odd-indexed** element.

```
byte[i] = (fp4_value[2i+1] << 4) | (fp4_value[2i] & 0x0F)
```

FP4 E2M1 value format (4 bits = 1 sign + 2 exponent + 1 mantissa):

| Bits     | Meaning                                                   |
| -------- | --------------------------------------------------------- |
| 3        | Sign (0 = positive)                                       |
| 2–1      | Biased exponent (bias = 1)                                |
| 0        | Mantissa fraction                                         |

Representable values: `{±0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}`.
This encoding matches MXFP4 / Open Compute Project OCP-MXFP4 v1.0. Any
reader or writer that matches the canonical MXFP4 encoding table is
compliant; tests against reference vectors are in the §10 test plan.

### 5.2 FP8 sub-block scale

One FP8 E4M3 value per 32-element sub-block. E4M3 encoding (4 bits
exponent bias 7, 3 bits mantissa, 1 bit sign) matches the OCP FP8 spec.
The represented value is the per-sub-block scale such that

```
actual_value[i] = fp4_value[i] * sub_block_scale * block_scale
```

where `sub_block_scale` is the E4M3 value for the sub-block containing
element `i` and `block_scale` is the per-block scale (§5.3).

Sub-block scales are packed in order — byte 128 holds the scale for
sub-block 0 (elements 0..31), byte 129 for sub-block 1, …, byte 135 for
sub-block 7.

### 5.3 FP8 block scale

One FP8 E4M3 value per block. Stored at byte offset 136 of the block.
Combined with the sub-block scales as shown above. The block scale is
the coarse normaliser that lets the sub-block scales encode only the
*ratio* of one sub-block's magnitude to the block's maximum, which is
where the E4M3 dynamic range (needed < 16 by the DeepSeek condition) is
consumed.

## 6. FP8 layer data byte layout (down projection in Option B)

For each layer's FP8 projection file (`down_features_fp8.bin`):

Same outer structure as FP4 (layer → feature → block). Each block is
257 bytes:

| Offset (bytes) | Size  | Contents                           |
| -------------- | ----- | ---------------------------------- |
| 0–255          | 256 B | 256 FP8 E4M3 values                |
| 256            | 1 B   | 1 FP8 E4M3 block scale             |

No sub-block scales — FP8 E4M3 has sufficient dynamic range that
per-32-element scaling is unnecessary. The block scale still exists to
let the quantisation normalise per-block magnitude; this preserves most
of the E4M3 mantissa resolution on blocks that sit far from the
distribution mean.

Per-feature size: `blocks_per_feature_vec × 257` bytes. On 4B (hidden=2560,
B=10): 2,570 bytes per feature, matching the policy spec arithmetic.

## 7. Compliance sidecar

Filename: `fp4_compliance.json` (path recorded in `fp4.compliance_report`).
This is the verbatim output of `fp4_q1_scan` run at extract time, with
added extractor metadata:

```json
{
  "extracted_at": "2026-04-24T...",
  "extractor_version": "...",
  "scanner_version": "...",
  "block_elements_scanned": 256,
  "compliance_gate_threshold_ratio": 16.0,
  "compliance_gate_min_fraction": 0.99,
  "per_projection": [
    {"projection": "gate", "compliance_at_R16": 0.99999, "action": "wrote_fp4"},
    {"projection": "up",   "compliance_at_R16": 0.99999, "action": "wrote_fp4"},
    {"projection": "down", "compliance_at_R16": 0.99950, "action": "wrote_fp8_per_policy_default"}
  ],
  "full_scan": { /* embedded fp4_q1_scan.rs JSON output */ }
}
```

Valid values for `action`:
- `"wrote_fp4"` — projection satisfied the gate, FP4 file written.
- `"wrote_fp8_per_policy_default"` — policy specified FP8 for this
  projection regardless of compliance (Option B default on `down`).
- `"downgraded_fp4_to_fp8"` — policy specified FP4 but compliance gate
  failed; extractor wrote FP8 instead.
- `"downgraded_fp4_to_f16"` — compliance gate failed and fallback
  precision in `Fp4Config.compliance_gate.fallback_precision` was `f16`.
- `"user_override_f16"` — user forced F16 via extractor flag.

This field is advisory for humans; the manifest `projections.precision`
is authoritative for loaders.

## 8. Rust schema additions

New types in `larql-vindex::config::types`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Fp4,
    Fp8,
    F16,
    F32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionFormat {
    pub precision: Precision,
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceGate {
    pub threshold_ratio: f32,
    pub min_compliant_fraction: f32,
    pub fallback_precision: Precision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp4Config {
    pub fp4_format_version: u32,
    pub block_elements: u32,
    pub sub_block_elements: u32,
    pub sub_block_scale_dtype: String,   // "fp8_e4m3" for v1
    pub block_scale_dtype: String,       // "fp8_e4m3" for v1
    pub value_encoding: String,          // "fp4_e2m1_mxfp4_nibble_order" for v1
    pub projections: Projections,        // {gate, up, down}
    pub compliance_gate: ComplianceGate,
    pub compliance_report: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projections {
    pub gate: ProjectionFormat,
    pub up: ProjectionFormat,
    pub down: ProjectionFormat,
}

// Existing VindexConfig gains:
pub struct VindexConfig {
    // ...existing fields unchanged...
    #[serde(default)]
    pub fp4: Option<Fp4Config>,
}
```

## 9. Walk-kernel dispatch invariants

The walk kernel MUST:

1. Call `VindexConfig::fp4.as_ref()` once at load time.
2. If `Some(fp4)`, inspect each projection's `precision` tag and
   dispatch to one of {FP4 reader, FP8 reader, F16 reader, F32 reader}
   per projection.
3. Never sniff filenames to determine format.
4. Never assume all three projections share a precision.
5. Error out explicitly on unrecognised precision values (forward
   compatibility: an `fp6` tag written by a future writer must not be
   silently downgraded).

The walk kernel MAY:

1. Skip the FP4 path entirely if `fp4` is `None`, reading
   `gate_vectors.bin` etc. by the legacy F16/F32 path.
2. Cache dequantised feature vectors (optimisation decision; not a
   format concern).

## 10. Version and forward compatibility

- `VindexConfig.version` stays at 2. Adding the optional `fp4` field is
  not a breaking change; readers that ignore the field continue to work
  on legacy vindexes.
- `fp4.fp4_format_version = 1` is the FP4 data format version. Bump this
  to 2 when (and only when) the byte layout of blocks changes.
  Manifest-schema additions (new fields, new precision tags) do not bump
  this — they are introduced as optional fields with documented defaults.
- Adding a new precision variant (e.g. `fp6`) is a non-breaking change
  to the *schema* but requires a code path addition to every reader that
  wants to support it. Readers that don't support it should error
  explicitly rather than silently substituting.

## 11. Backward compatibility

- A vindex without the `fp4` field loads exactly as today.
- A vindex with `fp4` set but no `gate_vectors_fp4.bin` file is
  malformed and loaders MUST error. The policy spec's self-policing
  extractor will never produce such a vindex.
- Mixed legacy-and-FP4 vindexes (e.g. `fp4.down.precision = "f16"` using
  the legacy `down_features.bin`) are valid and supported. The `file`
  field in `ProjectionFormat` points to the actual file; loaders treat
  it as authoritative.

## 12. Tests (to be implemented alongside the writer)

Reference-vector tests at the codec level:

- Round-trip: random f32 data → FP4-encode → FP4-decode → compare to
  expected quantised values (deterministic given the encoding).
- Canonical MXFP4 test vectors from the OCP spec.
- FP4 E2M1 sign/zero/denormal edge cases.
- FP8 E4M3 round-trip.

**Required format-level test — the round-trip invariant.** Must ship
with the writer and reader, independent of the walk kernel. This is the
isolation boundary: if Q2 produces unexpected logit divergence, the
round-trip test answers "is it a format bug?" in seconds rather than
hours.

- Take a synthetic feature vector with a known scale distribution (e.g.
  Gaussian, uniform, and a deliberately pathological
  max/min-scale-ratio case).
- Write it through the FP4 path (full block encoding including both
  scale levels).
- Read it back through the FP4 path.
- Assert the reconstruction matches the source within FP4's
  per-sub-block representable quantisation bound — i.e., each element's
  absolute error ≤ the smallest representable step at that block's
  effective scale. Not a cosine threshold, a bound derived from the
  format itself.

The same invariant shipped for FP8 blocks against E4M3's representable
step.

Format-level tests:

- Write a small vindex (one layer, a few features), reload, assert
  per-byte identical to a pinned hex reference.
- Non-uniform layer widths (mirrors Gemma 4 E2B's mixed 6144/12288
  layout).
- Mixed-precision manifest (`{gate: fp4, up: fp4, down: fp8}`) and
  cross-projection file independence.

End-to-end tests (blocked on walk-kernel hookup, tracked in the build
plan, not this spec):

- FP4-stored gate + FP16 rest vs baseline F16 walk: measure logit KL.
- Full Option B vs baseline F16: Q2 sanity.

## 13. Non-goals for v1

- **Streaming writer.** v1 writer can hold a layer in RAM. Streaming is
  a later optimisation.
- **Partial-precision upgrades.** No support for "the first 10 layers in
  FP4, the rest in F16" within one projection. Precision is per-whole-
  projection for this version.
- **Compressed sub-block scales.** E4M3 sub-block scales are 1 byte
  each. Tighter encodings (4-bit scales, delta-encoded scales) are
  possible but not worth the complexity until there is a demonstrated
  bandwidth bottleneck.
- **GPU-friendly layouts.** The interleaved layout is tuned for the M3
  Max demand-paged walk kernel, not for hardware with coalesced-load
  constraints (NVIDIA warps). If LARQL grows a GPU walk backend, a
  different physical layout can be added as `fp4_format_version = 2`.

## 14. Open items before writer lands

These are small and should be resolved during writer implementation,
logged here so nothing slips:

1. **Endianness of FP8 and byte-order within nibbles.** Little-endian on
   byte values is standard; nibble order within a byte is specified in
   §5.1. Confirm the MXFP4 reference-vector tests match this choice; the
   OCP spec is ambiguous on a couple of corner cases.
2. **NaN/Inf handling in source data.** Extractor should error on
   non-finite input; FP4 E2M1 has no NaN representation.
3. **Denormal FP8 block scales.** E4M3 permits denormals; confirm the
   decoder handles them as expected.
4. **File trailer for checksumming.** Propose appending a SHA-256 of the
   file contents as a trailing 32 bytes, like other vindex binaries.
   This requires keeping the walk kernel from reading those bytes as
   data — handle by storing `file_size - 32` as the data extent in the
   manifest.

## 15. Artefacts this spec depends on

- `FP4_PRECISION_POLICY.md` — Option B recommendation and `block_elements
  = 256` derivation.
- `results.md` — Q1 compliance numbers justifying the defaults.
- `results/q1_gemma3_4b.json` — reference compliance data; format of
  the `full_scan` field in the compliance sidecar.
- `crates/larql-vindex/examples/fp4_q1_scan.rs` — to be promoted to a
  library entry in `larql-vindex::quant::scan` called from the
  extractor's self-policing step.
