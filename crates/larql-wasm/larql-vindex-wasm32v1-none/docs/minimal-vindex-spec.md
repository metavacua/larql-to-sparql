# Minimal-Viable Vindex (MVV) — formal specification

**Status:** v0.1 (reference implementation: `src/index/mvv/`).
**Relationship to the full format:** the MVV is a *strict minimal subset* of
`crates/larql-vindex/docs/format-spec.md` v0.4 (the canonical superset, which
also covers FP4/Q4K, MoE expert weights, attention, lm_head, per-layer LYRW
weights, model-config). Anything not named here is **out of MVV** and lives only
in the full format.

The MVV defines two things and the contract binding them:

1. a **format** — an invariant byte/in-memory layout for gate vectors + their
   descriptor + (optional) metadata; and
2. a **query kernel** — a *total* (panic-free, UB-free) gate-KNN over that
   format, certifiable into the strict wasm32v1-none floor of the dialect
   lattice.

Effectiveness before efficiency: the format is invariant and the **L0 scalar
kernel is the reference**; optimized kernels (matrixmultiply / simd128 / BLAS)
are a later, dialect-stratified efficiency layer that must meet the *same*
semantics + totality contract (§5).

---

## 1. Format

### 1.1 Gate-vector blob
A single little-endian byte buffer, gate matrices concatenated layer-by-layer,
no per-record framing:

```
[ layer 0 : num_features₀ × hidden_size elements ]
[ layer 1 : num_features₁ × hidden_size elements ]
...
```

Each element is `f32` (4 bytes) or `f16` (2 bytes) per the descriptor `dtype`.
f16→f32 decode is IEEE half precision (`larql_models::quant::half::decode_f16`).
A layer row is one FFN feature's gate vector of length `hidden_size`.

### 1.2 Descriptor (`index.json`) — the source of truth
The canonical descriptor is `index.json` (a superset `VindexConfig` v2). The MVV
reads only these **minimal required fields** (all others are ignored):

| field | type | meaning |
|---|---|---|
| `version` | u32 | descriptor version; MVV requires **2** |
| `num_layers` | usize | number of layers |
| `hidden_size` | usize | gate-vector dimension |
| `dtype` | `"f32"`\|`"f16"` | element encoding |
| `layers[]` | array | per-layer records |
| `layers[].layer` | usize | layer index |
| `layers[].num_features` | usize | feature (row) count for the layer |
| `layers[].offset` | u64 | byte offset into the blob |
| `layers[].length` | u64 | byte length of the layer's matrix |

A layer's float span is `floats[off/bpf .. off/bpf + num_features*hidden_size]`
where `bpf = bytes_per_float(dtype)` and `off = layers[i].offset`.

### 1.3 Metadata (optional)
`feature_meta(layer, feature)` returns a `FeatureMeta { top_token,
top_token_id, c_score, top_k[] }` when present. Metadata is *optional*: an MVV
index with no metadata answers `None` for every feature. (Binary `DMET` v1 or
the legacy NDJSON encodings from the full format may populate it; the minimal
form accepts pre-parsed `meta[layer][feature]`.)

### 1.4 Validation invariants (the totality basis)
A reader MUST reject a descriptor/blob pair unless **all** hold (each maps to a
typed `MvvError`):
- `version == 2` — else `UnsupportedVersion`;
- `layers.len() == num_layers` — else `LayerCountMismatch`;
- for every layer, `length == num_features * hidden_size * bpf` (overflow-checked)
  — else `LayerLengthMismatch`;
- `length % bpf == 0` — else `Misaligned`;
- `offset + length <= blob_len` (overflow-checked) — else `BlobLengthMismatch`;
- the layers **exactly cover** the blob: `max(offset+length) == blob_len` (a
  minimal blob has no trailing slack) — else `BlobLengthMismatch`.

After validation, every per-layer byte range is in-bounds and dimensionally
consistent — the precondition the kernel relies on to stay total without
`unsafe`.

### 1.5 Optional self-describing header
`index.json` is the **source of truth**. As a *redundant* self-validation aid,
a blob MAY carry a small leading header (`magic` + `version` + `dtype` +
`num_layers` + `hidden_size` [+ per-layer counts]). **Spec invariant:** the
header MUST agree field-for-field with `index.json`; a reader that is given both
validates `header ⟺ index.json` and returns `HeaderMismatch` on any
disagreement. The header never overrides index.json; its correctness is
*defined as* agreement with it.

---

## 2. Query kernel — semantics

The query core is the **intersection** of the dense and MoE/trace surfaces:

| fn | signature | semantics |
|---|---|---|
| `num_features` | `(layer) -> usize` | feature count; **0** if `layer` out of range |
| `feature_meta` | `(layer, feature) -> Option<&FeatureMeta>` | metadata or `None` |
| `gate_knn` | `(layer, residual: &[f32], top_k) -> Result<Vec<(usize,f32)>, MvvError>` | top-K features by \|gate·residual\|, descending |

Optional extensions (same contract):
- `gate_knn_expert(layer, residual, feat_start, feat_end, top_k)` — top-K within
  a feature range (MoE expert / feature-range scope);
- `walk(residual, layers, top_k)` — multi-layer trace (composes `gate_knn` +
  `feature_meta`); *not yet in the reference*.

Ranking is by **descending absolute dot product**; ties and `NaN` are ordered by
`f32::total_cmp` of the magnitude (deterministic, never panics).

---

## 3. Totality contract

Every core/extension function returns a **defined value or a typed `MvvError`
on every input** — no panic, no UB. Concretely the reference (and any conforming
impl) MUST observe:

- **no `unsafe` / `from_raw_parts`** — decode via checked `chunks_exact` +
  `from_le_bytes` (alignment-independent); all slicing via `.get(..)`;
- **no `.unwrap()` / `.expect()`** — bad layer/feature/range, `residual.len() !=
  hidden_size`, short/misaligned blob, malformed `index.json` → typed error;
- **`NaN` policy** — `total_cmp` on the magnitude; `NaN` scores are ordered
  deterministically, never panic the sort;
- **bounded loops, no recursion, no locks** — straight-line + bounded iteration
  over the blob (single-threaded wasm).

`MvvError` variants: `BadIndexJson`, `UnsupportedVersion`, `LayerCountMismatch`,
`LayerLengthMismatch`, `Misaligned`, `BlobLengthMismatch`, `LayerOutOfRange`,
`DimMismatch`, `FeatureRange`, `HeaderMismatch`.

---

## 4. Certification contract

The reference kernel MUST be a member of the intersection of three independent,
**statically checkable** strata (the lattice from the project's WASM-subset
analysis):

- **WASM-valid** — a valid wasm module;
- **wasm-safe** — *no non-intrinsic host imports* and *no `call_indirect`* over
  the export-reachable closure (capability-contained + no arbitrary-code-
  execution ⇒ a fully static call graph);
- **Q-classified / total** — bounded, recursion-free, `unsafe`-free, lock-free
  ⇒ behaviour is tractably analyzable and "constructively provably bug-free on
  all inputs" is reduced to the §3 totality contract plus the static checks.

Target stratum: the **floor** — `wasm32v1-none` MVP, scalar, zero imports.
Placement is confirmed empirically by §6.

---

## 5. Dialect-stratified efficiency ladder

The **format is invariant**; the kernel implementation is selected per lattice
point, and **every level satisfies the identical §2 semantics + §3 totality
contract** (only dialect/performance differ):

| level | kernel | dialect |
|---|---|---|
| **L0** (reference) | pure scalar checked matvec, zero deps | wasm32v1-none MVP (the floor) |
| L1 | `matrixmultiply` (no_std, scalar kernels) | wasm32v1-none MVP |
| L2 | `+simd128` (`core::arch::wasm32` / matrixmultiply / nalgebra) | browser SIMD dialect |
| L3 (native) | existing `ndarray` + BLAS / faer | native |

A library/SIMD backend is correct *only at its level*; it must never appear in
L0 (it would defeat the minimality + certification of the floor).

---

## 6. Conformance & verification

- **Conformance suite** (`tests/mvv_conformance.rs`): one battery run against
  every impl, identical results, including **adversarial/malformed inputs**
  (truncated blob, trailing slack, wrong dtype length, OOB layer/feature, dim
  mismatch, `NaN` residual, `header ⟺ index.json` mismatch) each asserting a
  typed error and **no panic**. This is the totality proof harness.
- **Lattice placement:** `cargo check` on both targets; `census.py --inventory`
  (the MVV query surface in SET B); the minimal wasm-safe certifier (import-free
  + `call_indirect`-free over the export closure, retargeted to wasm32v1-none).
- **Native equivalence (optional):** the L0 result equals the L3 (ndarray/BLAS)
  result on identical data — a native-only cross-check.

---

## 7. Versioning

- MVV descriptor version pins to `VindexConfig` **v2**.
- This spec is **v0.1**; it tracks the reference in `src/index/mvv/` and the
  canonical superset `format-spec.md` v0.4. Breaking changes bump the spec minor
  and, if the on-disk descriptor changes, the descriptor `version`.
