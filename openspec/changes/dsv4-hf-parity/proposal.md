## Why

DSv4-Flash inference (the whole `dsv4_*` stack) is validated today only
against **internal** references: CPU-vs-CUDA greedy-token equality, and
quant-vs-f32 tolerance (the archived `dsv4-quant-residency` change).
Those catch *correlated drift* between our own paths but not a shared
error — if the forward has a systematic bug (a wrong norm, a transposed
projection, a mis-scaled RoPE) that both CPU and CUDA reproduce, every
internal test stays green. Tasks #63/#69 (the long-outstanding "HF
parity test on real DSv4 GGUF") close that hole: pin the forward to an
**external authority** — HuggingFace `transformers` running the
reference DeepSeek-V4-Flash — on a fixed prompt.

The blocker has always been that generating the reference needs the HF
weights + a `transformers` env, which can't run inside a Rust test. The
fix is to decouple the two: a small Python generator dumps reference
logits to a versioned JSON once; a Rust test consumes that dump and
compares, skipping cleanly when the dump is absent (like the existing
real-GGUF `#[ignore]` tests). This lands the harness now; the dump is
sourced out-of-band.

## What Changes

- A reference-dump format (JSON): the HF-tokenized prompt `token_ids`
  plus the final-position next-token **top-K** `(token_id, logit)` from
  the HF reference. Small, committable, human-readable.
- A Python generator (`scripts/dsv4_hf_reference.py`) that loads the HF
  DeepSeek-V4-Flash model via `transformers`, runs the fixed prompt, and
  writes the dump. Documented; run out-of-band by a maintainer.
- A Rust parity test (`#[ignore]`, real-GGUF) that loads the dump, runs
  the DSv4 GGUF forward on the dump's `token_ids`, and asserts: greedy
  next-token matches the reference top-1; the top-K sets overlap; and
  top-1 logit agrees within a documented (generous) tolerance — the
  GGUF is Q4_K-quantized vs the HF f16/bf16 reference, so value
  agreement is loose but argmax is stable for non-degenerate gaps.
- The test skips (not fails) when no dump is present, so it's green-by-
  skip in CI and runs only where the dump + GGUF exist.

## Capabilities

### New Capabilities
- `dsv4-hf-parity`: DSv4-Flash GGUF forward logits are pinned to an
  external HuggingFace `transformers` reference on a fixed prompt —
  greedy-token match + top-K overlap + tolerance-bounded top-1 logit —
  via a decoupled reference-dump format consumed by an `#[ignore]`d
  Rust test, generated out-of-band by a Python script.

### Modified Capabilities
None. This adds an external-authority correctness gate; it does not
change the forward, the storage, or any existing capability's behaviour.

## Impact

- **Code**: new `scripts/dsv4_hf_reference.py`; new Rust parity test
  (`crates/larql-inference/tests/test_dsv4_hf_parity.rs` or a `lib`
  test); a small reference-dump loader. No production-path change.
- **Dependencies**: the dump generation needs HF `transformers` + the
  DeepSeek-V4-Flash weights (out-of-band, maintainer-run). The Rust
  side needs only `serde_json` (already in the tree).
- **Numerics**: GGUF Q4_K vs HF f16/bf16 → logit values differ by
  quantization error; the test's primary assertion is greedy-token +
  top-K-set, with a loose documented logit tolerance.
- **No external API change**; on-disk GGUF format unchanged.
