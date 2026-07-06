# ADR-010: Resource-Bounded Extraction via Pluggable WeightSource

**Status**: Accepted; implementation in progress (see #257, #258)
**Date**: 2026-06-15
**Context**: Default extraction (`--level inference` or higher) OOMs on constrained dev hosts — the whole model dequantizes to f32 in RAM before any weight is written (#166, measured: ~8 GB for a 2B model on a 6 GB/no-swap host). Only `--level browse` streamed; everything else assumed RAM was cheap.

## Decision

Extraction's peak RAM is bounded by making `write_model_weights` generic over a `WeightSource` trait (`format/weights/write_f32.rs`) rather than assuming an in-RAM model. Three implementations, one per source format, each with a different resource profile:

| Impl | Backing | Peak RAM per weight |
|---|---|---|
| `ModelWeights` | in-RAM | whole model (no bound — original, OOMs) |
| `StreamingWeights` | safetensors mmap | one tensor at a time |
| `GgufWeightSource` | GGUF mmap + per-tensor `ggml::dequantize` | one tensor at a time |

`GgufWeightSource` is the newest arm (#167): it dispatches through the same `write_model_weights_with_opts` call every other source uses, so the streaming writer itself doesn't need to know or care which format it's reading from — only the source's own accessor (`get_tensor`/`get_vector`) does the format-specific work, one tensor at a time, backed by the format's own mmap.

**This is deliberately split from CPU governance (#168 — OpenBLAS saturates all cores).** RAM-boundedness (this ADR) and CPU-boundedness are orthogonal resource constraints with independent fixes: the RAM arm is a data-flow/ownership change (streaming vs. materializing), the CPU arm is a runtime/threading configuration (`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`nice`, currently an env-var mitigation, not yet a code-level default). Do not conflate them into one fix — #166 is the umbrella bug report covering both, but the actual resolutions are two independent PRs/patterns.

**MVP scope, deliberate:** dense architectures only, `--quant none`. MoE GGUF is deferred (blocked on #153's fused-expert load — a different, unrelated gap); `--quant q4k` GGUF returns a clear, explicit error rather than silently falling back to the unbounded in-memory path.

## Implementation

```rust
// format/weights/write_f32.rs — the abstraction boundary
pub trait WeightSource {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)>;
    fn get_vector(&self, key: &str) -> Option<Vec<f32>>;
    fn arch(&self) -> &dyn ModelArchitecture;
    fn num_layers(&self) -> usize;
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)>;
    fn vector_names(&self) -> Vec<String>;
    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>>;
}

// extract/streaming/stages/model_weights.rs — dispatch, format-agnostic
if let Some(gguf) = self.tensor_source.gguf_source() {
    let src = GgufWeightSource { gguf, arch: &*self.arch, num_layers: self.num_layers };
    write_model_weights_with_opts(&src, self.output_dir, self.callbacks, level_opts)?;
    return Ok(());
}
// ... existing safetensors_mmap_refs() / ModelWeights branches, untouched
```

## Verification obligation (not yet fully discharged)

The decision's actual correctness claim — that this bounds peak RAM to roughly the embedding size rather than whole-model-f32 — requires a real measurement, not just a passing compile: peak RSS of a GGUF inference-level extract, CPU-capped, on a real dense model, recorded against the pre-fix in-memory estimate. As of this ADR, that measurement has not been taken (tracked directly on #167) — the architecture is implemented (#257, #258, both open/unmerged) but not yet empirically closed. Do not treat this ADR's existence as proof the OOM is fixed; treat #167's own tracked measurement as that proof once posted.
