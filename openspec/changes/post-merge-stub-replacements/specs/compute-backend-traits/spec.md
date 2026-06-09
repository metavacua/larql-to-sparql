## ADDED Requirements

### Requirement: DecodeBackend exposes head-replacement intervention hook

The `larql_compute::DecodeBackend` trait SHALL define
`full_pipeline_q4_with_head_replacement(layers, x, hidden, inter,
seq_len, use_qk_norm, softcap, target_layer, target_head,
replacement_delta) -> Option<Vec<f32>>`. This method is the GPU-side
intervention hook used by the `dev ov_rd` circuit-analysis CLI: at
`target_layer` the dispatcher zeros head `target_head` and adds
`replacement_delta` (a `[seq_len * head_dim]` f32 slice) in its
place; the remaining layers see the intervened residual stream.

Backends without the intervention kernel (CPU, CUDA on the current
fork) SHALL provide the default impl returning `None`. The MetalBackend
inherent implementation supplies the real kernel under
`#[cfg(all(feature = "metal", target_os = "macos"))]`.

#### Scenario: CPU returns None (default impl)
- **WHEN** a CPU `ComputeBackend` is asked to run
  `full_pipeline_q4_with_head_replacement(&[], &[], 16, 32, 1, false,
  0.0, 0, 0, &[])`
- **THEN** the call SHALL return `None`.
<!-- test: larql_compute::backend::decode::tests::cpu_full_pipeline_q4_with_head_replacement_returns_none -->

### Requirement: DecodeBackend exposes pre-W_O capture hook

The `larql_compute::DecodeBackend` trait SHALL define
`full_pipeline_q4_capture_pre_wo(layers, x, hidden, inter, seq_len,
use_qk_norm, softcap, target_layer, target_head) -> Option<Vec<f32>>`.
This hook captures head `target_head`'s pre-W_O output (i.e. the
head's contribution to the residual stream before the attention
output projection mixes heads together) at `target_layer`, then
short-circuits the forward pass. The returned vec is
`[seq_len * head_dim]` f32.

Backends without the capture kernel SHALL return `None`. The
MetalBackend inherent implementation supplies the real kernel
under the same Metal cfg gate.

#### Scenario: CPU returns None (default impl)
- **WHEN** a CPU `ComputeBackend` is asked to run
  `full_pipeline_q4_capture_pre_wo(&[], &[], 16, 32, 1, false, 0.0,
  0, 0)`
- **THEN** the call SHALL return `None`.
<!-- test: larql_compute::backend::decode::tests::cpu_full_pipeline_q4_capture_pre_wo_returns_none -->
