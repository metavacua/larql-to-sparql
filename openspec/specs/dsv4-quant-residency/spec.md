# dsv4-quant-residency Specification

## Purpose
TBD - created by archiving change dsv4-quant-residency. Update Purpose after archive.
## Requirements
### Requirement: Dual-representation layer storage

DSv4 layer and FFN weight storage SHALL hold each large matmul weight
as an optional resident `QuantTensor` alongside the existing f32
array, where exactly one representation is populated per tensor. The
f32 array SHALL remain the fallback so that any tensor whose GGUF
format is not supported by the quantized matmul path still loads and
runs.

#### Scenario: Quantized tensor populates the QuantTensor field

- **WHEN** a layer weight is read from a GGUF tensor whose type is a
  format the lazy-quant path supports (Q4_K, Q5_K, Q6_K, Q8_0)
- **THEN** storage SHALL hold it as a `Some(QuantTensor)` over the raw
  quantized bytes, and the corresponding f32 array SHALL be empty
  (0×0)
<!-- test: larql_inference::attention::dsv4_storage_build::build_layer_storage_resident_populates_quant_and_empties_f32 -->

#### Scenario: Indexer Q-up held resident on indexer layers

- **WHEN** an Indexer-variant layer (`compress_ratio == 4`) is loaded
  resident and its `indexer.attn_q_b` is a supported quantized format
- **THEN** the indexer `wq_b` SHALL be held as a `Some(QuantTensor)`
  with its f32 array empty (0×0), while the indexer's `wproj` remains
  f32, and the indexer scoring SHALL dispatch the `wq_b` matmul through
  the lazy-quant path
<!-- test: larql_inference::attention::dsv4_storage_build::build_layer_storage_resident_hca_weights_quantized -->

#### Scenario: HCA compressor wkv/wgate held resident

- **WHEN** an HCA layer (Compress or Indexer variant) is loaded resident
  and its compressor `wkv`/`wgate` (`attn_compress_kv/gate` and, for
  Indexer layers, `indexer.compress_kv/gate`) are a supported quantized
  format
- **THEN** both projections SHALL be held as `Some(QuantTensor)` with
  their f32 arrays empty (0×0), the compressor `ape`/`norm` SHALL remain
  f32, and the compressor SHALL dispatch the `wkv`/`wgate` matmuls
  through the lazy-quant path
<!-- test: larql_inference::attention::dsv4_storage_build::build_layer_storage_resident_hca_weights_quantized -->

#### Scenario: Unsupported format falls back to f32

- **WHEN** a layer weight's GGUF tensor type is not supported by the
  quantized matmul path (e.g. an exotic or F32-native tensor)
- **THEN** storage SHALL dequantize it once into the f32 array and
  leave the `QuantTensor` field `None`

### Requirement: Quantized weights are not eagerly dequantized

The DSv4 GGUF loader SHALL build resident `QuantTensor`s directly
from GGUF tensor bytes (via `QuantTensor::from_raw`) for the large
matmul weights, without producing a full f32 copy of those tensors at
load time. Building the resident weight set for DSv4-Flash SHALL fit
in the quantized footprint (~161 GB) rather than the f32 footprint
(~1.1 TB).

#### Scenario: Loader keeps Q4_K bytes quantized

- **WHEN** the loader reads a Q4_K FFN expert tensor from the GGUF
- **THEN** it SHALL retain the Q4_K bytes in a `QuantTensor` and SHALL
  NOT allocate the dequantized f32 expansion of that tensor
<!-- test: larql_inference::attention::dsv4_gguf_reader::real_gguf_resident_expert_footprint -->

#### Scenario: Resident footprint fits the quantized size

- **WHEN** all layers of DSv4-Flash are loaded resident
- **THEN** the total weight memory SHALL be on the order of the
  quantized GGUF size, not the f32 expansion
<!-- test: larql_inference::attention::dsv4_gguf_reader::real_gguf_resident_expert_footprint -->

### Requirement: Quant-aware forward dispatch

Each DSv4 matmul site SHALL dispatch on the weight representation:
when a `QuantTensor` is present it SHALL compute via the lazy-quant
`matvec` (single token) or `matmul` (batch) path; otherwise it SHALL
use the existing f32 `dot_proj_gpu` path. The dispatch SHALL preserve
the existing `Option<&dyn ComputeBackend>` threading so the f32 path
is unchanged when no quantized weight is present.

#### Scenario: Quantized weight uses the lazy-quant path

- **WHEN** a per-layer projection has a resident `QuantTensor` weight
  and receives an input activation
- **THEN** the projection SHALL be computed by the `QuantTensor`
  matvec/matmul kernel without materializing the full f32 weight

#### Scenario: f32 fallback weight uses the existing path

- **WHEN** a per-layer projection's weight is f32-only (no
  `QuantTensor`)
- **THEN** the projection SHALL be computed by the existing
  `dot_proj_gpu(&x, &w, backend)` path, identical to today

#### Scenario: MoE experts read quantized slices without re-dequant

- **WHEN** the routed-MoE dispatch processes a token's selected
  experts whose weights are resident `QuantTensor`s
- **THEN** each expert's gate/up/down SHALL be obtained as a
  zero-copy quantized slice (`QuantTensor::expert_slice`) and run
  through the lazy-quant matmul, with no per-expert f32 dequant
<!-- test: larql_inference::attention::dsv4_moe_dispatch::quant_moe_dispatch_matches_f32_within_tolerance -->

### Requirement: Resident (non-streaming) model forward

DSv4 SHALL provide a forward path that loads all layers' resident
`QuantTensor` weights once and reuses them across every decode step,
rather than reloading and re-dequantizing each layer from the GGUF
per token. The existing streaming forward SHALL remain available for
the case where the quantized weight set exceeds host RAM.

#### Scenario: Weights loaded once for multi-step decode

- **WHEN** a multi-step decode runs against the resident forward
- **THEN** each layer's weights SHALL be loaded from the GGUF at most
  once for the whole decode, not once per token
<!-- test: larql_inference::attention::dsv4_streaming_model_forward::resident_forward_matches_streaming_forward -->

#### Scenario: Streaming path retained for oversized models

- **WHEN** the quantized weight set does not fit in host RAM
- **THEN** the streaming load-forward-drop path SHALL still be usable
  and unchanged

### Requirement: Quantized forward numerical parity

The quant-resident forward SHALL produce results within a documented
tolerance of the f32-dequant forward — not bit-identical, because the
per-row quantized dot accumulates in a different order than f32
dequant + BLAS. Greedy decode token sequences SHALL match between the
two paths for non-degenerate logit gaps.

#### Scenario: Logits within tolerance

- **WHEN** the same prompt is run through the quant-resident forward
  and the f32 forward
- **THEN** the per-position logits SHALL agree within the documented
  relative tolerance
<!-- test: larql_inference::attention::dsv4_streaming_model_forward::resident_quant_forward_within_tolerance_of_streaming_f32 -->

#### Scenario: Greedy tokens match

- **WHEN** greedy decoding the same prompt through both forwards
- **THEN** the generated token sequences SHALL be identical (argmax is
  stable across the tolerance gap for non-degenerate cases)
<!-- test: larql_inference::attention::dsv4_streaming_model_forward::resident_quant_forward_within_tolerance_of_streaming_f32 -->

### Requirement: CPU-FFN / GPU-attention hybrid placement

DSv4 SHALL support running the FFN/MoE matmuls on CPU against the
resident quantized weights while offloading the attention matmuls to
a GPU `ComputeBackend`, within a single process. The placement SHALL
be selected per matmul site (attention sites receive the GPU backend;
FFN/MoE sites receive `None` / CPU).

#### Scenario: Attention on GPU, FFN on CPU

- **WHEN** a decode step runs with a CUDA backend present and hybrid
  placement enabled
- **THEN** attention projections SHALL dispatch to the GPU backend and
  FFN/MoE expert matmuls SHALL run on CPU against resident quantized
  weights
<!-- test: larql_inference::attention::dsv4_generate::dsv4_bench_cpu_vs_cuda -->

#### Scenario: Attention weights fit device memory

- **WHEN** all layers' attention weights are pushed to the GPU
- **THEN** their device footprint SHALL fit within the target GPU's
  memory (DSv4-Flash attention weights are a few GB in f32), leaving
  room for the KV cache
<!-- test: larql_inference::attention::dsv4_generate::dsv4_bench_cpu_vs_cuda -->

