## ADDED Requirements

### Requirement: CPU decode-step FFN MUST support direct Q4_K × Q8_K matvec

`predict_q4k_hidden_with_cache` SHALL dispatch the FFN forward through
the direct Q4_K × Q8_K matvec path (`Q4kDirectFfn`) when (a) the
residual batch is a single row (`h.shape()[0] == 1`), (b) the layer is
not a hybrid-MoE layer (`arch.is_hybrid_moe() == false`), and (c) the
hidden size is a multiple of the Q8_K super-block size (256). All other
layer shapes — multi-row prefill, hybrid-MoE layers, models with
non-Q8_K-aligned `hidden_size` (e.g., Gemma 3 1B at `hidden=1152`) —
SHALL continue to dispatch through `WeightFfn` so they benefit from
BLAS GEMM over the dequant cache.

The direct path SHALL avoid f32 materialisation of FFN gate / up / down
weights for the forward step: gate+up flow through
`larql_compute::cpu::ops::q4k_q8k_dot::q4k_q8k_gate_up_into` and down
flows through `q4k_q8k_matvec_into` (Q4_K) or `q6k_q8k_matvec_into`
(Q6_K) per the vindex's per-tensor format string.

#### Scenario: Decode-step on dense Q4_K layer routes to Q4kDirectFfn

- **WHEN** `predict_q4k_hidden_with_cache` is called with a single-token
  input (`token_ids.len() == 1`) on a non-MoE architecture
- **THEN** the FFN forward for every layer SHALL use the `Q4kDirectFfn`
  backend (`ffn.name() == "q4k-direct"`) rather than `WeightFfn`
<!-- test: unbacked -->

#### Scenario: Prefill on dense Q4_K layer routes to WeightFfn

- **WHEN** `predict_q4k_hidden_with_cache` is called with a multi-token
  input (`token_ids.len() > 1`) on a non-MoE architecture
- **THEN** the FFN forward for every layer SHALL use the `WeightFfn`
  backend (`ffn.name() == "weights"`) so prefill stays on the BLAS GEMM
  path
<!-- test: unbacked -->

#### Scenario: Non-Q8_K-aligned hidden_size routes to WeightFfn even on decode step

- **GIVEN** a model whose `hidden_size` is not a multiple of 256 (e.g.,
  Gemma 3 1B at `hidden=1152`)
- **WHEN** `predict_q4k_hidden_with_cache` is called with a single-token
  input
- **THEN** the FFN forward SHALL use the `WeightFfn` backend; the direct
  path requires Q8_K activation alignment
<!-- test: unbacked -->

#### Scenario: Hybrid-MoE layers route to WeightFfn even on decode step

- **WHEN** the model architecture is hybrid MoE (`arch.is_hybrid_moe()
  == true`) and `predict_q4k_hidden_with_cache` is called with a
  single-token input
- **THEN** the FFN forward SHALL use the `WeightFfn` backend; the direct
  path does not yet model the MoE expert dispatch
<!-- test: unbacked -->

### Requirement: Q4kDirectFfn MUST produce output equivalent to WeightFfn within Q8_K activation noise

`Q4kDirectFfn::forward(layer, &x)` SHALL match `WeightFfn::forward(layer,
&x)` row-wise within the Q8_K activation quantisation noise envelope
(≤ 1.5 % relative error) for any input `x` of shape `[1, hidden]` that
is a post-FFN-norm activation. End-to-end CPU generation under
`Q4kDirectFfn` SHALL produce coherent output on the same Gemma 3 1B /
4B Q4_K_M vindexes that pass with `WeightFfn`.

#### Scenario: Q4kDirectFfn matches WeightFfn on synthetic activation

- **GIVEN** a real Gemma 3 Q4_K vindex with `insert_q4k_layer_tensors`
  populated for layer L
- **WHEN** `Q4kDirectFfn::forward(L, &x)` and `WeightFfn::forward(L, &x)`
  are both called with the same `[1, hidden]` input
- **THEN** the outputs SHALL agree row-wise within ≤ 1.5 % relative
  error (Q8_K activation noise envelope)
<!-- test: unbacked -->
