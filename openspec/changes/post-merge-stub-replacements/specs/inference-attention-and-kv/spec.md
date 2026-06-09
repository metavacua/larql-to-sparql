## ADDED Requirements

### Requirement: Synthetic Gemma 3 fixture exercises the arch branches

`make_gemma3_test_weights()` SHALL construct a synthetic
`ModelWeights` whose `arch` is a real `Gemma3Arch`, populating every
tensor + vector that the attention, FFN, and forward layer code
reach for on the Gemma 3 path so unit tests for those paths run
end-to-end without panicking.

The fixture SHALL include, per layer:

- Attention Q/K/V/O projection matrices at the Gemma 3 GQA shape.
- Per-head QK norm vectors via `arch.attn_q_norm_key` and
  `arch.attn_k_norm_key` (Gemma 3 returns `Some(_)` from both).
- Pre/post FFN norm vectors via `arch.pre_feedforward_layernorm_key`
  and `arch.post_feedforward_layernorm_key` so the post-norms
  branch in the forward layer dispatch fires.

The fixture MUST set `norm_weight_offset = 1.0` semantics by storing
zero-vector norm weights (Gemma 3 adds `1.0` at runtime, so a zero
saved weight is the identity).

#### Scenario: Gemma3 fixture exposes QK and post-norm keys
- **WHEN** a caller invokes `make_gemma3_test_weights()`
- **THEN** for every layer in `0..num_layers`,
  `arch.attn_q_norm_key(layer)` SHALL return `Some(k)` and
  `weights.vectors` SHALL contain key `k`. The same SHALL hold for
  `arch.attn_k_norm_key`. Additionally,
  `arch.has_post_norms()` SHALL be `true` and
  `arch.norm_weight_offset()` SHALL be `1.0`.
<!-- test: larql_inference::engines::test_utils::tests::gemma3_fixture_has_qk_norm_and_post_norm_keys -->

#### Scenario: Gemma3 fixture shape matches its declared dims
- **WHEN** a caller invokes `make_gemma3_test_weights()`
- **THEN** `weights.embed.shape()` SHALL equal `[vocab_size,
  hidden_size]` and `weights.lm_head.shape()` SHALL equal
  `[vocab_size, hidden_size]`, with `num_layers == 2`.
<!-- test: larql_inference::engines::test_utils::tests::gemma3_fixture_shape_matches_dims -->

### Requirement: Synthetic StarCoder 2 fixture exercises the arch's distinguishing branches

The `make_starcoder2_test_weights()` helper SHALL construct a
synthetic `ModelWeights` whose `arch` is a real `StarCoder2Arch`,
populating the dormant branches in the attention, FFN, and forward
layer code:

- `norm_type() == NormType::LayerNorm` (not RMSNorm).
- `ffn_type() == FfnType::Standard` (not gated).
- `arch.ffn_up_key` SHALL return a string containing `c_fc`, and
  `arch.ffn_down_key` SHALL return one containing `c_proj`.
- Attention biases via `attn_{q,k,v,o}_bias_key` SHALL return
  `Some(_)` and the corresponding vectors SHALL be present.
- FFN up/down biases SHALL be present.
- Activation SHALL be `GeluTanh`.

#### Scenario: StarCoder2 fixture wires LayerNorm + biases + c_fc/c_proj naming
- **WHEN** a caller invokes `make_starcoder2_test_weights()`
- **THEN** `arch.family() == "starcoder2"`, `arch.norm_type()` is
  `NormType::LayerNorm`, `arch.ffn_type()` is `FfnType::Standard`,
  `arch.ffn_up_key(0)` contains `"c_fc"`, `arch.ffn_down_key(0)`
  contains `"c_proj"`, and for every layer the Q/K/V/O bias keys
  and FFN bias keys SHALL be present in `weights.vectors`.
<!-- test: larql_inference::engines::test_utils::tests::starcoder2_fixture_has_layernorm_and_biases -->

#### Scenario: StarCoder2 fixture shape matches its declared dims
- **WHEN** a caller invokes `make_starcoder2_test_weights()`
- **THEN** the embed and lm_head shapes equal `[vocab_size,
  hidden_size]`, with `num_layers == 2`.
<!-- test: larql_inference::engines::test_utils::tests::starcoder2_fixture_shape_matches_dims -->

### Requirement: Fallible generation wrappers return typed Result

The `try_generate*` family SHALL each run the corresponding
infallible `generate*` and then convert the embedded
`GenerateResult::error: Option<GenerateError>` to a
`Result<GenerateResult, GenerateError>` via
`GenerateResult::into_result`. On the `Ok` branch the returned
`GenerateResult.error` SHALL be `None`. On the `Err` branch the
typed enum variant of the underlying `GenerateError` SHALL be
preserved.

#### Scenario: empty_success round-trips to Ok
- **WHEN** a `GenerateResult::empty_success()` value is passed to
  `into_result()`
- **THEN** the call SHALL return `Ok(result)` with
  `result.error.is_none() == true`.
<!-- test: larql_inference::layer_graph::generate::tests::try_generate_wraps_ok_result -->

#### Scenario: empty_error round-trips to typed Err
- **WHEN** a `GenerateResult::empty_error(GenerateError::unsupported_backend("no Q4"))`
  value is passed to `into_result()`
- **THEN** the call SHALL return
  `Err(GenerateError::UnsupportedBackend { reason: "no Q4" })`.
<!-- test: larql_inference::layer_graph::generate::tests::try_generate_wraps_typed_error -->

#### Scenario: Partial output preserved across Err mapping
- **WHEN** a `GenerateResult` with non-empty `tokens` and
  `error: Some(PrefillFailed)` is passed to `into_result()`
- **THEN** the call SHALL return `Err(PrefillFailed { .. })`. (The
  tokens vec is dropped on the `Err` branch; this is the contract.)
<!-- test: larql_inference::layer_graph::generate::tests::try_generate_with_sampling_preserves_partial_tokens_on_error -->

#### Scenario: try_generate_streaming has Result return shape
- **WHEN** the `try_generate_streaming` wrapper is invoked with any
  callback closure
- **THEN** its return type SHALL be `Result<GenerateResult, GenerateError>`.
<!-- test: larql_inference::layer_graph::generate::tests::try_generate_streaming_signature_returns_result -->
