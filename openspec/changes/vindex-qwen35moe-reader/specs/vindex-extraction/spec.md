## ADDED Requirements

### Requirement: Vindex SHALL expose DeltaNet weights to consumers

The vindex storage façade SHALL provide a per-layer accessor that
returns the 5 DeltaNet matmul tensors written by
`vindex-qwen35moe-extraction`'s `deltanet_weights_q4k.bin`. The
accessor SHALL return `(byte_slice, format_tag)` tuples for each of
`attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out` in
that order, mirroring `attn_q4k_layer_data`'s 4-tuple Q/K/V/O
contract.

The accessor SHALL return `None` for layers where
`arch.is_linear_attention_layer(layer) == false` (full-attention
layers carry no DeltaNet tensors) AND for vindexes whose
`deltanet_weights_q4k.bin` is absent (non-hybrid models).

#### Scenario: Qwen 3.6 35B-A3B linear layer 0 returns 5 DeltaNet tuples

- **GIVEN** the vindex at `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex`
  produced by PR #147's smoke run (`full_attention_interval=4`,
  layer 0 is a linear-attention layer)
- **WHEN** `vindex.deltanet_q4k_layer_data(0)` is called
- **THEN** the result SHALL be `Some([...; 5])` with the five
  Q4_K-tagged byte slices for the layer's DeltaNet matmuls
<!-- test: unbacked -->

### Requirement: Vindex-loaded Qwen35Weights SHALL be functionally equivalent to GGUF-loaded

A new loader function in larql-inference SHALL populate every
field of the in-memory `Qwen35Weights` struct from a vindex
directory such that `qwen35_forward_step` produces non-NaN,
non-zero logits when invoked on the resulting weight struct.

Equivalence to the GGUF-loaded path is NOT required to be
bit-exact in this change — Q4_K → f32 → Q4_K round-trips can
introduce ≤ 1% per-element noise. A separate parity-vs-llama.cpp
change covers the floor.

#### Scenario: vindex-loaded forward produces non-degenerate logits

- **GIVEN** the vindex at `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex`
- **WHEN** `qwen35_forward_step(token_id, weights, ...)` is
  invoked with `weights = load_qwen35_weights_from_vindex(dir)`
- **THEN** the returned `Array1<f32>` SHALL contain ≥ 100 distinct
  finite values across the 248044-entry vocab, and the top-1
  argmax SHALL be a valid token id in `0..vocab_size`
<!-- test: unbacked -->

### Requirement: larql-server SHALL dispatch qwen35-family arches through the hybrid forward

The server SHALL route request decoding through the
`qwen35_forward_step`-based helper for any vindex whose
`VindexConfig::model_config::model_type` is in `{"qwen35",
"qwen35moe"}`. The helper SHALL forward all 40 layers including
the 30 DeltaNet + 10 full-attention split. The standard
transformer decode path used by other arches SHALL NOT be used
for these models.

Non-qwen35 arches SHALL continue to use the standard
`predict_q4k_hidden_with_cache` path unchanged.

#### Scenario: server dispatches qwen35moe to the qwen35 forward

- **GIVEN** the server loads
  `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex` with
  `model_type == "qwen35moe"`
- **WHEN** a `/v1/chat/completions` request arrives
- **THEN** the decoded response SHALL come from
  `qwen35_forward_step` (verified by a route-tracing test hook),
  not from `predict_q4k_hidden_with_cache`
<!-- test: unbacked -->
