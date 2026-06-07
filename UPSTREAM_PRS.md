# Upstream PR descriptions

## PR 1: `feat(ggml): TQ1_0 / TQ2_0 ternary quantisation for BitNet 1.58`

**Branch:** `gburd:larql:feat/bitnet-vindex` → `chrishayuk:larql:main`

### Why

BitNet b1.58 models (Microsoft's 1.58-bit ternary weights) are distributed
as GGUF files using the canonical ternary quantisation formats `TQ1_0`
(1.6875 bpw) and `TQ2_0` (2.0625 bpw).  These types are absent from
`larql-models/src/quant/ggml/`, so the streaming GGUF extract pipeline
(landed in #145) refuses BitNet inputs with `UnsupportedDtype`.  This PR
adds both decoders so `larql extract --gguf bitnet.gguf` works end to
end for a vindex.

Specifically targets `microsoft/bitnet-b1.58-2B-4T-gguf` on HF, which
ships TQ2_0 tensors.

### What

- Type IDs 34 (TQ1_0) and 35 (TQ2_0) added to `quant/ggml/mod.rs`
  alongside the existing Q*_K family.
- Block geometry constants (`TQ1_0_BLOCK_BYTES = 54`,
  `TQ2_0_BLOCK_BYTES = 66`) and dispatch wiring through
  `tensor_data_size`, `type_name`, and `dequantize`.
- New file `quant/ggml/tq.rs` (~500 LOC) with:
  - Decoders that mirror the canonical llama.cpp wire layout:
    - TQ2_0 packs 4 trits per byte at 2 bits each, plus an f16 scale.
    - TQ1_0 packs 5 trits per byte in base-3 plus 4 trailing trits
      per byte in `qh[]`, plus an f16 scale.
  - Reference encoders used by tests (not on a hot path).
  - Inline IEEE-754 binary16 codec to keep the module dependency-free.

### Tests

11 passing in `cargo test -p larql-models quant::ggml::tq`:

- TQ2_0: round-trip at unit + scaled (0.5, 0.25, 2.0); zero-block
  produces all-zero; two-blocks-with-different-scales preserves
  per-block scaling; truncated input errors; non-multiple n_elements
  errors.
- TQ1_0: zero-block round-trip; truncated input errors.
- Dispatch: `dequantize()` routes type IDs 34/35 correctly.
- Type-name and `tensor_data_size` recognise both.
- `tq1_0_round_trip_*` are `#[ignore]` pending validation against a
  real Microsoft BitNet GGUF (the digit-extraction trick is correct
  for the sparse patterns I tested but I want ground truth before
  pinning the canonical encoder).  TQ2_0 is what `bitnet-b1.58-2B-4T`
  ships and is fully wired.

### Backwards-compat

Strictly additive.  No existing dispatch path changes, no existing test
results change.

---

## PR 2: `feat(larql-cloud): outbound LLM client crate`

**Branch:** `gburd:larql:feat/cloud-clients` → `chrishayuk:larql:main`

### Why

`larql-server` already speaks the OpenAI API as an *answerer* (see
`crates/larql-server/src/routes/openai/`).  This PR is the inverse: a
uniform Rust trait for *calling* external LLM services from larql, so a
single `larql-server` instance can serve either a local vindex or proxy
to a remote LLM behind one HTTP endpoint.  The driving use case is
heterogeneous PostgreSQL deployments where some queries should hit a
local BitNet vindex (fast, free) and others should hit a frontier model
on Bedrock or Exoscale (high-quality, paid).

### What

New workspace member `crates/larql-cloud/`.  Two impls of the
`CloudClient` trait (infer / embed / chat):

**`OpenAiCompatible`** — one impl drives every OpenAI-API-shaped backend:

| Constructor | Backend | Auth env var |
|---|---|---|
| `openai()` | OpenAI proper | `OPENAI_API_KEY` |
| `exoscale()` | Exoscale.ch AI gateway | `EXOSCALE_API_KEY` |
| `together()` | Together AI | `TOGETHER_API_KEY` |
| `local()` | vLLM / llama.cpp / ollama / passthrough | optional bearer |

All four hit `/v1/chat/completions` and `/v1/embeddings` with the OpenAI
JSON shape.  `infer()` synthesises top-K predictions from chat output
because few non-OpenAI providers expose `logprobs`.

**`BedrockClient`** with two auth modes:

- `BedrockAuth::BearerToken` — `AWS_BEARER_TOKEN_BEDROCK` (the 2024
  short-lived API-key flow).  Plain `Authorization: Bearer …`.
- `BedrockAuth::SigV4` — full AWS Signature V4 from
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (+ optional
  `AWS_SESSION_TOKEN`) and `AWS_REGION`.  Required for IAM-role creds
  on EC2/Fargate.  Hand-rolled in 60 LOC to avoid pulling the full
  `aws-sigv4` crate.

Speaks the **Anthropic Messages API** for chat/infer (the dominant
Bedrock model family), and **`amazon.titan-embed-text-v2:0`** for
embeddings.  Non-Anthropic chat and non-Titan embed return
`ProviderError::Unsupported` rather than silently sending the wrong
wire shape.

### Tests

11 passing in `cargo test -p larql-cloud`:

- OpenAI: chat round-trip + embed round-trip + infer-from-chat
  synthesis + transport-error propagation, all driven by an in-process
  hyper mock that asserts request URIs.
- Bedrock: env-precedence (bearer beats SigV4), missing-creds errors
  out with the right env-var name, `provider_id()` reflects the auth
  mode, embed/chat reject wrong-family models with `Unsupported`,
  **SigV4 derived signing key matches the AWS canonical test vector**
  (`wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY` →
  `c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9`).

### Features

- `default = ["openai", "bedrock"]`
- `openai` — OpenAI-compat client (cheap, just reqwest + serde).
- `bedrock` — gates the SigV4 module + `hmac`/`sha2`/`hex` deps.
  BearerToken auth doesn't need them.

### Backwards-compat

Strictly additive — new workspace member, no edits to other crates.
