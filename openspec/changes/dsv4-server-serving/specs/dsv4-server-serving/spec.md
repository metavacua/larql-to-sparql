## ADDED Requirements

### Requirement: GGUF-backed DSv4 model residency

`larql-server` SHALL serve a `deepseek_v4` model directly from a GGUF
file: on the first request for such a model it SHALL open the GGUF,
extract the DSv4 hyperparameters, and load the resident-quantized layers
and head **once**, holding them for the process lifetime and reusing them
across all later requests. A load failure (missing GGUF, insufficient
memory) SHALL return a typed error, not panic.

#### Scenario: Resident layers loaded once

- **WHEN** the first `deepseek_v4` chat request arrives for a model
- **THEN** the server SHALL load that model's resident-Q4_K layers + head
  from its GGUF exactly once and reuse them for every subsequent request
  (no per-request reload)

#### Scenario: Load failure is a typed error

- **WHEN** the GGUF is missing or the resident set does not fit in memory
- **THEN** the server SHALL return a typed error describing the failure,
  and SHALL NOT crash the process

### Requirement: DSv4 chat completions

`larql-server` SHALL handle `deepseek_v4` through the OpenAI chat route —
replacing the prior "not supported" rejection — by tokenizing the prompt,
running the DSv4 resident generate, and detokenizing the output, honoring
`max_tokens`, the sampling parameters, the EOS token, and stop strings.

#### Scenario: DSv4 chat request returns a completion

- **WHEN** a `/v1/chat/completions` request targets a served DSv4 model
- **THEN** the server SHALL return a coherent completion produced by the
  DSv4 resident forward, respecting `max_tokens` and stopping on EOS or a
  stop string

#### Scenario: DSv4 is no longer rejected

- **WHEN** the served model's architecture is `deepseek_v4`
- **THEN** the route SHALL NOT return the legacy "not yet available"
  rejection

### Requirement: DSv4 token streaming

When streaming is requested, the server SHALL emit one SSE chunk per
generated token for a DSv4 model, producing the same token sequence as
the buffered (non-streaming) path for an identical request.

#### Scenario: Streamed tokens match buffered

- **WHEN** the same greedy DSv4 request is run streaming and buffered
- **THEN** the concatenated streamed tokens SHALL equal the buffered
  completion

### Requirement: DSv4 prefix-cache reuse in serving

The server SHALL back DSv4 serving with a per-model on-disk prefix cache,
so that a request sharing a prompt prefix with an earlier request reuses
the cached prefix instead of re-prefilling it. Enabling the cache SHALL
NOT change generated output (greedy-transparent).

#### Scenario: Shared prefix is reused across requests

- **WHEN** two DSv4 requests share a block-aligned prompt prefix
- **THEN** the second SHALL reuse the first's cached prefix (skipping its
  re-prefill) and produce the same greedy output it would have produced
  cold
