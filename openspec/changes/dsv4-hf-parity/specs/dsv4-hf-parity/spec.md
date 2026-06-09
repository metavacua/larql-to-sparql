## ADDED Requirements

### Requirement: External HF reference dump format

The harness SHALL define a versioned, human-readable reference-dump
format carrying the HF-tokenized prompt and the HF reference's
final-position next-token distribution, so reference generation
(Python + `transformers`) is fully decoupled from the Rust test that
consumes it.

#### Scenario: Dump carries tokenized prompt and top-K reference

- **WHEN** the Python generator runs the HF DeepSeek-V4-Flash model on
  the fixed prompt
- **THEN** it SHALL write a JSON dump containing the model name, the
  prompt string, the HF-tokenized `token_ids`, and the final-position
  next-token **top-K** as `(token_id, logit)` pairs

#### Scenario: Rust consumes the dump without a Python dependency

- **WHEN** the Rust parity test loads the dump
- **THEN** it SHALL parse the JSON with no Python/`transformers`
  dependency, using the dump's `token_ids` directly (so tokenizer
  differences cannot cause a spurious mismatch)
<!-- test: larql_inference::test_dsv4_hf_parity::dsv4_forward_matches_hf_reference -->

### Requirement: DSv4 forward matches the HF reference

The DSv4-Flash GGUF forward SHALL, on the dump's `token_ids`, produce a
final-position next-token distribution that matches the HF reference:
the greedy (argmax) next token SHALL equal the reference top-1, and the
model's top-K SHALL overlap the reference top-K. Because the GGUF is
Q4_K-quantized while the reference is HF f16/bf16, logit *values* are
compared only within a documented, generous tolerance — argmax
stability, not bit-agreement, is the contract.

#### Scenario: Greedy next-token matches the reference

- **WHEN** the DSv4 GGUF forward runs on the reference prompt's
  `token_ids`
- **THEN** the argmax of the final-position logits SHALL equal the
  reference top-1 `token_id`
<!-- test: larql_inference::test_dsv4_hf_parity::dsv4_forward_matches_hf_reference -->

#### Scenario: Top-K overlaps and top-1 logit within tolerance

- **WHEN** the DSv4 forward's final-position top-K is compared to the
  reference top-K
- **THEN** the two top-K sets SHALL overlap (share the leading
  token(s)), and the top-1 logit SHALL agree within the documented
  relative tolerance
<!-- test: larql_inference::test_dsv4_hf_parity::dsv4_forward_matches_hf_reference -->

### Requirement: Harness skips cleanly without the reference dump

The parity test SHALL be inert in environments lacking the reference
dump or the real GGUF — it SHALL skip (not fail), so default CI stays
green and the gate runs only where both artifacts are present.

#### Scenario: Missing dump or GGUF skips the test

- **WHEN** the reference dump file or the DSv4-Flash GGUF is absent
- **THEN** the test SHALL print a skip notice and return success
  rather than failing
<!-- test: larql_inference::test_dsv4_hf_parity::dsv4_forward_matches_hf_reference -->
