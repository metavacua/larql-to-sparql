## Context

DSv4-Flash inference is validated only against internal references
(CPU-vs-CUDA greedy equality; quant-vs-f32 tolerance from the archived
`dsv4-quant-residency`). An external authority (HF `transformers`) is
needed to catch correlated bugs both internal paths share. The
long-standing blocker: HF reference generation needs the weights + a
Python env, which can't run in a Rust test.

The repo already has the *self-captured* golden pattern
(`tests/test_logits_goldens.rs`): pinned top-K token IDs + top-1 logit,
skip-if-absent, refresh-via-env. That deliberately avoids a Python HF
dependency by capturing the model's own output. This change adds the
*stronger* external-HF authority for DSv4 specifically, keeping the
Python step out-of-band.

## Goals / Non-Goals

**Goals:**
- A decoupled reference-dump format + Python generator + Rust consumer.
- Pin the DSv4 GGUF forward to HF `transformers` on a fixed prompt
  (greedy-token + top-K + tolerance-bounded logit).
- Skip cleanly without the dump/GGUF (green-by-skip CI).

**Non-Goals:**
- Running `transformers` inside the Rust test (it stays out-of-band).
- Bit-exact agreement (GGUF Q4_K vs HF f16/bf16 differ by quant error).
- Multi-prompt / full-sequence parity (one prompt, final-position
  next-token is enough to pin the forward; extend later if needed).
- Changing the forward or any production path.

## Decisions

**D1 — JSON dump, committed when small.** The dump is the HF-tokenized
`token_ids` + final-position top-K `(token_id, logit)`. Top-K (not the
full 129 280-wide logit vector) keeps it tiny and committable; the
prompt's `token_ids` are carried so the Rust side bypasses any
tokenizer mismatch. Rationale: human-readable, diff-able, no binary
format or `npy` dependency.

**D2 — Decoupled generation.** `scripts/dsv4_hf_reference.py` (HF
`transformers`) produces the dump out-of-band; the Rust test only
*reads* it (`serde_json`). Rationale: the existing goldens file
documents that a Python step inside a Rust test is fragile (HF version,
env, weights) — decoupling gets the external authority without that
fragility.

**D3 — Argmax-first assertions, loose logit tolerance.** Primary: the
GGUF forward's greedy next-token == reference top-1, and top-K sets
overlap. Secondary: top-1 logit within a generous *relative* tolerance.
Rationale: Q4_K vs f16/bf16 shifts logit magnitudes by quant error
(larger than the ~5e-2 CPU-vs-Metal noise the goldens test uses), but
argmax is stable for non-degenerate gaps — the meaningful correctness
signal. The tolerance is calibration-pending on the first real dump.

**D4 — Reuse the existing DSv4 forward + top-K.** The test calls
`dsv4_streaming_model_forward_cached` (full 43 layers) + the existing
`dsv4_topk_logits` extractor — no new forward path. (The resident
forward would give the same logits within the quant tolerance; streaming
is the simplest to drive in a test and needs no big-RAM host.)

**D5 — Skip, don't fail, when artifacts are absent.** Mirrors every
other real-GGUF `#[ignore]` test: check the path, print a skip, return.
The dump path is an env var (`LARQL_DSV4_HF_REF`) with a default under
`tests/goldens/`.

## Risks / Trade-offs

- **Tolerance calibration** → the logit-value bound is unknown until the
  first real dump; start generous + greedy-token-primary, tighten later
  (D3). Mitigated by making greedy-token the load-bearing assertion.
- **Tokenizer drift** → carrying `token_ids` in the dump (D1) removes
  the GGUF-tokenizer-vs-HF-tokenizer variable entirely.
- **HF model availability** → the dump is maintainer-sourced; the
  harness is inert without it (D5), so this never blocks CI.
- **Quant error too large for argmax** → if Q4_K shifts the argmax on
  the chosen prompt, pick a prompt with a clear top-1 gap (the generator
  can report the gap; choose a confident prompt like the goldens'
  "The capital of France is").

## Migration Plan

Purely additive (new script + new test + new dump format). No migration;
nothing references it until the dump is generated. Rollback = delete the
test + script.

## Open Questions

- Which prompt(s)? Start with one confident-completion prompt; the
  generator can sweep a few and report argmax gaps to pick a stable one.
- Commit the dump or keep it env-pathed? Commit it once generated if the
  top-K JSON is small (it is); until then, env-pathed + skip.
