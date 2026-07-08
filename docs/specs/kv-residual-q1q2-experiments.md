# Experiments: KV cache vs residual stream (Q1, Q2)

Two CI experiments that empirically resolve the two open questions from the
kv-cache-vs-residual-stream research residual. They exist to support a design
decision for a larql-native coding agent: **can the model's running inference
state be extended in place with injected tokens (so the agent loop doesn't
re-prefill), and is the bounded-memory residual-stream engine usable on the
agent's actual model architectures?**

These are **discovery experiments**, not regression gates. The CI outcome *is*
the answer; a "failing"/divergent result is data, not a defect.

## Background (from the residual)

larql unifies decode behind a `KvEngine` trait with two fundamentally
different persistent-state mechanisms:

- **K/V cache** — `Standard` (per-token K/V, grows O(context)), `MarkovBounded`
  (sliding-window K/V), `NoCache` (O(N²) re-forward). General across archs.
- **Residual stream** — `MarkovResidualEngine` (`--engine markov-rs`):
  "replaces the per-token K/V cache with the residual stream itself as the
  persistent inference state" (KV-cache-free; State-Policy: *canonical =
  residual stream, derivative = hot K/V*). Bit-identical to Standard **on
  supported architectures** — validated on Gemma 3 4B.

The generation driver is `prefill(prompt_ids)` then a loop of
`decode_step(token_id)`, each call advancing a persistent store by one token.
`decode_step` takes an *arbitrary* `token_id`.

## Q1 — is `decode_step` provenance-agnostic?  → `exp1-decode-injection-parity`

If feeding a token via `decode_step` produces the same state as having it in
the prompt, the agent can inject a larql command's OUTPUT tokens into the live
state and continue — no re-prefill.

**Test** (`crates/larql-kv/tests/exp1_decode_injection.rs`, synthetic weights,
no download): the final-position hidden after `prefill([t0,t1,t2,t3])` must be
bit-identical to that after `prefill([t0,t1]) + decode_step(t2) +
decode_step(t3)`.

- green ⇒ **Q1 ACCEPT** — injection is sound; the persistent-context primitive
  already exists.
- red ⇒ **Q1 REJECT** — injection diverges; use `prefill_from_hidden` or
  re-prefill.

## Q2 — is the residual-stream engine bit-identical off Gemma?  → `exp2-engine-parity`

The agent's candidate vindexes are SmolLM2 / Qwen2, not Gemma. The residual
contract only promises bit-identity on *supported* architectures.

**Job** (`workflow_dispatch`, defaults to SmolLM2-135M): extract → run one
forward pass through `--engine standard` and `--engine markov-rs` on the same
prompt → diff the top-10 next-token tables.

- identical ⇒ **Q2 ACCEPT** — residual-stream (bounded-memory) substrate valid
  on this arch.
- divergent ⇒ **Q2 REJECT** — non-bit-identical (silent-wrong-logits risk);
  the agent rides Standard K/V and accepts O(context) growth.
- markov-rs errors ⇒ precondition-gated; also a clean answer.

## Why these gate the design

If Q1 = accept and Q2 = accept, the agent's loop can maintain the model's
context on the **residual stream** — bounded memory, aligned with larql's own
"canonical persistent inference state," and the reflexive/self-hosting model.
If Q2 = reject, the same loop still works on Standard K/V, just with growth
that `larql-probe` currently contains. Q1 is the load-bearing one: it decides
whether "extend the running context in place" is even available, versus
re-prefilling a flat context each turn.
