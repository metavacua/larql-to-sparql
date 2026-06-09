## Why

The CPU-FFN AVX2 arc (PRs #102–#117, doc capability `cpu-kquant-matvec-correctness-avx2`) landed real per-matvec speedups verified by `cargo bench` at ~17 Gelem/s on Q4_K / Q4_KF / Q6_K. But when we tried to measure **end-to-end tok/s + VRAM head-to-head against llama.cpp** on 2026-05-14, two distinct gaps in the surrounding plumbing blocked a real comparison. This proposal documents both so they're trackable and the next person can pick them up without rediscovering them.

No code lands under this id — proposal-only, sibling to `cpu-kquant-matvec-correctness-avx2` and `cuda-decode-perf-results-followup`.

## What was measured cleanly

| Run                                      | Decode tok/s | VRAM | CPU RAM | Status |
|------------------------------------------|-------------:|-----:|--------:|--------|
| llama.cpp `-ngl 999` (Gemma 3 4B Q4_K_M) | **238**       | 6.8 GB | minimal | ✓ clean |
| llama.cpp `-ngl 0` (CPU-only)            | **16.2**      | 0.7 GB idle | ~2.3 GB | ✓ clean |
| `larql bench --backends cpu` (research path) | **0.1**       | ~0.5 GB (CUDA init) | 10.7 GB | ✓ but off-path |
| `cargo bench -p larql-compute` AVX2 matvecs (isolated) | 17 Gelem/s | — | — | ✓ kernel-level |
| **larql `/v1/chat/completions` end-to-end** (Gemma 3 4B Q4_K) | **0.106** (32 tok in 301 s) | — | ~11 GB | ✓ measured after #123 |
| **larql Qwen 3.6 35B-A3B vindex** (GGUF→f16) | not benched | — | — | ✓ extraction OK after #122; format is f16-only — fast-decode requires #130 |

### End-to-end gap (post-Gaps-1+2-fix)

- Production CPU Q4K decode: **0.106 tok/s** (#127 unblocked the chat handler, so this number is now measurable).
- llama.cpp CPU-only on the same weights: **16.2 tok/s**.
- **Gap: ~153×.**
- Per-matvec AVX2 wins (PRs #102–#119) closed the kernel-level distance; the remaining factor is algorithmic — `generate_via_cpu_q4k` is documented as O(N²) per token (no KV cache on the CPU Q4K path). Closing the gap requires KV cache wiring on the CPU production path, not more matvec micro-optimisation.

## Gap 1 — `/v1/chat/completions` and `/v1/infer` hang on Gemma 3 4B vindex

### Reproducer

```bash
cargo build --release -p larql-cli --bin larql
./target/release/larql serve output/gemma-3-4b-it-vindex --port 8888 --host 127.0.0.1 &

curl -s http://127.0.0.1:8888/v1/health
# {"requests_served":1,"status":"ok","uptime_seconds":2}

curl -s http://127.0.0.1:8888/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-4b-it","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
# (hangs forever; 18-min curl returned empty body)
```

### Observations

- Server boots fine. `/v1/stats` reports `loaded.inference: true`, `q4k_ffn.cache_slots: 0`.
- Bootstrap log line worth noting: `Down features: not available` — likely the trigger.
- During the hung request, server process stays at ~1.4 % CPU. Not actively computing.
- VRAM holds at ~1.3 GB (CUDA init residue, not actual work).
- RSS holds at 10.7 GB (vindex mmap fully resident).

### Impact

Blocks any end-to-end tok/s measurement of the production CPU FFN path. The microbench wins from PRs #102–#117 are real at the kernel level but can't be demonstrated through the server until this is fixed.

### Hypothesis (unverified)

The chat-completion route lazy-loads inference state and may be waiting on a structure that the `extract --quant q4k` output omits when `down_features_q4k.bin` is missing. Worth checking:

- `crates/larql-server/src/routes/openai/chat.rs` handler — what state does it wait on?
- `crates/larql-inference/src/vindex/q4k_forward/walk_ffn.rs` — does down-features-missing trigger a fallback that compiles to a no-op or a `loop { sleep }`?
- The `--feature-major-down` extraction flag controls whether `down_features_q4k.bin` is written. The Gemma 3 4B vindex was extracted without it. Check whether the chat route silently requires it.

## Gap 2 — `convert gguf-to-vindex` doesn't extract MoE expert tensors

### Reproducer

```bash
hf download unsloth/Qwen3.6-35B-A3B tokenizer.json --local-dir <GGUF-dir>

./target/release/larql convert gguf-to-vindex \
  output/gguf-cache/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -o output/Qwen3.6-35B-A3B-q4k.vindex \
  --level all --f16
```

### Observations

- Extraction reports `intermediate_size=0` — Qwen 3.6 MoE has per-expert intermediate, not a single number, and the converter doesn't translate this.
- Output vindex is 2.5 GB total (expected ~22 GB for a Q4_K 35B model).
- Expert tensor files are present but **empty** (0 bytes):
  - `down_weights.bin`
  - `up_weights.bin`
  - `gate_vectors.bin`
- Attention, embeddings, lm_head, tokenizer, and norms all extract correctly.

### Impact

You cannot run a Qwen 3.6 35B-A3B model through larql via the GGUF path without manually building the MoE FFN structure or re-extracting from safetensors. The unsloth GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`) is the only place to get Q4_K_M for this model size without re-quantising; safetensors-based extraction requires the full 65 GB BF16 download.

### Adjacent precedent

Task #34 ("Extract Qwen3.6-35B-A3B vindex", completed earlier) used the `extract` pipeline against safetensors which DOES handle MoE (see `crates/larql-vindex/src/format/weights/write_q4k/moe_layers.rs` — write_moe_layers exists and writes Q4_K MoE entries). The `convert gguf-to-vindex` path is missing this branch.

### Suggested fix shape

In `crates/larql-cli/src/commands/extraction/convert_cmd.rs` (or wherever `gguf-to-vindex` lives), detect MoE architecture from GGUF metadata (`general.architecture` = `qwen35moe` or similar) and dispatch to a MoE-aware writer that:

- Reads `blk.{L}.ffn_gate_inp.weight` (router)
- Reads `blk.{L}.ffn_{gate,up,down}_exps.weight` (3-D expert tensors, GGUF stores them as a single per-layer tensor with shape `[expert_count, intermediate, hidden]`)
- Writes through the existing MoE path that the safetensors `extract --quant q4k` already uses.

## Capabilities

`server-attention-service` — gains a normative scenario that POSTs to `/v1/chat/completions` must complete in bounded time on any vindex `load_inference()` accepted; failure to respond should fast-fail with a clear error, not hang.

## Impact

- **Affected files**: none directly — this is a status / gap doc.
- **Affected systems**: documentation of two blockers for end-to-end bench-vs-llama.cpp.
- **Out of scope**: actually fixing either gap. Each warrants its own change once root cause is established.

## Resolution (2026-05-14)

Both gaps closed, end-to-end bench is now runnable. New gap surfaced.

- **Gap 1** — `/v1/chat/completions` hang: **fixed in PR #123**. Root cause was a `std::sync::RwLock` self-deadlock: both chat paths held the `model.weights` write lock and then called `pick_template`, which tried to take a read lock on the same RwLock (non-reentrant on glibc). The `Down features: not available` hypothesis in this doc was a red herring. Fix: hoist `pick_template` + `render` above `lock_weights_for_gen` in both `run_chat_completion` and `stream_chat_completion`.
- **Gap 2** — MoE GGUF extraction: **fixed in PRs #119, #120, #121, #122**. End-to-end verified on `unsloth/Qwen3.6-35B-A3B-GGUF` — extraction completes in ~2 min, all expert weight files non-zero. Note the resulting vindex is f16, not interleaved Q4K — production fast-decode for GGUF-sourced vindexes still needs a Q4K-passthrough writer (task #130).
- **New gap surfaced** — `generate_via_cpu_q4k` is O(N²) per token (no KV cache on the CPU Q4K production path). At Gemma 3 4B that's a ~153× gap vs llama.cpp `-ngl 0`. Closing it requires KV cache wiring on the CPU production path, not more per-matvec AVX2 work. Tracked as a follow-up; not filed under this change.
