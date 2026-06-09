# larql resume — session of 2026-05-14 (CI/test-infra + bench measurement arc)

> Sibling to the GPU-arc `RESUME_PROMPT.md`. The work captured here is
> orthogonal: it cleaned up CI debt, unblocked the bench measurement
> that #127 had been blocking, and traced the bench-output gibberish
> end-to-end. **The real bug**: `predict_q4k_hidden`'s
> `direct_all_layers` optimization (in
> `crates/larql-inference/src/vindex/q4k_forward/hidden.rs:186-198`)
> skips `insert_q4k_layer_tensors` on Gemma 3 4B; the direct path
> needs a `kv_cache` to engage; both callers (`predict_q4k` and the
> test harness) pass `None`; `block.rs:96` falls back to
> `weights.tensors`, which is now empty; attention returns `None`;
> forward is a no-op. **One-line fix candidate**: have `predict_q4k`
> allocate and pass a fresh `KvCache`. See "Correctness arc" below
> for full evidence + three fix options.

## What landed today

Eleven PRs (#122 → #131), all merged to `main` except #131 (which
was opened awaiting merge — check `gh pr view 131` before assuming).

| PR | Title | What it unblocks |
|---:|---|---|
| #122 | `feat(gguf-moe)`: unblock 35B-A3B vindex extraction | Qwen 3.6-35B-A3B GGUF → vindex (~2 min wall, was hours) |
| #123 | `fix(server)`: chat completions deadlock | `/v1/chat/completions` returns; was hanging forever |
| #124 | `docs(bench)`: close blockers proposal with measured numbers | Bench gap diagnosed: 153× algorithmic |
| #125 | `feat(convert)`: `--quant q4k` flag for `gguf-to-vindex` | One-step GGUF → fast-decode vindex (dense models) |
| #126 | `test(server)`: fix infra drift + bounded-time regression for #123 | Server test suite goes from broken → 287 tests green |
| #127 | `chore(lint)`: drop unused re-export + `is_none_or` fix | Removes the last build warnings on `vindex` / `models` |
| #128 | `test(compute,vindex)`: fix trait + struct-field drift | 145 previously-broken tests unblocked |
| #129 | `test(workspace)`: close last build-target drift | `cargo build --workspace --all-targets` clean |
| #130 | `ci`: tighten check/clippy to `--all-targets` | Catches the *next* drift class at PR time |
| #131 | `ci(cli)`: re-enable clippy on larql-cli | Last "intentionally skipped" gate closed |

Posture after today:
- `cargo build --release --workspace --all-targets` — clean.
- Every crate enforces `clippy -- -D warnings` on its full `--all-targets`.
- 145+ previously-uncompilable tests now run.
- `/v1/chat/completions` no longer deadlocks (regression test pins it).
- GGUF → vindex extraction works end-to-end for MoE.

## The bench number that came out of this

The chat-completion hang (filed as task #127) had been blocking the
end-to-end measurement of larql's production decode path. With #123
fixed, the number is now measurable:

| Config | Decode (t/s) | Notes |
|---|---:|---|
| llama.cpp `-ngl 999` (Gemma 3 4B Q4_K_M) | **238** | RTX 4090, 6.8 GB VRAM |
| llama.cpp `-ngl 0` (CPU only) | **16.2** | ~2.3 GB RSS |
| **larql `/v1/chat/completions`** (Gemma 3 4B Q4_K) | **0.106** | 32 tokens in 301 s; ~11 GB RSS |

The ~153× gap to llama.cpp CPU is **algorithmic, not micro-kernel**.
`crates/larql-inference/src/layer_graph/generate/cpu.rs::generate_via_cpu_q4k`
loops `predict_q4k` per token — each call re-runs the full forward
pass over the entire prompt-so-far (no KV cache). The kernel-level
AVX2 wins from the cpu-kquant arc (PRs #102–#119) are real and
measured; closing the algorithmic gap is what compounds them into a
user-visible speedup.

See `openspec/changes/bench-vs-llama-cpp-end-to-end-blockers/proposal.md`
for the full diagnosis and the "HUNG → measurable" transition.

## ⚠ Correctness arc — V_proj fixed in current code, secondary token-N divergence open

**The 0.106 tok/s number measures throughput, not correctness.** Output
samples from the run:

- Warmup ("hi", `max_tokens=1`) → `"ům"`
- 32-token bench → `" Wndې...DeutschesYaml RemLaravel铎 XNUMX∂</ிறு ดู DBHelper..."`
- 5-token `/v1/chat/completions` ("The capital of France is", greedy):
  `" ekonomi", "ítja", "óln", "闆", " berlang"`
- `larql run` direct (no server, earlier in the session) → `"tragedy"`

This is **incoherent**, not just slow. Originally hypothesised as a
vocab mismatch, but the diagnostic checklist ran end-to-end on
2026-05-22 and proved otherwise — the bug is in extraction, not the
tokenizer.

### Diagnostic — what's known

The diagnostic ran end-to-end on 2026-05-22/23 with two test vindexes:

| Vindex | Extracted (date / path) | V_proj diff vs GGUF | Inference status |
|---|---|---|---|
| `output/gemma-3-4b-it-vindex` | 2026-04-16 via `extract --quant q4k` | **0.266** (14.86× rel) | ✗ gibberish from token 0 (`" ekonomi", "ítja", …`) |
| `output/gemma-3-4b-it-vindex-fresh` | 2026-05-29 via current `extract --quant q4k` | **0.0021** (Q6_K rounding noise) | ✓ token 1 = `"Paris"` (lp -0.0008, matches llama.cpp); tokens 2-5 drift into gibberish |
| `/tmp/gemma3-4b-q4k-fresh.vindex` | 2026-05-22 via `convert gguf-to-vindex --quant q4k` | **0.0** (bit-exact passthrough) | ✗ panics on chat (empty `gate_vectors.bin` from the convert path — secondary bug) |

Layer-level diagnostics already ruled out the easy explanations:

| Layer | Status | Evidence |
|---|---|---|
| Tokenizer | ✓ healthy | Round-trip on `output/gemma-3-4b-it-vindex/tokenizer.json`: `<bos>=2`, `<start_of_turn>=105`, `<end_of_turn>=106`, vocab size 262145, `"France"` → `[2, 31756]` → `"<bos>France"`. Standard Gemma 3 layout. |
| GGUF model file | ✓ healthy | `llama-cli` on `output/gguf-cache/gemma-3-4b-it/gemma-3-4b-it-Q4_K_M.gguf` with prompt `"The capital of France is"` → emits **`"Paris! 🇫🇷"`**. |
| Q6_K kernel roundtrip | ✓ healthy | `test_q6k_roundtrip` on V_proj row 0: max diff 0.0014 (expected Q6_K rounding). |
| In-memory writer roundtrip | ✓ healthy | `test_v_proj_writer_roundtrip` (quantize_q6_k → dequantize_q6_k in memory on GGUF V_proj): max diff 0.0021. |

### Diagnosis

**The V_proj corruption is fixed in current code.** Vindexes extracted
since (at the latest) 2026-05-29 produce healthy V_proj at Q6_K rounding
levels. The April 16 vindex was extracted with a buggy older writer; the
fix is **just re-extract** — no source-code change needed.

The smoke test on the fresh vindex (`/v1/chat/completions` on `"The
capital of France is"`) emits `"Paris"` as the first generated token
with logprob -0.0008. **But this is a coincidence**, not a correctness
win — see below.

**The real root cause** — found by running the user's staged
`test_gemma3_layer_health.rs` against the fresh vindex on 2026-05-22/23:

> All 35 per-layer dump files (`cpu_h_embed.f32` + `cpu_layer_00.f32`
> … `cpu_layer_33.f32`) have **identical MD5 hashes**. Layer-by-layer
> stats are bit-exact: mean=-0.0186, std=1.035, max_abs=29.25 for
> every single layer.

The forward pass on `predict_q4k_hidden` is a **silent no-op**.
Looking at the loop body in
`crates/larql-inference/src/vindex/q4k_forward/hidden.rs` ~line 240:

```rust
} else if let Some((h_new, _, kv_out)) = run_layer_with_ffn_with_cache(
    weights, &h, layer, ffn_backend, false, ple_inputs.get(layer),
    shared_kv, kv_cache.as_deref_mut(), Some(index),
) {
    h = h_new;
    if let Some(kv) = kv_out {
        shared_kv_cache.insert(layer, kv);
    }
}
```

When `run_layer_with_ffn_with_cache` returns `None`, `h` is never
reassigned. That's exactly what's happening: every layer silently
returns `None`, the residual stream stays at the input embedding all
the way through layer 33, and `lm_head @ embed(last_prompt_token)`
gets argmaxed.

**Why "Paris" emerged**: the last prompt token is `"is"`. With tied
embed/lm_head matrices, `lm_head @ embed("is")` reduces to inner
products against all embedding rows — and `embed("Paris") · embed("is")`
is high because "is Paris" co-occurs frequently in training. That's
the accident.

**Why tokens 2-5 are gibberish**: after emitting "Paris", the model
re-runs forward on `"...France is Paris"`. With forward still a
no-op, lm_head argmax on `embed("Paris")` returns *whatever's nearest
to "Paris" in cosine space* — which happens to be Portuguese
`"pessoal"`, Czech/Slovak `"pohod"`, Tamil `"அச்ச"`, Chinese `"澼"`.
This isn't a "model drift after first token"; it's the same no-op
forward seeing a different last-prompt-token each step.

**Why every layer returns None** — root cause identified on
2026-05-23:

The bug is the interaction between an over-eager skip optimization
in `predict_q4k_hidden` and a caller-side contract that isn't
documented.

`crates/larql-inference/src/vindex/q4k_forward/hidden.rs:186-198`:

```rust
let direct_all_layers = !arch_pre.is_hybrid_moe()
    && h.shape()[1].is_multiple_of(256)
    && q_dim_pre.is_multiple_of(256)
    && !(0..num_layers).any(|l| arch_pre.kv_shared_source_layer(l).is_some());

for layer in 0..num_layers {
    let inserted = if direct_all_layers {
        Vec::new()                                                     // skip f32 dequant
    } else {
        insert_q4k_layer_tensors(weights, index, layer)
            .unwrap_or_else(|err| panic!("{err}"))
    };
    ...
}
```

Gemma 3 4B satisfies all four conditions (`is_hybrid_moe=false`,
`hidden=2560 % 256 == 0`, `q_dim=8*128=1024 % 256 == 0`, no KV
sharing across layers). So `direct_all_layers = true` and
`insert_q4k_layer_tensors` is **skipped**. The optimization comment
at line 173-185 says this is safe "when the layer can fully run
through the direct Q4_K × Q8_K paths" — but it depends on the
caller passing a `kv_cache`.

`crates/larql-inference/src/attention/block.rs:96`:

```rust
let Some(cache) = kv_cache else {
    return run_attention_block_with_kv_out(
        weights, h, layer, capture_attention, shared_kv,
    );
};
```

When `kv_cache = None`, block.rs early-exits to
`run_attention_block_with_kv_out` (line 27), which doesn't take a
vindex and reads from `weights.tensors` directly. But the dequant
insert that would have populated `weights.tensors` was skipped.
Lookup misses → `?` propagates `None` → forward becomes a no-op.

**Both callers pass `kv_cache = None`**:

- The diagnostic test (`test_gemma3_layer_health`) calls
  `predict_q4k_hidden(weights, token_ids, index, None)` directly.
- The chat path goes `generate_via_cpu_q4k` →
  `predict_q4k(weights, tokenizer, token_ids, 5, index)` →
  `predict_q4k_hidden(weights, token_ids, index, None)` at
  `crates/larql-inference/src/vindex/q4k_forward/generation.rs:17`.

So **every CPU Q4K decode on Gemma 3 4B (and any other model that
satisfies the four `direct_all_layers` conditions) silently runs a
no-op forward pass** if the caller doesn't supply a KV cache. The
production chat path is in this group.

### Fix candidates

In increasing order of cleanup-required:

1. **Always run `insert_q4k_layer_tensors`** (drop the skip
   optimization). Restores correctness at the cost of ~10 GB RSS
   on Gemma 3 4B prefill. One-line change in `hidden.rs`. Quickest
   to land.

2. **Make the caller pass a fresh `KvCache`**. Update
   `predict_q4k` (`generation.rs:17`) and the test to allocate a
   `KvCache::new(num_layers)` and pass `Some(&mut cache)`. The
   cache is discarded after; the side effect is just to engage the
   direct path in `block.rs:96`. Smaller patch but spreads
   responsibility — every future caller of `predict_q4k_hidden`
   now needs to remember to allocate a cache when none is
   logically needed.

3. **Teach `run_attention_block_with_kv_out` to use the vindex
   when one is available.** Cleanest separation of concerns: the
   no-cache path should still try the direct vindex matvec before
   falling back to `weights.tensors`. Larger patch (refactor of
   `block.rs:27`'s signature + dispatch) but doesn't require
   callers to allocate dead caches and doesn't sacrifice the 10 GB
   savings.

Recommended: **option 2** for the immediate fix (single-call-site,
no API change, no perf regression), with option 3 filed as a
follow-up cleanup once the broader CPU KV cache arc (open arc #2
below) lands and changes the function signatures anyway.

**Two side bugs surfaced during the diagnostic**:

1. **Token-N apparent divergence is illusory** — same forward no-op
   bug as token 1. Don't waste time on session-12-style residual diff
   diagnostics; fix the layer no-op first.

2. **`larql convert gguf-to-vindex --quant q4k` writes empty
   `gate_vectors.bin`**. The convert path's gate KNN write step
   silently produces a 0-byte file, so inference forward panics in
   `predict/honest.rs:247:92` (unwrap on None). Separate bug from
   the layer no-op above. Safetensors `extract --quant q4k` path is
   unaffected (writes gate_vectors.bin correctly).

### What's already in flight

The user has these diagnostic test files staged but uncommitted in
the working tree (parallel session on 2026-05-22):

- `crates/larql-inference/tests/test_gemma3_v_proj_source_compare.rs` —
  the test that produced the 14.86× ratio above.
- `crates/larql-inference/tests/test_gemma3_wv_dump.rs` — per-layer V
  dump for visual inspection.
- `crates/larql-inference/tests/test_v_proj_writer_roundtrip.rs` —
  isolates the writer's own round-trip behaviour.
- `crates/larql-inference/tests/test_q6k_roundtrip.rs` — Q6_K
  dequant/requant round-trip at the kernel level.
- `crates/larql-inference/tests/test_gemma3_layer_health.rs` and
  `profile_4b_decode.rs` — per-layer norm/health probes.

**Don't disturb those files** — the next session continues this arc.

### Recommended next steps

In order of impact:

1. **Fix the `direct_all_layers` / `kv_cache=None` interaction in
   `predict_q4k_hidden`** (THE gating bug). Pick one of the three
   candidates from "Fix candidates" above. Recommended:
   *option 2* — make `predict_q4k`
   (`crates/larql-inference/src/vindex/q4k_forward/generation.rs:17`)
   allocate a fresh `KvCache` and pass `Some(&mut cache)` to
   `predict_q4k_hidden`. Discard the cache after; the side effect
   engages the direct path in `block.rs:96` and the dequant skip
   becomes safe.

2. **Fix the convert path's empty `gate_vectors.bin`.** The
   `larql convert gguf-to-vindex --quant q4k` flow writes a 0-byte
   gate file even when source GGUF tensors are intact. Likely in the
   `build_vindex` call order — `write_gate_vectors` runs before the
   Q4K writer takes over but appears to be a no-op for GGUF-sourced
   weights. Compare against the safetensors `extract` path that
   correctly produces a 3.5 GB gate file.

3. **Replace the May 7 broken vindex with the fresh one.** The May
   7 `output/gemma-3-4b-it-vindex` is structurally broken and should
   not be used. `output/gemma-3-4b-it-vindex-fresh` is the correct
   one to bench against, *once the layer no-op is fixed*.

4. **Re-run the 0.106 tok/s perf measurement** after #1 lands. The
   existing number is on a no-op forward and means nothing.

## Open arcs from here

In order of impact:

1. **Forward layer no-op on Gemma 3 4B** —
   `predict_q4k_hidden`'s `direct_all_layers` optimization skips
   `insert_q4k_layer_tensors` because it assumes the direct Q4_K
   path will engage; the direct path requires `kv_cache = Some`;
   callers (both `predict_q4k` and the test) pass `None`; block.rs
   falls back to `weights.tensors` which is now empty; attention
   returns `None`; forward is a no-op. The "Paris" emission is
   embed cosine coincidence. **One-line fix candidate available in
   "Correctness arc" above.** Gates the bench head-to-head — any
   perf number on the current vindex is meaningless until this
   lands.**

2. **CPU Q4K KV cache** — the 153× perf win, *once the layer no-op
   is fixed*. Wire a KV cache through
   `crates/larql-inference/src/vindex/q4k_forward/hidden.rs` and
   its attention block. Metal already has it
   (`DecodeBackend::decode_token` + `populate_kv_layer` +
   `truncate_kv_cache`); CPU just doesn't use it. Probably 2-3
   focused PRs. Expected: ~10× speedup, closer to llama.cpp CPU.

3. **Hybrid SSM Q4_K writer** — extends PR #125. The current
   `--quant q4k` flag rejects hybrid SSM archs (Qwen 3.6 family)
   because the Q4_K attn writer at
   `crates/larql-vindex/src/format/weights/write_q4k/attn.rs` only
   iterates Q/K/V/O. DeltaNet layers need their own `ssm_*` tensor
   set. Probably 2-3 PRs. Unlocks `larql convert gguf-to-vindex
   Qwen3.6-35B-A3B-... --quant q4k --level all` and the MoE
   head-to-head bench.

4. **walk_path_audit resurrection** — gated behind `#[cfg(any())]`
   in PR #129. Needs `MaskedGateIndex`'s `impl GateIndex` split
   across `GateLookup`, `PatchOverrides`, `FfnRowAccess` (and its
   sub-supertraits). 1-2 hour focused session of method
   classification. Restores the per-path WalkFfn equivalence
   harness.

## Critical environment notes (read these)

- **No CUDA on this dev box** — decode defaults to CPU. The 0.106
  tok/s bench is CPU-only.
- **Linux x86_64, glibc.** The chat-completion `pick_template`
  deadlock fixed in #123 is glibc-specific (non-reentrant
  `pthread_rwlock_t` since 2.31). The fix avoids the pattern
  entirely — should be portable, but it's worth knowing the
  failure mode is glibc-only if you ever see it again on a
  different platform.
- **`larql serve` is a wrapper.** It spawns `larql-server` as a
  subprocess and the wrapper itself blocks in `wait4`. Don't gdb
  the wrapper PID — attach to the `larql-server` child.
  `--log-level` is passed by the wrapper and **overrides
  `RUST_LOG`**; use `--log-level trace` not env var when going
  through `larql serve`.
- **`ptrace_scope=1` on this host** — gdb attach to non-child
  processes blocked without sudo. Use instrumented `eprintln!` for
  live diagnosis. (This was how #123's deadlock was localized.)
- **`/health` returns 404** on `larql-server` — there's no health
  endpoint at that path. Use `/v1/models` for liveness.

## Standing rules

- **Don't self-merge PRs unattended.** The user authorizes each
  merge individually (`merge and continue`). Standing auth is
  per-PR, not session-wide.
  See `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/feedback_unattended_merging.md`.
- **OpenSpec workflow.** Every code change references a capability
  under `openspec/specs/<name>/spec.md` (or for in-flight work,
  `openspec/changes/<id>/specs/...`). Scenarios link to tests via
  `<!-- test: <fqn> -->` annotations. Run `make ci` before pushing.
  See `CLAUDE.md`.
- **Q4K vindex format is the production fast-decode path.** f16
  vindexes work but go through the slow CPU dequant loop. The
  `interleaved_q4k.bin` / `attn_weights_q4k.bin` / `lm_head_q4.bin`
  triad is what production decode reads. Today's `--quant q4k`
  flag (#125) is the GGUF entry point.

## Quick start prompt for a fresh session

The bug is fully localized — pick a fix candidate and land it.

> Read RESUME_PROMPT_SESSION_2026-05-14.md, specifically the
> "Correctness arc" section. The bug is fully localized: in
> `crates/larql-inference/src/vindex/q4k_forward/hidden.rs:186-198`,
> the `direct_all_layers` skip leaves `weights.tensors` empty for
> Gemma 3 4B; in `crates/larql-inference/src/attention/block.rs:96`
> the `kv_cache = None` early-exit reads from that empty
> `weights.tensors` and returns `None`; in
> `crates/larql-inference/src/vindex/q4k_forward/generation.rs:17`
> `predict_q4k` passes `None` for `kv_cache`. Every CPU Q4K decode
> on Gemma 3 4B is a silent no-op forward as a result. The "Paris"
> emission earlier is an embed-cosine accident, not a correctness
> win.
>
> Land fix candidate 2 (recommended in the resume): modify
> `predict_q4k` to allocate a fresh
> `larql_inference::attention::KvCache::new(num_layers)` and pass
> `Some(&mut cache)` to `predict_q4k_hidden`. The cache is
> discarded after the call. Verify:
>
> 1. `test_gemma3_layer_health` shows per-layer stats *differ*
>    (currently all 34 are bit-identical: mean=-0.0186 std=1.035
>    max_abs=29.25).
> 2. `/v1/chat/completions` on `output/gemma-3-4b-it-vindex-fresh`
>    with `"The capital of France is"` emits coherent English
>    starting from token 1 (currently emits `"Paris"` by accident
>    + drift).
> 3. Re-run the 0.106 tok/s bench measurement once the forward is
>    real.

Or for the CPU KV cache arc:

> Read RESUME_PROMPT_SESSION_2026-05-14.md. Start the CPU Q4K KV
> cache arc (option 2 in "Open arcs"). First PR: thread an optional
> `KvCache` parameter through `predict_q4k_hidden` and the
> attention block in
> `crates/larql-inference/src/vindex/q4k_forward/hidden.rs` without
> changing behavior — just plumbing. Verify it builds clean across
> the workspace and existing tests still pass. Open a PR when the
> plumbing compiles + tests pass.

Or for hybrid SSM Q4_K:

> Read RESUME_PROMPT_SESSION_2026-05-14.md option 3. Start the
> hybrid SSM Q4_K writer arc. First step: add an
> `arch.is_full_attention_layer(layer)` helper to
> `crates/larql-models/src/config.rs` (derived from
> `full_attention_interval`), then refactor
> `crates/larql-vindex/src/format/weights/write_q4k/attn.rs::write_attn_weights_q4k`
> to dispatch per layer. Test against
> `output/gguf-cache/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

## Files touched (for quick git blame / context)

| Path | PR | Why |
|---|---:|---|
| `crates/larql-vindex/src/extract/build.rs` | #122 | MoE down_meta early-out |
| `crates/larql-vindex/src/format/weights/write_f32.rs` | #122 | WeightSource quant_tensors fallback |
| `crates/larql-server/src/routes/openai/chat.rs` | #123 | pick_template above lock_weights_for_gen |
| `openspec/changes/bench-vs-llama-cpp-end-to-end-blockers/proposal.md` | #124 | Resolution section + bench numbers |
| `openspec/changes/bench-vs-llama-cpp-end-to-end-blockers/tasks.md` | #124 | Gap 1/Gap 2 closed |
| `crates/larql-cli/src/commands/extraction/convert_cmd.rs` | #125 | `--quant q4k` flag plumbing |
| `crates/larql-server/tests/common/mod.rs` | #126 | model_with_loaded_weights helper |
| `crates/larql-server/tests/test_http_embed.rs` | #126 | bounded-time regression tests |
| `crates/larql-vindex/src/format/weights/mod.rs` | #127 | drop unused re-export |
| `crates/larql-models/src/quant/lazy.rs` | #127 | is_none_or fix |
| `crates/larql-compute/tests/test_backend_matmul_quant.rs` | #128 | DecodeBackend trait stub realigned |
| `crates/larql-compute/examples/demo_architecture.rs` | #128 | full_pipeline_q4 arity fix |
| `crates/larql-vindex/tests/{test_vindex,compute_storage_regressions,persistence_regressions}.rs` | #128 | VindexModelConfig / ModelWeights field drift |
| `crates/larql-vindex/examples/demo_features.rs` | #128 | ModelWeights field drift |
| `crates/larql-inference/examples/{debug_layers,debug_gpu_step,debug_generate,walk_path_audit}.rs` | #129 | DecodeBackend arity fix + walk_path_audit stub |
| `crates/larql-server/benches/attention_service.rs` | #129 | LoadedModel/AppState field drift |
| `crates/larql-lql/src/executor/tests.rs` | #129 | ModelWeights + VindexModelConfig drift |
| `.github/workflows/larql-{server,cli,inference}.yml` | #130 | --all-targets tightening |
| `crates/larql-inference/Cargo.toml` | #130 | mech_interp_demo required-features |
| `.github/workflows/larql-cli.yml` | #131 | re-enable clippy gate |
