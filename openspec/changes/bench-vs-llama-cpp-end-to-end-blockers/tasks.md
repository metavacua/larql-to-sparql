# Tasks

This change is documentation-only. The action items below capture the work each gap implies; they're held outside this change so root-cause investigation can produce focused fix PRs.

## Open follow-ups

- [x] **Gap 1**: `/v1/chat/completions` hang on Gemma 3 4B vindex — **fixed in PR #123** (task #127). Root cause was `std::sync::RwLock` self-deadlock in `pick_template` inside the chat handler's write-lock scope, not the `--feature-major-down` hypothesis. Bounded-time integration test still TODO.

- [x] **Gap 2**: MoE branch in `convert gguf-to-vindex` — **fixed in PRs #119, #120, #121, #122** (tasks #128, #129). End-to-end verified on `unsloth/Qwen3.6-35B-A3B-GGUF`: all expert weight files non-zero, extraction ~2 min wall.

- [x] **Re-run the head-to-head bench** — done 2026-05-14, numbers folded into `proposal.md`. Headline: production CPU `/v1/chat/completions` Gemma 3 4B Q4_K = **0.106 tok/s** vs llama.cpp CPU = 16.2 tok/s (~153× gap). See `## Resolution` in proposal.md for details.

## New follow-ups surfaced by the bench

- [ ] **CPU Q4K KV cache** — close the 153× gap by wiring KV cache into `generate_via_cpu_q4k`. Currently the function explicitly comments "O(N²) in context length (no KV cache)". The per-matvec AVX2 wins from PRs #102–#119 are kernel-level only; the remaining gap is algorithmic.
- [ ] **Bounded-time integration test for `/v1/chat/completions`** — kicks the regression that #123 fixed if it ever comes back. Spec scenario already drafted in `openspec/specs/server-attention-service`.
- [ ] **Q4K-passthrough writer for `convert gguf-to-vindex`** (task #130) — emits the fast-decode `interleaved_q4k.bin` directly from GGUF Q4_K blocks instead of dequantising to f16. Without this, GGUF-sourced vindexes can't use the fast-decode production path.
