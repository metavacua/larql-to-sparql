# WindowedCheckpointEngine — Specification

**Status:** ✅ Shipped. W1-GPU step 4 wired + bench-validated
2026-05-17: 28 → 56.0 tok/s on Metal (window=256, Gemma 3 4B,
M3 Max, 50-token decode). W10 HOnly default-on (2026-05-21):
94.2 tok/s on the same bench.
**Audience:** LARQL contributors.

> **State Policy slot**: `(canonical = K/V within window +
> per-window checkpoints + token archive, derivative =
> `current_window_kv` (CPU shadow of the Metal cache), contract
> = exact_logits within window)`. The CPU shadow is dropped under
> W10 HOnly mask (default-on 2026-05-21); engine sits at 94.2
> tok/s on Gemma 3 4B Q4K, ~3.5% behind `standard`'s 97.6. The
> residual gap is `close_window`'s `KvDispatch::read_kv_row_at`
> readback cost — a windowed-engine concern that doesn't apply to
> the windowless residual-state engines (which reach the None
> mask and the full ceiling). See
> [state-policy.md §3.1](../../../larql-kv/docs/state-policy.md).

---

## 1. Purpose

`WindowedCheckpointEngine` provides effectively-unlimited decoding
context with bounded current memory — by checkpointing the K/V
state at fixed window boundaries (`window_size` tokens) and
archiving the prompt token IDs, then reconstructing any prior
window's full K/V via `replay_window`. The persistent state grows
linearly in *number of windows*, not in token count × kv_dim.

Use case: a 370K-token chat or document context where keeping
~26 GB of K/V resident isn't feasible, but you can afford ~30 MB
of checkpoints + token archive and accept the cost of re-prefill
when accessing earlier windows.

The engine is **not** a sliding-window cache. Sliding window drops
old tokens; `windowed_checkpoint` keeps them in the cold tier and can
replay them on demand.

---

## 2. Contract

### 2.1 Correctness contract

> Within the active window, decode output is bit-identical to
> `StandardEngine` on the same `(prompt, sampling_config)`.
> `replay_window(id)` reconstructs the K/V state for any archived
> window via `rs_extend_from_checkpoint_*` and is verified by
> `replay_window_succeeds_after_window_overflow`.

The contract does NOT extend to "decode after replay matches decode
before replay" in the cross-window case — the replay path uses the
boundary checkpoint as a position-N approximation to the original
position-N state, which is a documented information loss (the
checkpoint stores only the last-position K/V row per layer, not the
full window).

### 2.2 Window state machine

```
current_window_tokens = []
on each new token:
  - append to current_window_tokens
  - extend current_window_kv by one row per layer
  - if current_window_tokens.len() == window_size:
      close_window():
        - save last-row K/V per layer → CheckpointStore[current_window_id]
        - archive (current_window_id, current_window_tokens, abs_offset) → TokenArchive
        - reset: current_window_tokens = [], current_window_kv = None, abs_offset += window_size, current_window_id += 1
```

### 2.3 Memory contract

> Hot state: per-layer K/V for tokens in the current (partial)
> window. Cold state: a `last_row` K/V checkpoint per archived
> window (per layer) + a token-ID archive for each window.

For Gemma 3 4B at 370K tokens with `window_size=512`:
- Hot: ≤ 512 × 34 × kv_dim × 4 B ≈ 17 MB
- Cold checkpoints: 722 windows × 34 × kv_dim × 4 B (last row only) ≈ 24 MB
- Token archive: 370K × 4 B ≈ 1.5 MB
- Total: ~42 MB vs 26 GB for `Standard` unwindowed.

Compression ratio at this scale: ~600× vs full K/V.

---

## 3. Replay

`replay_window(window_id)` reconstructs the full K/V state for an
archived window:

1. Load checkpoint from `CheckpointStore[window_id - 1]` (boundary
   K/V at the end of the prior window) — falls back to `empty_prior`
   for window 0.
2. Run `rs_extend_from_checkpoint_backend` over the archived tokens,
   seeded with the prior checkpoint.
3. Returns `(Vec<SharedKV>, abs_end)`.

Replay is O(window_size) per layer, dominated by the per-token
forward pass. On Gemma 3 4B with `window_size=512`: ~6 s / window
on CPU, ~5 s / window on Metal (CPU dequant + Metal compute mix).

---

## 4. W1-GPU integration (2026-05-17)

The engine routes through `KvDispatch::coarse_*_with_state` when
the backend implements it (currently `CpuBackend`; `MetalBackend`
once the parallel `larql-compute::kv_dispatch` refactor lands):

- `try_prefill_via_dispatch` calls
  `coarse_prefill_with_state(token_ids, Some(&mut state))`; the
  captured `state.k_new_per_layer` / `v_new_per_layer` populate
  `current_window_kv` directly.
- `decode_step_via_dispatch` calls
  `coarse_decode_step_with_state`; the new K/V row per layer is
  appended to `current_window_kv`; `close_window` fires when
  `current_window_tokens.len()` hits `window_size`.

The legacy CPU per-layer walk path (`process_quant` via
`rs_extend_from_checkpoint_quant`) remains as the fallback.

**Memory note:** with W1-GPU active, the backend's internal K/V
cache grows alongside the engine's `current_window_kv` shadow.
This defeats the engine's memory-bounded contract at long contexts
— the backend cache holds the full sequence even though the engine
checkpoints + evicts. Follow-up: expose `KvHandle::evict_oldest(n)`
on `KvDispatch` so engines can bound the backend cache to match
their window.

---

## 5. Implementation

| Concern | Location |
|---|---|
| Engine struct + `KvEngine` impl | `crates/larql-kv/src/engines/windowed_checkpoint/engine.rs` |
| Checkpoint storage | `engines/windowed_checkpoint/checkpoint_store.rs` |
| Token archive | `engines/windowed_checkpoint/token_archive.rs` |
| Per-token K/V extension | `engines/windowed_checkpoint/extend.rs::rs_extend_from_checkpoint_*` |
| W1-GPU dispatch helpers | `engines/windowed_checkpoint/engine.rs::try_prefill_via_dispatch` + `decode_step_via_dispatch` |

---

## 6. Non-goals

- **Replay of arbitrary positions inside a window.** Replay
  reconstructs the *whole window* from the prior checkpoint;
  there's no API to extract K/V at a single sub-window position.
- **Compression of the cold tier.** That's `markov_residual_codec`
  / `boundary_per_layer`. `windowed_checkpoint` keeps cold
  checkpoints as raw f32 K/V (one row × kv_dim × 4 B per layer per
  window).
- **Cross-session resume.** `windowed_checkpoint`'s archive lives
  in-process; for persisted resume use `boundary_kv` (which emits
  `larql-boundary` frames to disk).

---

## 7. P1 follow-ups (from `crates/larql-kv/ROADMAP.md`)

- **Auto-rewind variant of `boundary_kv`** — emits boundary
  checkpoints + resets Metal's K/V cache, then re-prefills from the
  last frame. Cleaner alternative for "bounded memory at fused
  speed" since it explicitly composes with `standard` rather than
  maintaining a shadow K/V store. Should be benchmarked against
  W1-GPU'd `windowed_checkpoint` once both are wired.
- **Page-aligned KV slabs.** The current `CheckpointStore` uses
  owned `Vec<f32>` per layer per checkpoint; a hugepage-backed slab
  would cut allocation churn during 370K-token replays.

---

## 8. W10 (2026-05-18) — state-bridge mask cascade (opt-in)

Under `LARQL_W10_HONLY=1`, the engine drops its `current_window_kv`
shadow on Metal and requests the `HOnly` capture mask — Metal's own
kv cache becomes the K/V source of truth within the window. The
shadow only existed to satisfy `close_window`'s checkpoint
emission, which now pulls the last position's K/V back from the
Metal cache on demand via `KvDispatch::read_kv_row_at`.

| Path | Mask | Engine shadow |
|---|---|---|
| `LARQL_W10_HONLY=0` (default) | `Full` | `current_window_kv` pre-allocated |
| `LARQL_W10_HONLY=1` | `HOnly` | `current_window_kv = None`; close_window reads back from Metal |

Preserves the **exact within window** contract — the kv cache
Metal maintains for attention is the same one we'd otherwise have
shadowed on CPU. Measured: 88.2 → 92.8 tok/s under `HOnly`, hot
memory 9.6 MB → 0 MB (window=256).

The `None` mask is **not** applied here because h_in is still
needed for cold-tier replay across windows (the engine's
canonical state extends beyond the current window via the
checkpoint chain). A future Phase C-v2 with a Metal-side residual
cache would unlock `None` for unlimited too; deferred until the
~hidden_size × max_seq GPU memory cost justifies it.
