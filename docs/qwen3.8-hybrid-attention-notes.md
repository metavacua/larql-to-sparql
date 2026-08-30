# Qwen3.8-27B — hybrid-attention architecture notes for R2

**Programme:** informs [K3 Funnel](k3-funnel.md) R2 (Kimi Linear 48B-A3B, the KDA
stack) — Qwen3.8-27B is a second real-world instance of the same problem class
(hybrid linear/full attention + MTP), so it's evidence for how the R2
abstraction should be shaped, not a separate work item.
**Scope:** ground-truth architecture facts for `Qwen/Qwen3.8-27B`, and what
they imply VINDEX3 needs to be able to represent once R2 lands for real.
**Status:** research note, 2026-08-19. Nothing here is implemented; the
concurrent `worktree-qwen35-linear-cfg` branch (larql#274) is the actual
(config-plumbing-only) code change this cycle.

---

## 1. Ground truth: what Qwen3.8-27B actually declares

Pulled directly from the checkpoint's own `config.json`
(`Qwen/Qwen3.8-27B`, `model_type: qwen3_5`) — not from secondhand
descriptions. `larql vindex3 plan` against this checkpoint is
**inadmissible** (30 blocking findings: 1 mismatched, 29 unrepresented); see
`worktree-qwen35-linear-cfg` for the plumbing pass addressing the text-side
subset.

| Fact | Declared value |
|---|---|
| Layers | 64 total |
| Attention layout | `full_attention_interval: 4` → **48 `linear_attention` layers + 16 `full_attention` layers** (every 4th layer is full) |
| Linear-attention (GDN) dims | `linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`, `linear_value_head_dim: 128`, `linear_num_key_heads: 16`, `linear_num_value_heads: 48` |
| Linear-attention state dtype | `mamba_ssm_dtype: "float32"` — **the SSM/recurrent state is kept in fp32 even though the model's own default dtype is bf16.** This is Qwen's own precision choice, and it corroborates §2.2's "state deserves higher precision than bulk weights" argument directly from the primary source. |
| Attention output gate | `attn_output_gate: true`, `output_gate_type: "swish"` |
| MTP head | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` (drafts share the main embedding table) |
| RoPE | `partial_rotary_factor: 0.25` (declared identically at `text_config` and `text_config.rope_parameters`), `rope_theta: 1e7`, plus M-RoPE (`mrope_interleaved: true`, `mrope_section: [11, 11, 10]`) for the multimodal position scheme |
| Vision tower | Qwen3-VL-style ViT: `depth: 27`, `hidden_size: 1152`, `num_heads: 16`, `patch_size: 16`, `spatial_merge_size: 2`, `temporal_patch_size: 2`, `deepstack_visual_indexes: []`. Execution-surface build currently fails outright (`hidden 1152 not divisible by 0 heads`) — **out of scope for this note and for R2**; tracked separately, no ETA. |

---

## 2. What this implies for VINDEX3

### 2.1 Decode / verify / prefill are different *operations*, not one matmul at different M

Ordinary decode is `W × vector` (M=1). An MTP verify step is `W × [tok₁ tok₂
tok₃ tok₄...]` — small-M matmul, not matrix-vector, with a materially
different cost profile per shape and hardware. That's exactly the shape of
thing an `OperationPlan` is supposed to capture — `DecodeMatMul(M=1)` /
`VerifyMatMul(M=2..8)` / `PrefillMatMul(M≫1)` as distinct lowerings of the
same logical operand, each free to pick its own representation and kernel.
Whether this becomes real VINDEX3 surface is an R2/P1 design question, not
decided here.

### 2.2 Precision should be able to follow semantic authority, not just a global `--q4`

`mamba_ssm_dtype: "float32"` (§1) is Qwen's own admission that GDN state is
precision-sensitive — the recurrence carries error forward across the whole
sequence, so it doesn't tolerate the same rounding a one-shot bulk-MLP
weight does. This is a natural extension of the
`representable`/`unrepresented`/`mismatched` vocabulary `vindex3 plan`
already has — from *representability* to *precision selection* — via
properties like "authoritative", "state-amplifying", "verification-only",
"reconstructible" on an operand. (The MTP head is the clean example on the
other end: it's draft-only and can't alter the accepted output, so it's a
natural candidate for aggressive precision regardless of what the rest of
the model needs.) Not proposed as a concrete schema change here; flagged as
the natural next question once R2 actually has a hybrid architecture to
design against.

### 2.3 Keep two different benchmarking questions separate

"Fastest absolute tok/s" and "speedup from speculative/MTP execution at a
*fixed* precision" are different claims — the second is the one with no
inherent quality trade, since an MTP draft that gets rejected costs time,
not correctness, as long as verification is exact. Recommend, when R2
produces real numbers: report a same-precision AR-vs-MTP comparison and a
separate representation/precision sweep (PPL delta, KL, top-1 agreement,
bytes, tok/s) rather than one blended tok/s headline that conflates the two.

---

## 3. Relationship to current work

- **This cycle's actual code change** is `worktree-qwen35-linear-cfg`
  (`6edab530`): the 12 hybrid-attention/gate/MTP/mRoPE fields in §1's table
  now parse into `ModelConfig`, and `layer_types` no longer *fabricates* a
  pass — it previously resolved every `linear_attention` layer to
  `full_attention` and reported "declared interleave honoured" regardless;
  it now correctly reports `mismatched`. No compute, no vision,
  config-plumbing only. **`vindex3 plan`'s blocking count is unchanged at
  30/30**, independently re-verified against the real checkpoint — the goal
  was turning silent gaps into labeled, honest ones, not passing the gate
  outright, and that's what happened: 12 fields moved from silent `unknown`
  to named `execution_semantic` blocks, `layer_types` moved from a false
  `representable` to an honest `mismatched`, and `full_attention_interval`
  dropped off the blocking list as a recognized alias. Along the way the
  pass surfaced (but did not fix — out of scope) a separate real gap: MTP's
  non-`fc` tensor groups (`mtp.layers`, `mtp.norm`,
  `mtp.pre_fc_norm_{hidden,embedding}`) are currently silently absorbed
  into the main decoder's tensor groups by generic substring
  classification, architecture-wide, not Qwen-specific. That's a
  prerequisite for any of §2 being buildable, not an alternative to it.
- **R2/P1's own first task** — "verify a reference implementation runs on
  Apple Silicon before any adapter work" — now has two candidate
  architectures instead of one (Kimi Linear, Qwen3.8), which is useful:
  designing the linear-attention/MTP abstraction against a single model
  risks overfitting the abstraction to that model's specific quirks.
- **Recommended experiment sequence once R2 actually starts:** exact AR
  parity → per-shape M=1..8 roofline for the dominant linear layers →
  native MTP → acceptance-depth measurement → theoretical vs. actual
  bytes/token → decompose any remaining gap into "catch-up engineering"
  (kernel/MTP/representation/scheduling multipliers) versus "we're moving
  more bytes than semantically required" (the actual LARQL research
  question). Don't set the initial success criterion at a tok/s number; set
  it at whether the gap decomposes cleanly.

## 4. Explicitly not claimed here

Not claimed: that any of §2 is a committed design, only a question R2
should design against. Not claimed: that vision-tower support, MTP compute,
or GDN kernels are scheduled — all remain explicitly deferred to real R2
work with a reference implementation to verify against.
