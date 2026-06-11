# Linear Attention: ComplexLinearAttn vs Gated DeltaNet

Two linear-attention implementations exist in this project ecosystem. This document
records their mathematical relationship, where each lives, and what they imply for
the wasm32v1-none minimal-model constraint.

---

## The shared motivation: replacing the unbounded KV cache

Standard softmax attention requires an O(n · L · d) KV cache that grows without
bound as context length n increases. For wasm32v1-none inference — no memory.grow,
bounded WebAssembly.Memory — this is a hard constraint violation at any non-trivial
context length.

Both implementations below replace that unbounded cache with a **fixed-size state**
that does not grow with n.

---

## ComplexLinearAttn — this crate (`src/attention/linear_complex.rs`)

**Kernel:** φ(x) = exp(ix) — complex unit-circle embedding of each head-dim
coordinate.

**Per-token accumulation (prefill or decode):**

```
KtV[i, j] += φ(k_t)[i]* · v_t[j]          (outer product, d×d complex accumulator)
o_t[j] = Re[ Σ_i φ(q_t)[i] · KtV[i, j] ] / n
```

**State:** one `d × d` complex accumulator per attention head —
O(n_heads · head_dim²) total, fixed regardless of sequence length.

**Properties:**
- Pure sum of outer products; no learned decay or write gate
- Infinite memory (all past tokens contribute equally — no forgetting)
- Prefill is embarrassingly parallel (the sum is commutative)
- Applies to the Q/K/V weight matrices of **any** existing transformer;
  no architecture-specific tensors required
- O(n · d²) total compute; 7.9× faster than O(n² · d) at n=256 (see `benches/complexity.rs`)
- No `KvCache` dependency — the accumulator is the entire context state

---

## Gated DeltaNet — `larql-ian-wt` (`crates/larql-inference/src/attention/deltanet_recurrence.rs`)

**Architecture:** Qwen3.5/3.6 hybrid models (NVlabs, arXiv 2412.06464).
48 of 64 layers use DeltaNet; 16 use standard softmax attention (at every 4th layer).
Only the 16 softmax layers maintain a `KvCache`; the 48 DeltaNet layers maintain
a `DeltaNetStateCache` instead.

**Per-token recurrence (decode only — inherently sequential):**

```
S_t ← g_t · S_{t-1} + β_t · (v_t − S_{t-1} · k_t) ⊗ k_t     (delta rule update)
o_t = S_t · q_t
```

where `g_t ∈ (0,1)` is a per-head learned decay, and `β_t ∈ (0,1)` is a
per-head learned write rate (projected from the residual stream, not fixed).

**State:** `[S_v × S_v × H_v]` matrix per DeltaNet layer —
`[128 × 128 × 48]` for Qwen3.6-27B → 786,432 f32 per layer, fixed.

**Properties:**
- **Finite memory** via learned per-head decay g — older tokens are exponentially
  down-weighted; the model can "forget"
- The correction term `v_t − S·k_t` makes the update error-driven (like a Hebbian
  delta rule), which is why it handles long contexts (256K tokens in Qwen3.6)
  without attention score collapse
- Requires architecture-specific tensors (`ssm_a`, `ssm_beta`, `ssm_alpha`,
  `ssm_conv1d`) — not applicable to generic transformer weights
- Prefill requires the sequential recurrence; no parallel prefix-sum shortcut
  without additional chunked-parallel implementations
- The 16 full-attention layers still use a KV cache, so the model is not
  fully KV-cache-free

---

## Mathematical relationship

Both are instances of the general linear-attention form:

```
o_t = M_t · φ_Q(q_t),   where  M_t accumulates keys and values
```

| Property | ComplexLinearAttn | Gated DeltaNet |
|---|---|---|
| φ_K, φ_Q | exp(i·) | L2-norm (unit sphere) |
| State update | M_t = M_{t-1} + φ(k_t)* ⊗ v_t | M_t = g·M_{t-1} + β·(v_t − M_{t-1}·k_t)⊗k_t |
| Decay | None (infinite memory) | Learned per-head g ∈ (0,1) |
| Write gate | Fixed (always write) | Learned β ∈ (0,1) |
| Architecture dependency | None | Requires Qwen3.5/3.6 tensors |
| Prefill parallelism | Full (commutative sum) | Sequential (recurrence) |
| State size | n_heads · head_dim² | 128² · H_v per DeltaNet layer |
| wasm32v1-none | Yes — fixed state, no std deps | Partial — 16/64 layers still need KvCache |

For the wasm32v1-none minimal-model target, ComplexLinearAttn is the cleaner fit:
no KV cache at any layer, no architecture-specific tensors, applies directly to
the weights in an existing vindex. DeltaNet achieves better language-modelling
quality for long contexts but requires the Qwen3.5/3.6 architecture and retains
a partial KV cache.

---

## The KV-cache vs residual stream discovery

A central finding of the LARQL project is that the **KV cache and the residual
stream are not the same thing**, and conflating them is the source of several
architectural errors.

- **KV cache**: an implementation artifact of efficient softmax attention. Stores
  past key/value projections per layer so they need not be recomputed during
  autoregressive decode. Grows as O(n · L · d) with context length. Contains no
  information that isn't already encoded in the residual stream — it is a
  derivative of it, not an independent store.

- **Residual stream**: the sequence of hidden states `h_l ∈ ℝ^d` that flows
  through the transformer's layers. The *primary* computational object. Features
  extracted by the vindex (gate vectors, down vectors) are residual-stream
  entities — a gate vector fires when the residual stream projects onto it.
  Fixed size O(d) per position per layer; does not grow with context.

The LARQL vindex is a **residual-stream database**, not a KV store. The Walk FFN
works because it looks up which features the current residual state activates,
then retrieves their output contributions — this is entirely a residual-stream
operation. The attention KV cache is orthogonal to this.

Linear attention (both ComplexLinearAttn and DeltaNet) makes the distinction
explicit: the attention state is now a finite summary computed from the residual
stream, not an unbounded list of cached key/value vectors. For wasm32v1-none, the
right design is residual-stream inference with a fixed-size attention summary —
which is precisely what this crate implements.

---

## Rebase note

PR #74 (`lm-synthesis/toolchain`) was developed against an earlier base.
The chrishayuk/larql upstream has since advanced significantly with major
architectural improvements (substrate/engine split ADR-0022, Metal first-class
peer, KvDispatch/AsyncComputeBackend traits) and bug fixes (norm epsilon loading,
rope_scaling structured form, llama3 wavelength-dependent inv_freq, StarCoder2
norm alias). The synthesis work in this PR should be rebased onto the
chrishayuk/larql upstream before the WASM inference path is completed.
