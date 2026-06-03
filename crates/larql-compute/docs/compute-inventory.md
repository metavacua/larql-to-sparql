# Compute Kernel Portability Inventory

Tracks which kernels in `src/cpu/ops/` and `src/metal/shaders/` fall into which
portability class. The classification is at the **algorithm level** — it describes
what a correct GPU port *would* look like, not the current Rust implementation.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | portable as-is to this class / target |
| ⚠️ | portable in principle but requires work (see notes) |
| ❌ | not portable — fundamental constraint |

**¬L∧¬M class** = sub-Turing fragment: no recursion (¬L) and no unbounded heap
growth (¬M). Verified at binary level by `larql-wasm-certify --strict`. The current
MVV cdylib is wasm-safe but NOT ¬L∧¬M due to dlmalloc (M) and serde_json (L) in
its reachable closure.

**spirv candidate** = can this algorithm be expressed as a WGSL/SPIR-V compute
shader? Requires static workgroup dimensions and no OS-level imports. A ⚠️ means
"needs barrier op (reduction pass)" or "needs MSL→WGSL translation."

**nvptx candidate** = can this algorithm be expressed as a CUDA kernel? Similar
constraints to spirv; CUDA allows more dynamic memory but still bans host calls.

---

## CPU ops (`src/cpu/ops/`)

| kernel | source | ¬L∧¬M class | spirv candidate | nvptx candidate | notes |
|--------|--------|:-----------:|:---------------:|:---------------:|-------|
| `geglu_silu` (in-place) | `geglu.rs` | ✅ | ✅ | ✅ | Pointwise SiLU×up; bounded loop; no alloc |
| `geglu_silu_alloc` | `geglu.rs` | ⚠️ | ✅ | ✅ | Alloc form uses `Vec`; in-place form is the portable kernel |
| `causal_attention` | `attention.rs` | ⚠️ | ⚠️ | ⚠️ | Two-pass softmax: global L1 max-scan then normalize; causal mask breaks Toeplitz — **Phase 3 seam** |
| `matmul`, `matmul_transb` | `f32_matmul.rs` | ❌ | ✅ | ✅ | BLAS/ndarray dep (current impl); algorithm is GPU-native if rewritten in WGSL/CUDA |
| `dot`, `norm`, `cosine` | `vector.rs` | ⚠️ | ✅ | ✅ | ndarray/BLAS (current impl); algorithms are ¬L∧¬M-expressible as pure arithmetic |
| `cholesky`, `cholesky_solve` | `linalg.rs` | ❌ | ❌ | ❌ | Dynamic ndarray; sequential dependency chain (L[[i,j]] depends on L[[i,k]] for k<j) |
| `outer_post_norm_residual` | `outer_combine.rs` | ⚠️ | ⚠️ | ⚠️ | RMS norm = L2 reduction over hidden dim; elementwise residual add is ✅ |
| Q4/Q8 matvec (all variants) | `q4_matvec.rs`, `q8_matvec.rs`, `q4k_matvec.rs`, `q4_vecmat.rs` | ❌ | ❌ | ❌ | C FFI (`q4_0_matvec_c`); ARM SIMD intrinsics — not portable across ABIs |
| MoE dispatch | `moe/*.rs` | ❌ | ❌ | ❌ | Rayon + dynamic expert dispatch + OS-level parallelism |

---

## Metal shader algorithms (`src/metal/shaders/`)

These are Metal Shading Language (MSL) kernels expressed as Rust `metal-rs`
wrappers. "spirv candidate" means the *algorithm* maps to WGSL/SPIR-V; the
current MSL implementation would need to be re-expressed in WGSL.

| kernel family | representative files | ¬L∧¬M class | spirv/WGSL | nvptx/CUDA | notes |
|---------------|----------------------|:-----------:|:----------:|:----------:|-------|
| RoPE rotation | `rope.rs`, `qk_norm_rope_fused.rs` | ✅ | ✅ | ✅ | Per-element angle rotation; zero inter-element dependency |
| Elementwise activations | `activation.rs`, `geglu.rs` | ✅ | ✅ | ✅ | Pointwise; no reduction |
| Elementwise residual | `residual_inject.rs`, `post_attn_residual_norm_store.rs`, `post_ffn_norm_residual_add.rs` | ✅ | ✅ | ✅ | Elementwise add/scale; trivially parallel |
| QKV projection | `q4k_qkv_proj.rs`, `q4kf_qkv_proj.rs`, `q4k_q6k_qkv_proj.rs` | ✅ | ⚠️ | ✅ | Each output row is an independent dot product; ⚠️ = MSL→WGSL rewrite |
| Output projection | `q8_attn_proj.rs` | ✅ | ⚠️ | ✅ | Same as QKV; independent output rows |
| FFN gate+up | `q4k_ffn_gate_up.rs`, `q4kf_ffn_gate_up.rs`, `q4k_ffn_gate_up_8sg.rs`, `q4k_ffn_gate_up_coop.rs`, `q4k_ffn_gate_up_f16acc.rs` | ✅ | ⚠️ | ✅ | Paired matvec; each output element independent |
| FFN down (GEGLU) | `q4k_geglu_down.rs`, `q6k_geglu_down.rs`, `q6k_geglu_gelu_tanh_down_cached.rs` | ✅ | ⚠️ | ✅ | Matvec + pointwise activation |
| f32/f16 GEMV | `f32_gemv.rs`, `f16_gemv.rs` | ✅ | ✅ | ✅ | Pure inner product per output element; no barrier |
| Q4K/Q8 matvec | `q4k_matvec.rs`, `q4k_matvec_8sg.rs`, `q4k_matvec_stride32.rs`, `q4_matvec_v4.rs`, `q8_matvec.rs`, `q4_f32_matvec.rs` | ✅ | ⚠️ | ✅ | Quantized inner product; output rows independent; ⚠️ = dequant format is MSL-specific |
| Quantization | `turboquant_encode.rs`, `turboquant_decode.rs`, `quantize_q8.rs` | ✅ | ⚠️ | ✅ | Per-block amax scan (local reduction over 32 elements); otherwise pointwise |
| SGEMM | `sgemm.rs`, `sgemm_transb.rs`, `q4k_matmul.rs` | ⚠️ | ✅ | ✅ | Full matmul via threadgroup tiling; ⚠️ on ¬L∧¬M because alloc/blocking |
| QK norm | `qk_norm.rs`, `v_norm.rs` | ⚠️ | ⚠️ | ⚠️ | L2 norm reduction over head_dim; needs barrier + two passes |
| Layer/RMS norm | `layer_norm.rs` | ⚠️ | ⚠️ | ⚠️ | Global L2 reduction over hidden dim then normalize — same class as softmax seam |
| KV attention | `kv_attention.rs`, `kv_append_attend_fused.rs` | ⚠️ | ⚠️ | ⚠️ | Softmax over KV sequence length — **Phase 3 seam** |
| Causal / fused attention | `causal_attention.rs`, `attn_fused.rs`, `fused_attention.rs` | ⚠️ | ⚠️ | ⚠️ | Softmax global L1 reduction (causal mask breaks Toeplitz) — **Phase 3 seam** |
| Graph KNN walk | `graph_walk_knn.rs` | ❌ | ❌ | ❌ | Dynamic graph traversal; data-dependent branching and pointer chasing |

---

## Auto-parallelizable set (¬L∧¬M ∧ spirv ∧ nvptx = ✅/✅/✅)

Kernels marked ✅/✅/✅ form the auto-parallelizable set — no barrier operations,
no dynamic dispatch, no host imports. A correct WGSL/CUDA port is mechanical.

**CPU ops:** `geglu_silu` (in-place form only)

**Metal shader families:** RoPE, activations, elementwise residual, f32/f16 GEMV

**With WGSL rewrite (✅/⚠️/✅):** QKV projection, output projection, FFN gate+up,
FFN down, Q4K/Q8 matvec families

---

## Phase 3 seam — attention parallelizability frontier

The kernels marked ⚠️/⚠️/⚠️ with note "Phase 3 seam" share a common structure:
**softmax global reduction**. The L1 normalization denominator `sum(exp(score_i))`
requires reading all scores before writing any output. Under a causal mask the
scores are indexed `i ≤ j`, which breaks Toeplitz structure and prevents the
score matrix from being expressed as a convolution or stationary kernel.

This is the exact boundary documented in `docs/adr/0014-attention-parallelizability-seam.md`
(Phase 3). Kernels on the left of this seam are auto-parallelizable; kernels on
the right require either:
- A split-reduce tiling pass (Flash Attention style), or
- Replacement of L1 softmax with L2 normalization (Born Rule / Zizzi Lq — Phase 4).
