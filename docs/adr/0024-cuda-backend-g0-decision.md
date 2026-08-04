# ADR-0024: G0 — CUDA-native backend via cudarc + NVRTC (vs wgpu/Vulkan)

**Status:** accepted 2026-07-24
**Context:** DEC funnel v0.5 §4 G-ladder. The DEC-5 hero demo and any NVIDIA
fleet claim need a CUDA attention client; G0 is the time-boxed backend
decision. Three inputs were investigated before deciding (registry-grade
evidence in the G0 investigation, 2026-07-24).

## Inputs

**1. The prior CUDA PR is not the head start the ROADMAP implied.**
PR #53 (`bkearns/larql@0e2b5fb4`, closed 2026-05-08, unmerged, fork-only —
not reachable from this clone) implemented exactly one GPU primitive:
cuBLAS f32 SGEMV over CPU-dequantised weights, via hand-rolled CUDA Driver
API FFI + `dlopen`'d cuBLAS (deliberately no nvcc/SDK dependency).
`DecodeBackend` was a stub; no custom kernels, no attention math, no norms.
Its useful legacy is the **no-SDK runtime-linking pattern**, not kernels.
The crate ROADMAP's "re-land from earlier PR" entry is corrected by this ADR.

**2. The port is bounded: ~24 of ~48 Metal shader files.** The client-slice
decode/prefill path (G1+G2, no MoE — MoE runs CPU-side even on Metal) needs
~22 core kernels + 2 arch/LM-head files. Three porting gotchas, recorded so
G1/G2 estimates survive contact:
- **The Metal Q4K path is f32 dequant-and-FMA, not int8.** No packed-dot
  intrinsics exist anywhere in the shader set (Metal has no `dp4a`). The
  int8 Q4K×Q8K discipline lives only on the CPU side. A `__dp4a` CUDA port
  therefore *diverges numerically from the Metal reference*.
- **The fused attention kernels use a cross-threadgroup idempotent KV-write
  trick** (GQA Q-head TGs redundantly write the same K/V row behind a
  `mem_device` barrier) that must be reproduced or redesigned on CUDA.
- **Both fast and fallback SDPA paths are required for correctness**
  (fused kernels are gated to short spans + single-simdgroup head_dim;
  Gemma 4 global layers past 1024 tokens take the unfused path). There is
  no fine-grained Metal SDPA behind `KvDispatch` to copy — the fine intents
  delegate to CPU; only the coarse fused path is GPU.

**3. wgpu is no longer disqualified on arithmetic, but the tiebreaker is
control.** wgpu v26 ships `dot4I8Packed`/`dot4U8Packed` with native
intrinsics on Vulkan (`VK_KHR_shader_integer_dot_product`; NVIDIA supports
it) and polyfills elsewhere. However (a) the reference semantics to port
are f32-FMA anyway (input 2), so the int8 question is a *phase-2* concern;
(b) this project's measured history is that scheduling/asm-level control
pays real margins on quantised inner loops (CPU: hand-asm beat intrinsics;
Metal: dispatch geometry was worth 4×) — PTX-level control is the CUDA
analogue; (c) the portability argument is weak here: the Metal backend
already exists and stays, so wgpu's "one dialect" would be a third dialect,
not a consolidation.

## Decision

**CUDA-native, as sibling crate `larql-compute-cuda` on the ADR-019
template, using `cudarc` with runtime-compiled kernels (NVRTC) and
dynamic loading.**

- **Crate shape:** mirror `larql-compute-metal` exactly — trait crate
  untouched (the backend factory takes one registered constructor);
  `#[cfg(target_os = "linux")]`-gated body; `trait_impl/` split;
  own coverage-policy.json.
- **Kernel strategy:** embedded CUDA C source strings compiled at runtime
  via NVRTC — the direct analogue of Metal's embedded-MSL-compiled-at-
  runtime pattern, and it preserves PR #53's no-SDK property (cudarc's
  dynamic-loading feature; no nvcc at build time, works on marketplace
  boxes with only the driver present).
- **Phase 1 (G1/G2): port the f32-FMA reference semantics as-is.** Same
  math as Metal → the shannon-verify ≤0.5% bits/char gate (G3) compares
  like with like, and parity failures localise to the port, not to a
  simultaneous numerics change.
- **Phase 2 (post-G3, optional, perf):** `__dp4a` int8 fast-path kernels as
  an opt-in flag, parity-gated against the f32 path — the same discipline
  as `LARQL_Q4K_ASM` on CPU (opt-in, bit-comparison-tested, default only
  after e2e A/B).

## Revisit condition

Adopt/port to wgpu if and when a non-CUDA GPU target (AMD/Intel via Vulkan,
or browser) becomes a funnel requirement — at which point the f32-FMA
kernel sources translate mechanically and `dot4I8Packed` covers the phase-2
path. Nothing in this decision forecloses it; the expensive asset (kernel
semantics + parity harness) transfers.

## Consequences

- G1 scope is now a named list (~24 files) with three pre-registered
  hazards; estimates and progress track against it.
- The crate ROADMAP's "CUDA backend (re-land from earlier PR)" entry is
  superseded: re-landing PR #53 would deliver a dense-GEMV-only backend.
  Its dlopen pattern is inherited via cudarc's dynamic loading instead.
- G3 (shannon-verify CI on the CUDA path) is unchanged and remains the
  DEC-5 gate.
