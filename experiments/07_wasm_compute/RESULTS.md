# Experiment 07 — Results

**Chris Hay · LARQL Project · Apr 2026**

Consolidated findings across the token-level, residual-level, VM, AOT,
and production Tier-2 sub-experiments, plus the Rust-native host path
that graduated to the `model-compute` crate.

For original per-phase write-ups with full methodology, see [`archive/`](archive/).

---

## TL;DR

- **Phase 1 (token-level dispatch) failed.** Mid-generation solver
  injection corrupts the KV cache; post-hoc correction is vacuous on
  models that already compute arithmetic correctly (Gemma 3 4B on GSM8K).
  Superseded by residual-level compilation.
- **Residual-level compilation works at production scale.** One compiled
  edge fixes Gemma 3 4B's wrong answer to sum(1..100) (4050 → 5050), zero
  drift through save/load, other questions unaffected. First LARQL
  compilation on a real instruction-tuned model.
- **Straight-line programs up to depth 3 and 8-layer spans run on v11**
  at cos > 0.99 per hop. Gemma needs refresh edges after ~3 hops (measured
  0.10 → 0.62 recovery with one relay).
- **Conditional dispatch works** via a nonlinear WASM gate (D3) or a
  properly-chosen linear direction (D2). It doesn't work via naive frozen
  dot-product gates when the last-token residual doesn't discriminate.
- **Programmable VM (8 opcodes) runs in the residual stream**, including
  cross-layer assembly (source-blind gates) and conditional branching
  (CMP + JMP_IF).
- **AOT compilation of VM programs to static weight edges produces
  zero-drift checkpoints** that load in standard transformers / Ollama.
- **Rust-native host path shipped** as `model-compute::wasm`. Loads the
  same 26 KB CP-SAT `.wasm` the Python demo used; ~290 ms compile,
  ~0.2 ms per solve call.

---

## 1. Phase 1 — token-level dispatch (archived, failed)

| Approach | Outcome |
|---|---|
| Mid-generation injection (v1) | **Failed.** Triggers on partial expressions; KV cache corruption breaks model coherence |
| Post-hoc correction (v2) | Mechanically correct; vacuous on GSM8K because Gemma's arithmetic is already right |

**Lesson:** solvers add *capability* (algebraic simplification, constraint
satisfaction), not correction to broken arithmetic. Stopped pursuing
token-level dispatch as the primary path. See
`archive/RESULTS.md` for detailed error analysis.

---

## 2. Residual persistence

Write a unit direction at layer L; measure cosine with FFN input at L+k.

| Hops | v11 cos | Gemma 3 4B cos |
|---|---|---|
| 1 | 0.994 | 0.471 |
| 2 | 0.995 | 0.357 |
| 5 | 0.996 | 0.134 |
| 8 | 0.997 | 0.044 |

Gemma's attention mixes residuals ~87× more aggressively per layer than
v11's. Both substrates support the stack-machine primitive; depth budgets
differ. A single relay edge (trigger=tag, write=tag) at Gemma L15
recovers cos **0.10 → 0.62** at L16 — refresh is validated as the
mechanism for deeper programs on production models.

Data: `persistence_sweep.py`.

---

## 3. Straight-line WASM programs (v11)

| Program | Depth | Result | Logit |
|---|---|---|---|
| `(2+3) × 2 = 10` | 2 (L10 → L11) | top-1 `'ten'` | +26.0 |
| `(1+1) × 2 × 2 = 8` | 3 (L10-12) | top-1 `'eight'` | +21.7 |
| DOUBLE swept L11 → L19 after ADD at L10 | non-adjacent, 8 layers | top-1 `'ten'` through L18 | +27.9 |

Per-hop cos ~0.99 lets a written direction stay readable for effectively
the rest of the network. Multi-op programs don't need tight layer
packing.

---

## 4. Conditional dispatch

Two structurally-similar prompts with identical templates and different
operators produce nearly-identical L10 residuals (cos 0.995) on v11. A
naive dot-product gate can't discriminate.

| Fix | Outcome |
|---|---|
| Pick the right direction (D2: `diff_unit` between residuals) | Works — frozen gate cleanly fires for one prompt, not the other |
| WASM gate (D3): nonlinear discriminant in Python, branches, writes final answer directly | Works — collapses the two-layer four-edge program to one slot |

The stack-machine primitive isn't the issue; *dispatch* is. Detailed in
`archive/WASM_IN_FFN_RESULTS.md`.

---

## 5. Hybrid chain — synthetic-tag ABI

Three-stage chain (L10 WASM → L12 frozen → L14 frozen):

| Regime | Prompt X (sum=3) | Prompt Y (sum=4) |
|---|---|---|
| Baseline | `'s'` | `'s'` |
| WASM only | `'three'` ✓ | `'s'` ✓ (no tag) |
| WASM + L12 relay | `'six'` ✓ | `'s'` ✓ |
| Full 3-stage | `twelve = +45.1` ✓ | `'s'` ✓ (chain halted) |
| Neg-ctrl (no WASM) | `twelve = −3.9` ✓ | `twelve = −3.9` ✓ |

Using real token embeddings as triggers (v1) fails — natural residual
content accidentally fires the frozen edges. Using Gram-Schmidt-
orthogonalised synthetic tags (v2) gives clean gating. This is the
calling convention for multi-stage WASM/frozen chains. Detail in
`archive/WASM_GATE_ARCHITECTURE.md` §2.4.

---

## 6. Programmable VM (ISA)

Eight opcodes (LOAD_IMM, LOAD, STORE, ADD, MUL, CMP, JMP_IF, HALT) encoded
on orthogonal tag directions.

| Stage | Test | Result |
|---|---|---|
| 1 | Hand-loaded `2 × 3 = 6` program | ✅ passes |
| 2 | Cross-layer assembly — three hooks at L6/L8/L10 contributing one instruction each, single execute gate at L12 | ✅ source-blind, passes |
| 3 | Control flow — CMP + JMP_IF (r0=3 skips STORE, r0=7 executes) | ✅ data-dependent paths work |

Detail in `archive/VM_RESULTS.md`.

---

## 7. AOT compilation to static weights

Three passes, all zero floating-point drift between runtime install and
loaded checkpoint:

| Pass | Test | Result |
|---|---|---|
| 1 | Single edge round-trip (install → save → load → match) | ✅ 0.000000 logit diff |
| 2 | 5-instruction VM program constant-folded to 1 compiled edge | ✅ +11.6 logit on answer token |
| 3 | CMP + JMP_IF branches resolved at build time (0 or 1 edge per branch) | ✅ correct branch selection |

Output is standard safetensors — no runtime hooks, no sidecar files, no
LARQL dependency at inference. Detail in `archive/AOT_RESULTS.md`.

---

## 8. Tier 2 — first compilation on Gemma 3 4B Instruct

Standard production model gets the Gauss sum wrong.

| Test | Vanilla Gemma | Compiled checkpoint |
|---|---|---|
| Sum of 1..100 | **4050** ✗ | **5050** ✓ |
| 47 + 28 | 75 ✓ | 75 ✓ |
| 15 × 13 | 195 ✓ | 195 ✓ |
| Capital of France | Paris ✓ | Paris ✓ |

Three weight rows modified at L15 slot 9000: `gate_proj.weight[9000]`,
`up_proj.weight[9000]`, `down_proj.weight[:, 9000]`. Other 8.6 GB of
the checkpoint unchanged. Saves and reloads through
`AutoModelForCausalLM.from_pretrained()` with zero drift. GGUF
conversion preserves the edge; Ollama loads it as a standard model.
Detail in `archive/TIER2_RESULTS.md`.

---

## 9. Production landing in `model-compute`

The experimental code graduated into the workspace:

| Ex-experiment concept | Production location | Tests |
|---|---|---|
| `install_edge` primitive | `compile_cmd/edge.rs` | 7 |
| `solver/src/lib.rs` CP-SAT guest | Same location, still standalone | via examples |
| Python wasmtime host in `wasm_solver_demo_v11.py` | `model-compute::wasm::SolverRuntime` | 8 integration + 3 benches |
| Expression evaluation (for AOT) | `model-compute::native::{arithmetic, datetime}` | 30 |
| End-to-end CP-SAT demo | `model-compute/examples/cpsat_scheduling.rs` | runnable |

Wasmtime benchmark on M3 Max:
- Module compile: ~290 ms (one-time)
- Session instantiate: ~4 µs (per call)
- Round-trip solve on echo fixture: ~8 µs
- CP-SAT 5-task scheduling solve: ~0.15 ms, optimal makespan=4

For context: a Gemma 3 4B forward pass is ~25–50 ms/token on the same
hardware, so WASM dispatch overhead is <0.05%. Fuel-cap overhead is
already baked into these numbers. Embedding a solver in the forward pass
is not bounded by dispatch cost; the solver's own work dominates.

---

## 10. Bugs found while porting

While porting `wasm_solver_demo_v11.py` to Rust, uncovered a dead-end
handling bug in `solver/src/lib.rs`:

> `SolverState::solve()` treated `select_variable() → None` as
> unconditional success, without distinguishing "all assigned" from
> "unassigned var has an empty domain after propagation". On
> minimize-max problems with all-different constraints, this caused
> `target_max = 0` to be reported as an optimal solution (status 2)
> with an assignment of all zeros.

Fixed at `solver/src/lib.rs:338`; `.wasm` rebuilt. Documented in the
archived `WASM_IN_FFN_RESULTS.md` addendum and in today's code.

---

## 11. What we know now that we didn't at SPEC time

- **Tiers 2/3 are production.** The "compile an answer into weights"
  pattern works on real instruction-tuned models with zero drift.
- **Tier 1 splits by input determinism.** Bounded native kernels handle
  runtime-variable inputs with deterministic cost; WASM modules handle
  the rest (sandboxed, fuel-capped). Bundled into one crate with feature
  flags.
- **Substrate matters.** v11 and Gemma have different persistence
  profiles. Depth budget and refresh-edge placement are per-substrate
  properties.
- **Dispatch, not primitive, is the hard part.** The stack-machine
  mechanism always works; what fails is the linear gate's ability to
  discriminate structurally-similar prompts. Fix: better direction
  choice or a nonlinear gate.
- **Running a WASM solver in a forward pass is cheap** (~8 µs dispatch
  overhead). The research question is no longer "is the overhead
  tolerable" but "does the solver's output usefully change the model's
  answer" — which Tier 2 already demonstrates for constant-folded cases.

See [`ROADMAP.md`](ROADMAP.md) for what comes next.
