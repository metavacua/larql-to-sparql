# model-compute

Bounded-cost compute primitives for neural-model pipelines. Two modes,
pick with Cargo features:

| Feature | Module | Purpose | Weight |
|---|---|---|---|
| `native` (default) | `model_compute::native` | Deterministic Rust kernels — arithmetic, datetime | 3 deps |
| `wasm` | `model_compute::wasm` | Wasmtime-hosted WASM modules with fuel/memory caps | +wasmtime |

Both share the conceptual model of "bounded-cost input → output
computation." The difference is where the computation lives: native
kernels compile into your binary; WASM modules load at runtime through
a sandbox. Native is cheaper and tighter; WASM is for things that don't
fit as stdlib Rust (CP-SAT solvers, symbolic algebra, SMT).

## Portable

Named `model-*` rather than `larql-*` — intentionally has no LARQL
dependency. Currently lives in the larql mono-repo for iteration; will
extract to a sibling repo once the interface stabilises. Intended to be
equally useful in TinyModel and other neural-model-compiler projects.

## Native kernels (default)

```toml
[dependencies]
model-compute = "0.1"   # features = ["native"] by default
```

```rust
use model_compute::native::{Kernel, KernelRegistry};

let registry = KernelRegistry::with_defaults();
assert_eq!(registry.invoke("arithmetic", "sum(1..101)")?, "5050");
assert_eq!(registry.invoke("datetime", "weekday(2026-04-16)")?, "Thu");
```

| Kernel | Syntax | Output |
|---|---|---|
| `arithmetic` | `sum(1..101)` | `"5050"` |
| `arithmetic` | `math::pow(2.0, 10.0)` | `"1024"` |
| `arithmetic` | `factorial(10)` | `"3628800"` |
| `datetime` | `days_between(2026-01-01, 2026-04-16)` | `"105"` |
| `datetime` | `weekday(2026-04-16)` | `"Thu"` |
| `datetime` | `add_days(2026-04-16, 7)` | `"2026-04-23"` |

Run the demo:

```
cargo run --example gauss -p model-compute
```

Bounded guarantees: deterministic, pure, hard-capped cost (ranges ≤ 10⁸
iterations, factorial ≤ 20). No panics on adversarial input.

## WASM modules (opt-in)

```toml
[dependencies]
model-compute = { version = "0.1", features = ["wasm"] }
```

```rust
use model_compute::wasm::SolverRuntime;

let runtime = SolverRuntime::new()?;
let module = runtime.compile(&wasm_bytes)?;   // precompiled .wasm
let mut session = runtime.session(&module)?;  // fresh Store per call
let output = session.solve(&input_bytes)?;    // alloc → write → solve → read
```

Guest modules implement the canonical ABI:

| Export | Purpose |
|---|---|
| `alloc(u32) -> i32` | reserve input buffer |
| `solve(i32 ptr, u32 len) -> u32` | run compute, return status |
| `solution_ptr() -> i32` | pointer to output |
| `solution_len() -> u32` | length of output |

Every call runs in a fresh `Store` with explicit fuel + memory caps.
Exceeding either errors rather than wedges the host. This is what makes
unbounded-complexity solvers safe to embed.

**End-to-end demo:** the CP-SAT solver from `experiments/07_wasm_compute/solver/`
is a 26 KB constraint solver that runs through `SolverRuntime`:

```
cargo run --example cpsat_scheduling -p model-compute --features wasm
```

Solves a 5-task scheduling problem (all-different over 10 time slots,
minimise max) — optimal makespan = 4, solver returns in ~0.2 ms after
the one-time ~290 ms module compile.

## Pairs with

A weight-editing primitive — when a kernel or solver result needs to be
baked into a model's weights as a compiled edge. The compute crate
resolves the answer (e.g. `sum(1..100)` → `"5050"`); a caller converts
it to a token embedding and writes gate/up/down at a slot.

In the larql mono-repo today the edge-install primitive lives at
`crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs` —
extracted to its own crate only when a second consumer needs it.

## Out of scope

- Model loading, tokenizers, forward pass — those are model-host concerns.
- Forward-pass dispatch — kernels/solvers run at compile time (or as
  explicit calls), not during inference.
