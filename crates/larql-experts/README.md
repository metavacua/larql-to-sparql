# larql-experts

WASM modules that the LARQL inference engine can dispatch to for deterministic
computation. Each expert is a standalone `wasm32-wasip1` `cdylib` exposing a
small, language-neutral JSON ABI.

## What this is

A pluggable registry of sandboxed compute primitives. An LLM (or any caller)
emits a structured call — op name plus a JSON args object — and the registry
routes it to the WASM module that advertises that op. No natural-language
parsing happens inside the experts; prompt → op translation lives upstream.

## The ABI

Every expert exposes exactly two WASM exports:

```c
// Returns a pointer to a null-terminated JSON ExpertResult, or 0 if the expert
// does not handle the requested op.
u32 larql_call(u32 op_ptr, u32 op_len, u32 args_ptr, u32 args_len);

// Returns a pointer to a null-terminated JSON ExpertMetadata.
u32 larql_metadata();
```

Plus the two host-side memory helpers `larql_alloc(len) -> u32` and
`larql_dealloc(ptr, len)` used to write the op / args strings into the module's
linear memory before invoking `larql_call`.

### Result shape

```json
{
  "value": <any JSON value>,
  "confidence": 1.0,
  "latency_ns": 0,
  "expert_id": "arithmetic",
  "op": "gcd"
}
```

### Metadata shape

```json
{
  "id": "arithmetic",
  "tier": 1,
  "description": "Arithmetic, number theory, ...",
  "version": "0.2.0",
  "ops": ["add", "sub", "mul", "div", "pow", "gcd", ...]
}
```

The registry reads `ops` on load and builds an op → expert index. Dispatch is
by op name, not by trying every expert against a prompt.

## Building

From the repo root:

```sh
cd crates/larql-experts
cargo build --target wasm32-wasip1 --release
```

Each expert produces `target/wasm32-wasip1/release/larql_expert_<id>.wasm`.

## Using from Rust (host side)

```rust
use larql_inference::experts::ExpertRegistry;
use serde_json::json;

let dir = std::path::Path::new(
    "crates/larql-experts/target/wasm32-wasip1/release",
);
let mut registry = ExpertRegistry::load_dir(dir)?;

// Structured call — no prompt, no English.
let result = registry
    .call("gcd", &json!({"a": 144, "b": 60}))
    .expect("dispatches");

assert_eq!(result.value, json!(12));
assert_eq!(result.expert_id, "arithmetic");
```

Other useful registry methods:

- `registry.ops()` — every op name the registry can dispatch
- `registry.list()` — metadata for every loaded expert
- `registry.wasm_infos()` — WASM file paths, on-disk bytes, live memory pages,
  and whether the expert is currently instantiated
- `registry.wasm_info_for(id)` — the same, for a single expert by id
- `registry.evict_all()` — drop every live `Store` + `Instance` without
  unloading the compiled modules (useful between long idle periods)

## Load-time + memory behaviour

**Compiled-module cache.** On first load each `.wasm` is compiled by wasmtime
(~30-40 ms per expert). The serialized result is written to a sibling
`.cwasm` file; subsequent loads deserialize that cached artifact instead.
Cold vs. warm load on a 19-expert workspace:

```
cold: Loaded 19 experts in 778ms
warm: Loaded 19 experts in 6ms     # 260× faster
```

The cache is keyed on the `.wasm` file's mtime — rebuilding an expert
invalidates it automatically. If wasmtime is upgraded to an incompatible
version, `Module::deserialize_file` returns an error and the loader falls back
to a fresh compile.

**Lazy instantiation.** `ExpertRegistry::load_dir` compiles (or loads from
cache) every module, but it does *not* keep live `Store` + `Instance` pairs
around. Instantiation happens on the first `call()` per expert, and that
instance is reused for subsequent calls. Net effect: a registry with 19
experts loaded but only 3 ops actually used in a session holds ~3 MiB of live
linear memory instead of ~20 MiB. `evict_all()` resets every expert back to
the no-instance state if you want to shed memory between idle periods —
re-materialising is microsecond-scale (no recompilation).

**Per-call memory stability.** Every `registry.call()` allocates three
buffers inside the module's linear memory (the op name, the args JSON, the
result JSON); the caller frees all three via `larql_dealloc` before
returning. Memory stays flat across millions of calls.

## Writing a new expert

```rust
use expert_interface::{expert_exports, json, Value};

expert_exports!(
    id = "mycrate",
    tier = 1,
    description = "What this expert does",
    version = "0.1.0",
    ops = ["op_a", "op_b"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "op_a" => Some(json!(args.get("x")?.as_f64()? * 2.0)),
        "op_b" => Some(json!(true)),
        _ => None,
    }
}
```

Add an entry to the workspace `Cargo.toml` and a `Cargo.toml` in the expert's
own directory that pulls in `expert-interface` and sets
`crate-type = ["cdylib"]`.

Ops must be language-neutral identifiers (`gcd`, `base64_encode`) and results
must be typed JSON values — not formatted English sentences. If you find
yourself writing `format!("{} is a leap year", n)`, return `true` / `false`
instead and let the caller format.

## Experts in this workspace

| Expert        | Tier | Ops | Summary |
| ------------- | ---: | --: | ------- |
| arithmetic    | 1    | 18  | add/sub/mul/div/pow/mod, gcd/lcm/factorial, primality, base & Roman conversions, percentages |
| conway        | 1    | 2   | Game of Life: single step, N-generation simulation |
| date          | 2    | 6   | Gregorian date arithmetic via Julian day number |
| dijkstra      | 1    | 3   | Shortest path, reachability, minimum spanning tree |
| element       | 1    | 4   | Periodic table lookup by Z, symbol, or IUPAC name |
| finance       | 1    | 9   | Future/present value, interest, mortgage, NPV, Bayes, Kelly, ROI |
| geometry      | 1    | 18  | Areas, perimeters, volumes, Pythagorean theorem |
| graph         | 1    | 6   | Centrality, cycles, components, topo sort, bipartite, degrees |
| hash          | 1    | 7   | Base64, hex, URL percent, FNV-1a 32-bit |
| http_status   | 1    | 1   | IANA HTTP status codes with category |
| isbn          | 1    | 3   | ISBN-10/13 validation and conversion |
| logic         | 1    | 4   | Propositional eval, simplify, truth table, classify |
| luhn          | 1    | 3   | Luhn checksum and card-network detection |
| markov        | 1    | 2   | Expected value, steady state (power method) |
| sql           | 1    | 1   | In-memory SQL over CREATE/INSERT/SELECT + aggregates |
| statistics    | 1    | 11  | Mean, median, mode, stddev, variance, sort, min/max, sum, range, count |
| string_ops    | 1    | 14  | Reverse, palindrome, anagram, caesar/rot13, case, count, match helpers |
| trig          | 1    | 11  | sin/cos/tan/sec/csc/cot + inverses, deg↔rad (angles in radians) |
| unit          | 1    | 3   | Length/mass/temp/volume/speed/energy conversion |

**Totals:** 19 experts, 126 ops, ~1.8 MB of WASM total.

See each expert's `src/lib.rs` module doc for the full op → arg → return type
reference.

## Tiers

Experts advertise a `tier` that the registry uses purely as a sort key at load
time (lower tiers sort earlier). This matters when two experts advertise the
same op name: the lower-tier expert wins, and the higher-tier one is shadowed
for that op. Today every expert is tier 1 except `date` (tier 2).

## Demo + tests

Host-side integration tests and a demo binary live in the `larql-inference`
crate:

```sh
# Build the WASM modules first.
(cd crates/larql-experts && cargo build --target wasm32-wasip1 --release)

# 175 integration tests covering every advertised op.
cargo test -p larql-inference --test test_experts

# End-to-end demo: loads all experts, prints WASM file sizes and live linear
# memory pages, runs 68 structured calls, benchmarks WASM vs native on one op,
# and shows the sandbox containing a division-by-zero.
cargo run -p larql-inference --example experts_demo --release
```

## Design notes

- **No English in the ABI.** Op names, argument keys, and result values are
  language-neutral. A French-language prompt and a Japanese-language prompt
  reach the same expert through the same call.
- **Typed values, not prose.** `is_prime` returns `true` / `false`, not
  `"97 is prime"`. `http_status.lookup` returns `{code, reason, category}`,
  not `"404 Not Found — The requested resource could not be found."`.
- **Experts advertise their ops.** Routing is O(1) by op name after load, not a
  linear scan of "try each expert against the prompt until one matches".
- **WASM execution is real.** Each call crosses a `wasmtime` sandbox — the
  demo prints the file path and live memory pages per module, and the WASM
  vs. native benchmark shows the (non-zero) sandbox overhead.
- **Compile once, instantiate on demand.** The compiled form is cached on
  disk (`.cwasm`), so warm loads are ~260× faster than cold. Instances are
  created lazily, so a large registry that's mostly unused stays small in
  memory.
- **No leaks per call.** The caller pairs every `larql_alloc` (op, args,
  result) with a `larql_dealloc`. Memory pages are stable across long
  sessions; this is covered by an integration test.

## Relationship to `expert-interface`

All shared types (`ExpertResult`, `ExpertMetadata`), helpers (`arg_f64`,
`arg_str`, …), and the `expert_exports!` macro live in the `expert-interface`
crate. That crate is `no_std + alloc` so it compiles into the WASM modules
without dragging in `std`. The experts themselves use `std` (they run on
`wasm32-wasip1`, which ships a `std`).
