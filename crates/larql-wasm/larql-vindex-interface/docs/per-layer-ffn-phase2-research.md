# Per-Layer FFN Phase 2 — research and spec

**Status:** Research only. No code changes proposed at this stage.

**Bottom line:** **Phase 2 is already shipped in code.** The ROADMAP entry
in `crates/larql-vindex/ROADMAP.md` claiming `~120 ms/token allocation
overhead` and `300 buffer allocations per decode token` is stale. The
in-process Metal MoE path and the server expert RPC path both cache
`MoeScratch` by shape; the ~120 ms cost is paid **once at first use**,
not per token. What remains is a benchmark to quantify the steady-state
win and a ROADMAP correction.

---

## 1. The cache machinery (already shipped)

### In-process: `MetalBackend::moe_scratch`

`crates/larql-compute/src/metal/mod.rs:250` declares

```rust
moe_scratch: std::sync::Mutex<Option<moe_dispatch::MoeScratch>>,
```

`MetalBackend::decode_token_q4k_moe` at
`crates/larql-compute/src/metal/moe_dispatch.rs:138-187` holds the lock
across the entire decode and only calls `MoeScratch::new` on a shape
mismatch:

```rust
let mut scratch_guard = self.moe_scratch.lock().unwrap();
if let Some(shape) = layers
    .iter()
    .find_map(|l| l.moe.as_ref())
    .map(|m| (m.top_k, hidden, m.intermediate_size))
{
    let needs_alloc = match scratch_guard.as_ref() {
        Some(s) => (s.top_k, s.hidden, s.inter) != shape,
        None => true,
    };
    if needs_alloc {
        *scratch_guard = Some(MoeScratch::new(&self.bufs, shape.0, shape.1, shape.2));
    }
}
```

The doc comment on lines 163-173 is explicit about the design intent:

> Cache scratch by `(top_k, hidden, intermediate_size)` on the backend so
> the ~15 Metal buffer allocations (~120 ms on Gemma 4 26B-A4B, M3 Max)
> only happen at first use, not per token.

Verification grep confirms the cache is never invalidated:

```
$ grep -rnE 'moe_scratch\.(take|reset|clear)|\*\s*scratch_guard\s*=' crates/larql-compute/src/
crates/larql-compute/src/metal/moe_dispatch.rs:185:  *scratch_guard = Some(MoeScratch::new(&self.bufs, shape.0, shape.1, shape.2));
```

The only assignment is the shape-mismatch reallocation; no `.take()` or
`.clear()` exists anywhere in the crate.

### Server: `AppState::moe_scratches`

`crates/larql-server/src/state.rs:106-108` declares

```rust
pub moe_scratches: std::sync::Mutex<
    std::collections::HashMap<(usize, usize, usize), Arc<larql_compute::MoeScratch>>,
>,
```

`crates/larql-server/src/routes/expert/metal.rs:131-136` looks the
scratch up by shape and inserts on miss:

```rust
let scratch_key = (top_k, hidden, inter);
let mut scratch_cache = model.moe_scratches.lock().expect("moe_scratches poisoned");
let scratch = scratch_cache
    .entry(scratch_key)
    .or_insert_with(|| Arc::new(MoeScratch::new_public(backend, top_k, hidden, inter)));
```

`MoeScratch::new_public` at `moe_dispatch.rs:78-80` is the public-API
wrapper around the same constructor. The server cache survives across
RPC calls; only the first call for a given shape pays the allocation
cost.

## 2. What `MoeScratch` actually pre-allocates

Per `moe_dispatch.rs:82-128`, `MoeScratch::new` allocates **10 buffers
once** at the per-decode shape:

| Buffer | Size formula | Gemma 4 26B A4B (top_k=8, hidden=2560, inter=2112) |
|---|---|---|
| `gate_buf` | `top_k × inter × row_bytes` | ~2.2 MB |
| `up_buf` | `top_k × inter × row_bytes` | ~2.2 MB |
| `down_bufs[0..top_k]` | `top_k × hidden × down_row_bytes` | 8 × ~150 KB = ~1.2 MB |
| `x_buf` | `hidden × 4` | 10 KB |
| `g_out` | `top_k × inter × 4` | ~67 KB |
| `u_out` | `top_k × inter × 4` | ~67 KB |
| `act_buf` | `top_k × inter_padded × 4` | ~74 KB (zero-init at construction) |
| `expert_outs` | `top_k × hidden × 4` | ~80 KB |

Total: ~6 MB held resident. Reused for every decode token, every MoE
layer within the token.

The "~300 allocations per token" framing in the ROADMAP referred to a
pre-cache code state (10 buffers × 30 layers = 300). With the cache
shipped, layers reuse the same scratch and steady-state is **0 buffer
allocations per token**.

## 3. What the per-token decode actually costs

After the first token of a model load, `gpu_moe_dispatch_with_scratch`
(`moe_dispatch.rs:701-...`) does:

1. **CPU pre-experts norm + router pass** (one pass per MoE layer,
   `~hidden²` FLOPs).
2. **`top_k × 2` host→shared-memory `memcpy`s per layer**: gate+up byte
   slice and down byte slice for each of the `top_k` experts that route
   for this token. For Gemma 4 26B A4B: 8 × ~564 KB gate+up + 8 ×
   ~150 KB down = ~5.7 MB / layer / token in memcpy traffic.
3. **One fused gate+up dispatch + `top_k` activation dispatches +
   `top_k` down dispatches**, all in one encoder, committed and waited
   once per layer.

No `device.new_buffer*` calls fire on the steady-state hot path.

## 4. Shape variability — handled

Section 3 of the agent report verified that
`(top_k, hidden, intermediate_size)` is stable across MoE layers in all
currently-targeted architectures. The backend cache assumes this; the
shape-mismatch fallback (line 184-186) handles a hypothetical
heterogeneous-MoE architecture by reallocating, with no correctness
risk — only a perf cost on the first token of each new shape.

## 5. ROADMAP claim provenance

The 120 ms / 300 allocation numbers come from
`crates/larql-compute/ROADMAP.md` based on a 194 ms baseline minus
40 ms compute and 30 ms syncs, on Gemma 4 26B A4B M3 Max. That
measurement was correct **for the first decode token** before the cache
landed. After the cache landed, the measurement was not refreshed in
either ROADMAP. The vindex ROADMAP entry inherited the stale framing.

The "~50 ms/tok ≈ 20 tok/s" target in the table at
`crates/larql-vindex/ROADMAP.md:108-113` is also unmeasured — it is
arithmetic (`190 - 120 = 70`, then "round down for residual savings").
No bench has tied it out.

## 6. What's actually open

Three follow-ups exist; none of them require new allocator design.

### 6a. Benchmark steady-state decode

Wired up by extending `crates/larql-inference/examples/bench_generate.rs`
with a Cold-vs-Warm summary that fires when `--warmup 0`.

Recommended invocation on Gemma 4 26B A4B M3 Max:

```
cargo run --release --features metal -p larql-inference \
    --example bench_generate -- \
    --vindex output/gemma4-26b-a4b-q4k.vindex \
    --warmup 0 \
    --max-tokens 10
```

The output ends with a block like:

```
Cold vs warm (warmup=0, MoE Phase 2 acceptance check):
  Cold (token 1):              ???.?ms  (?? tok/s)
  Warm (mean of tokens 2-10):  ???.?ms  (?? tok/s)
  First-token overhead:        ???.?ms  (cold/warm = ?.??×)
```

**Interpretation:**

- **First-token overhead ≈ 120 ms (cold/warm > 1.5×):** The cache is
  working as designed; the 120 ms allocation cost is paid once per
  loaded model, not per token. Phase 2 acceptance is met by existing
  code; close the ROADMAP entry.
- **First-token overhead ≪ 120 ms (cold/warm ≈ 1×):** The cache is
  warming up during the prefill pass (which also creates the scratch
  before the first decode token), so the "cold" we measured here is
  already partly warm. Re-run with a probe that explicitly exercises
  decode before any prefill.
- **Warm decode still ~190 ms/tok:** The cache is irrelevant to the
  bottleneck — the ROADMAP's allocation framing was wrong. Investigate
  per-layer encoder commit/wait, expert byte memcpy, or sync cost as
  the next perf target.

### 6b. ROADMAP correction

`crates/larql-vindex/ROADMAP.md:104-161` should be rewritten:
- Remove "Phase 2 open" — the cache is shipped.
- Replace "300 allocations / 120 ms / 5× target" with the warm vs
  cold-token measurements from 6a.
- If the warm-token decode is already at the projected ~50 ms target
  (or close), close the entry as Done. If it's above that, open a new
  entry with the actual residual cost as the bottleneck (likely
  per-layer encoder commit/wait overhead, not allocation).

### 6c. inter=704 accuracy bug (separate from allocation)

`crates/larql-server/src/routes/expert/metal.rs:42-57` documents an
unresolved accuracy bug in the Metal MoE expert kernel for Gemma 4
26B-A4B-it (inter=704, top_k=8): cosine similarity ≈ 0.7 vs CPU
reference. The whole Metal expert path is therefore opt-in
(`LARQL_USE_METAL_EXPERTS=1`). This is a separate kernel-correctness
issue, not a Phase 2 perf concern, but it blocks the server expert
path from defaulting to Metal regardless of how fast the cache makes
it.

## 7. Recommendation

**Don't build new allocator code.** The Phase 2 design described in the
ROADMAP is already implemented; further work would be duplicate
caching. Concretely:

1. **Run the cold/warm benchmark in 6a.** Numbers in `cargo bench`
   output, on a quiet M3 Max, with `LARQL_USE_METAL_EXPERTS=1` if
   exercising the server path or via the in-process `decode_token_q4k_moe`
   directly.
2. **Update both ROADMAPs** based on the result. If the cache delivers
   the projected 5× win on warm tokens, mark Phase 2 done. If it
   doesn't, the ROADMAP entry should be rewritten around the actual
   bottleneck (likely encoder commit cost).
3. **Treat the inter=704 accuracy bug as separate work** — kernel
   correctness, not allocation.

If after benchmarks (6a) the cache is shown to be working as
designed, the only follow-on is the doc update. No vindex code change
is required for Phase 2 acceptance.
