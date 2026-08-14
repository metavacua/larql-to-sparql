# The KV residency contract

Three quantities get conflated whenever sliding-window KV comes up. They
are not the same thing, and most of the confusion in this area is a
failure to say which one is meant.

```text
architectural reachability   W          what attention may ever read
logical occupancy            <= 2W      rows physically retained
physical capacity            4096       rows the Metal buffer holds
```

- **`W`** comes from the model. Gemma 3 declares `sliding_window = 1024`
  and runs 29 sliding layers against 5 global ones.
- **Occupancy** is bounded by compaction, which reclaims only once
  occupancy reaches `COMPACTION_SLACK x W`. The slack exists so the
  memmove is O(1) amortised rather than O(W) per token.
- **Capacity** is `DEFAULT_GPU_KV_CACHE_MAX_SEQ`, fixed at allocation
  and never revisited. Every layer gets the same number regardless of
  its window.

The gap between the second and third rows is the one that matters:
**compaction reclaims occupancy, not bytes.** A layer evicted down to
`W` rows still owns its 4096-row buffer. Wiring compaction into a path
does not shrink an allocation, and reading it as a memory saving is the
specific mistake this document exists to prevent.

The extra `W` between reachability and occupancy is not waste either —
it is amortisation slack, deliberately traded for an O(1) compactor. A
ring buffer would let capacity approach the semantic bound without
changing any policy, by replacing memmove-plus-slack with circular
addressing.

## What each path implements

```text
                        coarse KvDispatch    fused Metal decode
attention span          W                    W
logical occupancy       <= 2W                grows unbounded
absolute position       monotonic            monotonic
rows read by attention  identical            identical
physical capacity       4096                 4096
```

Both paths clamp what attention *reads*: `attention_span` is applied in
[`decode/encode_attn.rs`](../crates/larql-compute-metal/src/decode/encode_attn.rs)
whoever is driving. Only the coarse path bounds what is *retained* —
`coarse_decode_step_windowed` calls `compact_kv_to_window` every step,
and nothing on the fused path does.

So the two paths agree on semantic visibility and disagree on physical
residency. That is the whole finding. It is not a bug in either path;
it is an unstated difference in contract, which is worse, because
neither path documents that it has one.

The executable form of this table lives in
[`kv_residency_contract.rs`](../crates/larql-compute-metal/src/kv_residency_contract.rs),
which drives both policies over the boundaries `W-1, W, 2W-1, 2W, 2W+1`
and a long run that crosses the compaction trigger twice. It asserts
each row: identical spans, diverging occupancy, monotonic and equal
absolute positions, byte-identical visible rows, and unchanged
allocation.

The visible-rows assertion is the output-parity claim reduced to its
cause. If the newest `span` rows hold the same absolute positions in the
same order, the attention inputs are identical and the logits cannot
differ. That is the precondition for ever wiring compaction into the
fused path — and it now has a test rather than an argument.

## Why "just call the compactor" is not the fix

Three operations touch a layer's rows, not two:

| operation | site | behaviour |
|---|---|---|
| allocate | `KVCache::new_per_layer` | one capacity for every layer |
| prefill | [`full_pipeline/kv_copy.rs`](../crates/larql-compute-metal/src/ops/full_pipeline/kv_copy.rs) | bulk-copies `seq_len` rows, sets `current_len = seq_len` |
| decode | [`decode/encode_attn.rs`](../crates/larql-compute-metal/src/decode/encode_attn.rs) | appends one row per step |

Compaction addresses only the third. Prefill writes `seq_len` rows in a
single `copy_nonoverlapping`; its SAFETY note claims the copy is
"bounded by max_seq", but that bound is supplied by the caller and
enforced nowhere local. Lowering a sliding layer's capacity below the
longest admissible prompt therefore turns prefill into an overrun, and
no amount of decode-time compaction prevents it.

So a per-layer capacity change has to answer this first:

> Does a sliding layer ever need every prefill row materialised at once,
> or only the terminal tail once that layer's prefill attention has
> consumed them?

The evidence points at the tail, and it is nearly in hand:

- Prefill attention **is** windowed on Metal, established by
  [`test_prefill_sliding_window.rs`](../crates/larql-compute-metal/tests/test_prefill_sliding_window.rs).
  Once it has run, nothing reads beyond `W` again.
- Prefill computes into per-layer **scratch** buffers (`lb.k_out`), and
  `populate_kv_one_layer` copies scratch into the persistent cache. The
  two are separate allocations, so the full-prefix requirement lands on
  the scratch, not on the cache.

Which suggests the shape of the eventual change:

```text
PREFILL, sliding layer

compute attention over the full causal/window geometry   (scratch, N rows)
                    |
                    v
copy only the terminal resident tail                     (cache, <= capacity)
                    |
                    v
current_len  <= retention capacity
abs_position  = full prompt length
```

That is a change to *what prefill writes*, not to *where it writes* —
materially different from resizing a destination buffer, and the reason
per-layer capacity is not a one-field patch.

Note also what this does **not** buy: the 5 global layers still need
full capacity, and `ensure_prompt_fits` guards the context ceiling on
their behalf. Bounding the 29 sliding layers does not remove the 4096
limit.

## Why the shape tuple is the wrong home for capacity

The obvious move is to widen `(num_kv_heads, head_dim)` to
`(num_kv_heads, head_dim, capacity)`. It is the wrong abstraction
pressure, on two grounds.

**Blast radius.** That tuple threads through
`kv_cache_shapes_for_arch` → `new_per_layer` / `grow_to_shapes` /
`has_shape_mismatch` / `preallocate_kv_cache_per_layer`: 134 references
across 26 files in 5 crates, including all four KV engines' dispatch
modules.

**Category.** `num_kv_heads` and `head_dim` describe the
*representation* — what a row is. Capacity, retention window and
attention window describe *residency policy* — how long rows live. They
belong in separate objects:

```rust
struct KvLayerGeometry {
    num_kv_heads: usize,
    head_dim: usize,
}

struct KvLayerResidency {
    capacity: usize,
    attention_window: Option<usize>,
    retention_window: Option<usize>,
}
```

## The shape a future seam should take

The residency object should be **submitted once as configuration**, not
queried per layer per token. This is not a style preference — the
synchronous per-layer form was built and measured, and it lost.
`crates/larql-compute-metal/src/kv_dispatch_impl.rs` is Step 4
scaffolding whose every method delegates to `CpuBackend`, and the Step 5
finding recorded in [`ROADMAP.md`](../ROADMAP.md) is that per-layer
Metal kernels at the sync trait's granularity are *slower* than the
fused decode path, because each call forces its own command-buffer
commit. `AsyncComputeBackend` — deferred dispatch, intent collection —
is the identified prerequisite for any win there.

`MetalBackend::set_engine_window` is the one policy decision that
already crosses into the fused path successfully, and it works precisely
because it is configuration consumed inside the existing dispatch:

```text
policy decision -> configure backend -> fused execution consumes it
```

rather than:

```text
layer -> call policy -> commit GPU -> layer -> call policy -> commit GPU
```

It is, in effect, the one-dimensional ancestor of a per-layer residency
intent. Whatever carries capacity should extend that pattern, not
reintroduce the per-layer callback.

## Status

- The divergence is **documented and tested**, not fixed.
- Compaction is **not** wired into the fused decode path, deliberately.
- Per-layer capacity is **not** implemented, and is blocked on the
  prefill question above rather than on the allocation change.
