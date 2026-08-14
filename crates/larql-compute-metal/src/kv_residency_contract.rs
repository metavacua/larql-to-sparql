//! The KV residency contract, and the divergence between the two paths
//! that implement it.
//!
//! Three quantities are routinely conflated. They are not the same, and
//! the whole contract is about how they relate:
//!
//! ```text
//! architectural reachability   W          what attention may ever read
//! logical occupancy            <= 2W      rows physically retained
//! physical capacity            4096       rows the Metal buffer holds
//! ```
//!
//! `W` comes from the model (`sliding_window`, 1024 on Gemma 3).
//! Occupancy is bounded by compaction, which reclaims only past
//! [`COMPACTION_SLACK`]` x W` so the memmove is O(1) amortised rather
//! than O(W) per token. Capacity is
//! `DEFAULT_GPU_KV_CACHE_MAX_SEQ`, chosen at allocation and never
//! revisited.
//!
//! The gap between the second and third rows is the point: **compaction
//! reclaims occupancy, not bytes.** A layer that has evicted down to `W`
//! still owns a 4096-row buffer. Wiring compaction into a path does not
//! shrink an allocation, and reading it as a memory saving is the
//! mistake this module exists to prevent.
//!
//! # What each path implements
//!
//! ```text
//!                        coarse KvDispatch    fused Metal decode
//! attention span         W                    W
//! logical occupancy      <= 2W                grows unbounded
//! absolute position      monotonic            monotonic
//! rows read by attention identical            identical
//! physical capacity      4096                 4096
//! ```
//!
//! Both paths clamp what attention *reads* — `attention_span` is applied
//! in `decode/encode_attn.rs` regardless of who is driving. Only the
//! coarse path bounds what is *retained*:
//! `coarse_decode_step_windowed` calls `compact_kv_to_window` each step,
//! and nothing on the fused path does. So the two paths agree on
//! semantic visibility and disagree on physical residency.
//!
//! That divergence is what the tests below pin down. They are not a
//! regression guard on a bug; they are the executable statement of the
//! contract, so that a later change to either path has to say which row
//! of the table it is moving.
//!
//! # Why this is not simply fixed by calling the compactor
//!
//! Three operations touch a layer's rows, not two:
//!
//! ```text
//! allocate   KVCache::new_per_layer, one capacity for every layer
//! prefill    full_pipeline/kv_copy.rs, bulk-copies seq_len rows
//! decode     encode_attn.rs, appends one row per step
//! ```
//!
//! Compaction addresses only the third. Prefill writes `seq_len` rows in
//! one `copy_nonoverlapping` and assigns `current_len = seq_len`; its
//! SAFETY note claims the copy is "bounded by max_seq", but that bound
//! is supplied by the caller and enforced nowhere local. Lowering a
//! sliding layer's capacity below the longest admissible prompt
//! therefore turns prefill into an overrun, and no amount of
//! decode-time compaction prevents it.
//!
//! So a per-layer capacity change has to answer a question first: does a
//! sliding layer ever need every prefill row materialised at once, or
//! only the terminal tail once that layer's prefill attention has
//! consumed them? The answer is probably the tail — but it has to be
//! established, not assumed, and it is a change to what prefill writes
//! rather than to where it writes.
//!
//! # The shape a future seam should take
//!
//! [`crate::MetalBackend::set_engine_window`] is the one place a policy
//! decision already crosses into the fused path, and it works because it
//! is *configuration consumed inside the existing dispatch* rather than
//! a callback per layer. Per-layer synchronous dispatch was measured and
//! lost — see the `KvDispatch` Step 5 finding in `ROADMAP.md` — because
//! each call forces its own command-buffer commit.
//!
//! Whatever eventually carries per-layer capacity should therefore be
//! submitted once at setup, alongside the window, not queried per layer
//! per token. Note also that the two ideas are different in kind:
//! `num_kv_heads` and `head_dim` describe the *representation*, while
//! capacity, retention and attention window describe *residency policy*.
//! Widening the shape tuple to carry both would put 134 references
//! across 26 files and 5 crates behind one type — the pressure is
//! pointing at a separate object, not a wider tuple.

#[cfg(test)]
mod tests {
    use crate::kv_dispatch_impl::COMPACTION_SLACK;
    use crate::ops::kv_cache::{attention_span, KVCache};
    use crate::MetalBackend;

    /// Gemma 3's real sliding window, so the boundaries exercised here
    /// are the ones production actually crosses.
    const W: usize = 1024;

    /// Rows allocated per layer today (`DEFAULT_GPU_KV_CACHE_MAX_SEQ`).
    /// Several assertions below exist to show compaction does *not*
    /// move this number.
    const ALLOCATED_ROWS: usize = 4096;

    /// Small per-row geometry — the contract is about row counts, not
    /// row width, and a narrow row keeps the buffers cheap.
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;

    /// Every transition the contract has: just inside the window, at the
    /// window, just inside and at the compaction trigger, one past it,
    /// and a long run that forces repeated slides rather than one.
    const BOUNDARIES: [usize; 6] = [W - 1, W, 2 * W - 1, 2 * W, 2 * W + 1, 3000];

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    fn install_single_layer_cache(b: &MetalBackend) {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        *guard = Some(KVCache::new(&b.bufs, 1, ALLOCATED_ROWS, KV_HEADS, HEAD_DIM));
    }

    /// Write `value` into every element of the row about to be appended,
    /// then advance. Mirrors the real append order — `encode_kv_append`
    /// writes at `current_len` and only then bumps it — so the stamp
    /// identifies which absolute position a surviving row came from.
    fn append_stamped(b: &MetalBackend, value: f32) {
        let mut guard = b.kv_cache.lock().expect("kv cache lock");
        let layer = &mut guard.as_mut().expect("cache installed").layers[0];
        let row = KV_HEADS * HEAD_DIM;
        let at = layer.current_len;
        for buf in [&layer.k_cache, &layer.v_cache] {
            let ptr = buf.contents() as *mut f32;
            // SAFETY: buffer is host-visible and sized ALLOCATED_ROWS * row;
            // `at < ALLOCATED_ROWS` holds for every count under test.
            unsafe {
                let slice = std::slice::from_raw_parts_mut(ptr, ALLOCATED_ROWS * row);
                for i in 0..row {
                    slice[at * row + i] = value;
                }
            }
        }
        layer.advance_one();
    }

    /// Drive `tokens` decode steps. `window` `Some` reproduces the
    /// coarse path, which compacts before each step; `None` reproduces
    /// the fused path, which never does.
    ///
    /// Returns `(current_len, abs_position)`.
    fn drive(b: &MetalBackend, tokens: usize, window: Option<usize>) -> (usize, usize) {
        install_single_layer_cache(b);
        for t in 0..tokens {
            // Compaction runs before the append, exactly as
            // `coarse_decode_step_windowed` orders it.
            if let Some(w) = window {
                b.compact_kv_to_window(w);
            }
            append_stamped(b, t as f32);
        }
        let guard = b.kv_cache.lock().expect("kv cache lock");
        let layer = &guard.as_ref().expect("cache installed").layers[0];
        (layer.current_len, layer.abs_position)
    }

    /// Read the newest `span` rows as the absolute positions they were
    /// stamped with — what attention would actually consume.
    fn visible_rows(b: &MetalBackend, span: usize) -> Vec<f32> {
        let guard = b.kv_cache.lock().expect("kv cache lock");
        let layer = &guard.as_ref().expect("cache installed").layers[0];
        let row = KV_HEADS * HEAD_DIM;
        let start = layer.current_len - span;
        let ptr = layer.k_cache.contents() as *const f32;
        // SAFETY: host-visible, sized ALLOCATED_ROWS * row, and
        // `current_len <= ALLOCATED_ROWS`.
        let slice = unsafe { std::slice::from_raw_parts(ptr, ALLOCATED_ROWS * row) };
        (start..layer.current_len).map(|r| slice[r * row]).collect()
    }

    // ── the four rows of the contract table ──────────────────────────

    /// Attention span is identical on both paths at every boundary.
    /// This is the claim that residency policy does not change what the
    /// model can see — without it, compaction would be a semantic change
    /// rather than a memory one.
    #[test]
    fn both_paths_present_the_same_attention_span() {
        let b = backend();
        for &n in &BOUNDARIES {
            let (fused_len, _) = drive(&b, n, None);
            let (coarse_len, _) = drive(&b, n, Some(W));

            let fused_span = attention_span(fused_len as u32, W as u32);
            let coarse_span = attention_span(coarse_len as u32, W as u32);

            assert_eq!(
                fused_span, coarse_span,
                "semantic visibility must not depend on residency policy (n={n})"
            );
            assert_eq!(
                fused_span as usize,
                n.min(W),
                "span is min(tokens, window) (n={n})"
            );
        }
    }

    /// Occupancy is where the paths part: the coarse path stays inside
    /// the slack bound, the fused path grows with the stream.
    #[test]
    fn occupancy_diverges_past_the_compaction_trigger() {
        let b = backend();
        let bound = W * COMPACTION_SLACK;
        for &n in &BOUNDARIES {
            let (fused_len, _) = drive(&b, n, None);
            let (coarse_len, _) = drive(&b, n, Some(W));

            assert_eq!(
                fused_len, n,
                "the fused path retains every row it has ever written (n={n})"
            );
            assert!(
                coarse_len <= bound,
                "the coarse path must stay within {bound} rows, got {coarse_len} (n={n})"
            );
            if n <= bound {
                assert_eq!(
                    coarse_len, n,
                    "below the trigger the two paths are identical (n={n})"
                );
            } else {
                assert!(
                    coarse_len < fused_len,
                    "past the trigger the coarse path must retain strictly less (n={n})"
                );
            }
        }
    }

    /// Absolute position is monotonic and identical on both paths —
    /// eviction lowers occupancy only. If this ever fails, RoPE rewinds
    /// and output goes fluent-but-wrong.
    #[test]
    fn absolute_position_tracks_tokens_seen_on_both_paths() {
        let b = backend();
        for &n in &BOUNDARIES {
            let (_, fused_pos) = drive(&b, n, None);
            let (_, coarse_pos) = drive(&b, n, Some(W));
            assert_eq!(
                fused_pos, n,
                "fused position must equal tokens seen (n={n})"
            );
            assert_eq!(
                coarse_pos, n,
                "compaction must not rewind the stream (n={n})"
            );
        }
    }

    /// The rows attention actually consumes are identical on both paths.
    ///
    /// This is the output-parity row of the table, reduced to its cause:
    /// if the visible window holds the same absolute positions in the
    /// same order, the attention inputs are the same and the logits
    /// cannot differ. It is the precondition for ever wiring compaction
    /// into the fused path.
    #[test]
    fn the_rows_attention_reads_are_identical_on_both_paths() {
        let b = backend();
        for &n in &BOUNDARIES {
            let span = n.min(W);

            drive(&b, n, None);
            let fused_visible = visible_rows(&b, span);

            drive(&b, n, Some(W));
            let coarse_visible = visible_rows(&b, span);

            assert_eq!(
                fused_visible, coarse_visible,
                "compaction changed what attention reads (n={n})"
            );
            // And both hold the newest `span` absolute positions, in order.
            let expected: Vec<f32> = ((n - span)..n).map(|p| p as f32).collect();
            assert_eq!(
                fused_visible, expected,
                "visible window is the tail (n={n})"
            );
        }
    }

    /// Compaction reclaims occupancy and **not** bytes. The buffer stays
    /// the size it was allocated, on both paths, at every boundary.
    ///
    /// This is the assertion that stops "wire in the compactor" from
    /// being mistaken for a memory saving: getting the allocation down
    /// needs a capacity change, which needs prefill solved first.
    #[test]
    fn compaction_does_not_shrink_the_allocation() {
        let b = backend();
        for &n in &BOUNDARIES {
            for window in [None, Some(W)] {
                drive(&b, n, window);
                let guard = b.kv_cache.lock().expect("kv cache lock");
                let layer = &guard.as_ref().expect("cache installed").layers[0];
                assert_eq!(
                    layer.max_seq, ALLOCATED_ROWS,
                    "capacity is fixed at allocation (n={n}, window={window:?})"
                );
                let row_bytes = (KV_HEADS * HEAD_DIM * 4) as u64;
                assert_eq!(
                    layer.k_cache.length(),
                    ALLOCATED_ROWS as u64 * row_bytes,
                    "the Metal buffer is untouched by eviction (n={n})"
                );
            }
        }
    }

    /// Repeated slides, not merely the first one.
    ///
    /// The first compaction fires when occupancy reaches `W x SLACK`
    /// (2048) and drops it to `W`; the next needs another `W` appends,
    /// so the second slide lands at `3W`. Anything short of that
    /// exercises a single memmove over pristine rows and would miss a
    /// compaction landing on already-compacted ones.
    #[test]
    fn repeated_slides_keep_the_tail_correct() {
        let b = backend();
        let first_slide = W * COMPACTION_SLACK;
        let second_slide = first_slide + W;
        let n = second_slide + 128;
        assert!(
            n > second_slide && n <= ALLOCATED_ROWS,
            "the long run must cross the trigger twice and still fit the buffer"
        );
        let (coarse_len, coarse_pos) = drive(&b, n, Some(W));
        assert!(coarse_len <= W * COMPACTION_SLACK);
        assert_eq!(coarse_pos, n);

        let span = W;
        let visible = visible_rows(&b, span);
        let expected: Vec<f32> = ((n - span)..n).map(|p| p as f32).collect();
        assert_eq!(
            visible, expected,
            "the tail must survive repeated compactions intact"
        );
    }
}

#[cfg(test)]
mod capacity_tests {
    use crate::ops::kv_cache::KVCache;
    use crate::MetalBackend;
    use larql_compute::pipeline_layer::{kv_capacity_for_window, KV_COMPACTION_SLACK};

    /// Gemma 3 4B: 34 layers, 4 KV heads, head_dim 256, window 1024, and
    /// the 5:1 global pattern that puts full attention on layers
    /// 5/11/17/23/29.
    const G_LAYERS: usize = 34;
    const G_KV_HEADS: usize = 4;
    const G_HEAD_DIM: usize = 256;
    const G_WINDOW: usize = 1024;
    const G_DEFAULT: usize = 4096;
    /// Gemma 3 4B's published split. Stated as a fact about the model
    /// rather than recomputed from its layer pattern: re-implementing
    /// `gemma3.rs::is_sliding_window_layer` here would be a second copy
    /// of an architecture rule that can drift silently, and this test is
    /// about allocation arithmetic, not about that rule.
    const G_SLIDING: usize = 29;
    const G_GLOBAL: usize = 5;

    /// Capacity is the compaction trigger, not the window — a capacity of
    /// `W` would be an overrun, since occupancy is allowed to reach
    /// `SLACK * W` before compaction reclaims it.
    #[test]
    fn capacity_accommodates_the_compaction_trigger() {
        assert_eq!(kv_capacity_for_window(G_WINDOW, G_DEFAULT), 2048);
        assert_eq!(
            kv_capacity_for_window(G_WINDOW, G_DEFAULT),
            G_WINDOW * KV_COMPACTION_SLACK,
            "capacity below the trigger is an overrun, not a saving"
        );
        // A global layer is unbounded and keeps the full default.
        assert_eq!(kv_capacity_for_window(0, G_DEFAULT), G_DEFAULT);
        // A window wider than the default cannot ask for more than it.
        assert_eq!(kv_capacity_for_window(8192, G_DEFAULT), G_DEFAULT);
    }

    /// The byte win, asserted on the real geometry so it cannot regress
    /// silently. Measures allocated bytes, not row counts — a row-count
    /// assertion would pass even if the buffers never shrank.
    #[test]
    fn gemma3_4b_allocation_falls() {
        let Some(metal) = MetalBackend::new() else {
            eprintln!("skip: no Metal device");
            return;
        };
        assert_eq!(
            G_SLIDING + G_GLOBAL,
            G_LAYERS,
            "the split must cover every layer"
        );
        let shapes = vec![(G_KV_HEADS, G_HEAD_DIM); G_LAYERS];
        let capacities: Vec<usize> = (0..G_LAYERS)
            .map(|l| {
                let w = if l < G_SLIDING { G_WINDOW } else { 0 };
                kv_capacity_for_window(w, G_DEFAULT)
            })
            .collect();

        let before = KVCache::new_per_layer(&metal.bufs, &shapes, G_DEFAULT).allocated_bytes();
        let after =
            KVCache::new_per_layer_with_capacities(&metal.bufs, &shapes, &capacities, G_DEFAULT)
                .allocated_bytes();

        // 32 MiB per layer at 4096 rows (K+V, f32) -> 1.088 GiB.
        assert_eq!(before, 1_140_850_688, "baseline allocation");
        // Sliding layers at 16 MiB, global at 32 MiB.
        assert_eq!(after, 654_311_424, "per-layer allocation");
        // Exactly 464 MiB reclaimed — G_SLIDING layers x 16 MiB.
        assert_eq!(
            before - after,
            464 * 1024 * 1024,
            "reclaimed bytes changed; if this is intended, update the figure \
             in docs/kv-residency-contract.md too"
        );
    }

    /// Growing must respect each layer's own bound. Growing to a single
    /// `max_seq` would re-inflate every sliding layer on the first decode
    /// step and silently undo the change.
    #[test]
    fn growing_does_not_re_inflate_a_sliding_layer() {
        let Some(metal) = MetalBackend::new() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let shapes = vec![(G_KV_HEADS, G_HEAD_DIM); 2];
        let capacities = vec![2048usize, G_DEFAULT];
        let mut kv =
            KVCache::new_per_layer_with_capacities(&metal.bufs, &shapes, &capacities, G_DEFAULT);
        let before = kv.allocated_bytes();

        kv.grow_to_capacities(&metal.bufs, &shapes, &capacities, G_DEFAULT);

        assert_eq!(
            kv.layers[0].max_seq, 2048,
            "sliding layer must stay bounded"
        );
        assert_eq!(kv.layers[1].max_seq, G_DEFAULT, "global layer unchanged");
        assert_eq!(kv.allocated_bytes(), before, "growing reallocated nothing");
    }

    /// A layer whose capacity genuinely rises is still grown.
    #[test]
    fn growing_still_enlarges_an_undersized_layer() {
        let Some(metal) = MetalBackend::new() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let shapes = vec![(2, 64)];
        let mut kv = KVCache::new_per_layer_with_capacities(&metal.bufs, &shapes, &[64], 64);
        assert_eq!(kv.layers[0].max_seq, 64);
        kv.grow_to_capacities(&metal.bufs, &shapes, &[256], 256);
        assert_eq!(kv.layers[0].max_seq, 256);
    }
}
