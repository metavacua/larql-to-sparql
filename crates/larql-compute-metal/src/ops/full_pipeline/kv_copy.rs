//! Post-commit KV cache population for prefill + decode paths.
//!
//! After `dispatch_full_pipeline` commits and waits, the GPU-computed
//! RoPE'd K/V tensors live in per-layer scratch buffers. This module
//! copies them into the persistent KV cache that subsequent
//! `decode_token` calls read from.
//!
//! Pulled out of the orchestrator so `dispatch_full_pipeline` ends at
//! "wait for command buffer" and the cache copy is its own labeled
//! step.

use super::buffers::LayerBuffers;
use crate::buffers::BufferCache;
use crate::decode::DEFAULT_KV_CACHE_MAX_SEQ;
use crate::ops::kv_cache::{KVCache, LayerKVCache};
use larql_compute::FullPipelineLayer;

/// Rows a layer must persist out of prefill.
///
/// A windowed layer's attention reads at most `sliding_window` rows, and
/// prefill's own attention is windowed too (see
/// `tests/test_prefill_sliding_window.rs`), so once prefill has run
/// nothing older than the tail can ever be consumed again. Persisting
/// only that tail is therefore output-neutral, not an approximation.
///
/// It persists `W`, not `COMPACTION_SLACK x W`. The slack exists to
/// amortise *incremental* decode compaction; immediately after prefill
/// there is no history to amortise. Decode then climbs `W -> 2W` and
/// compacts back, which is what makes `2W` genuinely capacity rather
/// than desired occupancy — see
/// [`docs/kv-residency-contract.md`](../../../../../docs/kv-residency-contract.md).
///
/// A window of `0` means unbounded (the global-layer sentinel), so the
/// whole prefix is persisted.
pub(super) fn persisted_rows(seq_len: usize, sliding_window: usize) -> usize {
    if sliding_window == 0 {
        seq_len
    } else {
        seq_len.min(sliding_window)
    }
}

/// Copy the persisted tail of one layer's scratch K/V into its cache.
///
/// `current_len` becomes the number of rows retained, while
/// `abs_position` stays at the full prompt length — the split that lets
/// RoPE keep climbing while occupancy is bounded. After this,
/// **physical row index no longer equals absolute position** for a
/// windowed layer; read the tail, not a position (the same transition
/// issue #200 made on the engine path).
fn copy_persisted_tail(
    cache: &mut LayerKVCache,
    k_src: *const f32,
    v_src: *const f32,
    seq_len: usize,
    sliding_window: usize,
) {
    let row = cache.num_kv_heads * cache.head_dim;
    let keep = persisted_rows(seq_len, sliding_window);
    let skipped = seq_len - keep;
    let n = keep * row;
    // SAFETY: caller has committed + waited, so the source buffers are
    // GPU-finished and hold `seq_len * row` floats; `skipped + keep ==
    // seq_len` keeps the read in bounds. The destination is allocated
    // for `max_seq * row` and `keep <= seq_len <= max_seq`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            k_src.add(skipped * row),
            cache.k_cache.contents() as *mut f32,
            n,
        );
        std::ptr::copy_nonoverlapping(
            v_src.add(skipped * row),
            cache.v_cache.contents() as *mut f32,
            n,
        );
    }
    cache.current_len = keep;
    // Prefill wrote positions 0..seq_len, so the stream is at seq_len
    // regardless of how many rows were kept.
    cache.abs_position = seq_len;
}

/// Copy one layer's K/V scratch into the persistent KV cache.
/// Called inside the per-layer MoE commit loop so the cache is current
/// before the CPU MoE callback reads `h_post_attn` and writes to `new_h`.
pub(super) fn populate_kv_one_layer(
    kv: &mut KVCache,
    bufs: &BufferCache,
    lb: &LayerBuffers,
    layer: &FullPipelineLayer<'_>,
    layer_idx: usize,
    seq_len: usize,
) {
    let lhd = layer.head_dim;
    let lnkv = layer.num_kv_heads;
    while kv.layers.len() <= layer_idx {
        kv.layers
            .push(LayerKVCache::new(bufs, DEFAULT_KV_CACHE_MAX_SEQ, lnkv, lhd));
    }
    let k_src = lb.k_out[layer_idx].contents() as *const f32;
    let v_src = lb.v_out[layer_idx].contents() as *const f32;
    copy_persisted_tail(
        &mut kv.layers[layer_idx],
        k_src,
        v_src,
        seq_len,
        layer.sliding_window,
    );
}

/// Copy each layer's K/V scratch (post-RoPE) into the persistent KV
/// cache. Grows the cache's per-layer storage on demand so it sizes
/// to whichever model variant called us first.
pub(super) fn populate_kv_after_commit(
    kv_cache: Option<&mut KVCache>,
    bufs: &BufferCache,
    lb: &LayerBuffers,
    layers: &[FullPipelineLayer<'_>],
    seq_len: usize,
) {
    let Some(kv) = kv_cache else {
        return;
    };
    for (l, layer) in layers.iter().enumerate() {
        let lhd = layer.head_dim;
        let lnkv = layer.num_kv_heads;
        while kv.layers.len() <= l {
            kv.layers
                .push(LayerKVCache::new(bufs, DEFAULT_KV_CACHE_MAX_SEQ, lnkv, lhd));
        }
        let k_src = lb.k_out[l].contents() as *const f32;
        let v_src = lb.v_out[l].contents() as *const f32;
        copy_persisted_tail(
            &mut kv.layers[l],
            k_src,
            v_src,
            seq_len,
            layer.sliding_window,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalBackend;
    use larql_compute::pipeline::*;

    /// Construct a minimal `FullPipelineLayer` with the per-layer
    /// dims this test cares about. All other fields hold the smallest
    /// valid value.
    fn synth_layer(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> FullPipelineLayer<'static> {
        let q4 = Box::leak(vec![0u8; 32 * 18].into_boxed_slice());
        let norm = Box::leak(vec![1.0f32; 32].into_boxed_slice());
        let q4w = || QuantWeight::new(QuantFormat::Q4_K, q4, larql_compute::QuantAux::None);
        FullPipelineLayer {
            attn_sinks: None,
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_o_bias: None,
            attn_softcap: 0.0,
            wq: q4w(),
            wk: q4w(),
            wv: q4w(),
            wo: q4w(),
            gate: q4w(),
            up: q4w(),
            down: q4w(),
            input_norm: norm,
            post_attn_norm: norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 1.0,
            eps: larql_compute::pipeline::RMSNORM_EPSILON_DEFAULT,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: larql_compute::pipeline::ROPE_BASE_DEFAULT,
            rotary_dim: 0,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                head_dim,
                0_usize,
                larql_compute::pipeline::ROPE_BASE_DEFAULT as f64,
            ),
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
        }
    }

    /// Read a Metal Buffer's contents as f32s.
    fn read_metal_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    /// Write a known f32 pattern into a Metal Buffer's contents.
    fn write_metal_f32(buf: &metal::Buffer, src: &[f32]) {
        let ptr = buf.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
        }
    }

    /// `None` cache → no-op. Function returns silently without panicking.
    #[test]
    fn populate_kv_after_commit_with_none_cache_is_a_noop() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let layers = vec![synth_layer(8, 4, 64)];
        let lb = LayerBuffers::allocate(metal.bufs(), &layers, &[0.0; 64], 64, 256, 1, 8 * 64);
        // Pre-condition: function returns without touching anything.
        populate_kv_after_commit(None, metal.bufs(), &lb, &layers, 1);
    }

    /// Cache pre-sized to num_layers — copies land at the right
    /// destination layer with the right byte count and `current_len`.
    #[test]
    fn populate_kv_after_commit_copies_into_correct_layer() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let bufs = metal.bufs();

        let head_dim = 64;
        let num_kv_heads = 4;
        let lkv = num_kv_heads * head_dim; // 256
        let seq_len = 3;
        let total = seq_len * lkv; // 768 floats per layer
        let layers = vec![
            synth_layer(8, num_kv_heads, head_dim),
            synth_layer(8, num_kv_heads, head_dim),
        ];
        let lb = LayerBuffers::allocate(bufs, &layers, &[0.0; 64], 64, 256, seq_len, 8 * head_dim);

        // Stamp distinguishable patterns into each layer's k_out / v_out.
        // L0 K = [100.0, 100.1, 100.2, …]; L0 V = [200.0, …]; L1 K = [300.0, …]; L1 V = [400.0, …].
        let mk_pattern =
            |base: f32, n: usize| -> Vec<f32> { (0..n).map(|i| base + i as f32 * 0.1).collect() };
        let l0_k = mk_pattern(100.0, total);
        let l0_v = mk_pattern(200.0, total);
        let l1_k = mk_pattern(300.0, total);
        let l1_v = mk_pattern(400.0, total);
        write_metal_f32(&lb.k_out[0], &l0_k);
        write_metal_f32(&lb.v_out[0], &l0_v);
        write_metal_f32(&lb.k_out[1], &l1_k);
        write_metal_f32(&lb.v_out[1], &l1_v);

        // Pre-allocated cache, 2 layers same dims.
        let mut kv = KVCache::new(bufs, 2, DEFAULT_KV_CACHE_MAX_SEQ, num_kv_heads, head_dim);
        assert_eq!(kv.layers[0].current_len, 0);
        assert_eq!(kv.layers[1].current_len, 0);

        populate_kv_after_commit(Some(&mut kv), bufs, &lb, &layers, seq_len);

        // current_len updated.
        assert_eq!(kv.layers[0].current_len, seq_len);
        assert_eq!(kv.layers[1].current_len, seq_len);

        // Cache contents match what we stamped — and only the first
        // `total` floats; the rest of the cache stays
        // at the buffer's zero-init.
        let l0_k_got = read_metal_f32(&kv.layers[0].k_cache, total);
        let l0_v_got = read_metal_f32(&kv.layers[0].v_cache, total);
        let l1_k_got = read_metal_f32(&kv.layers[1].k_cache, total);
        let l1_v_got = read_metal_f32(&kv.layers[1].v_cache, total);
        assert_eq!(l0_k_got, l0_k, "L0 K cache mismatch");
        assert_eq!(l0_v_got, l0_v, "L0 V cache mismatch");
        assert_eq!(l1_k_got, l1_k, "L1 K cache mismatch");
        assert_eq!(l1_v_got, l1_v, "L1 V cache mismatch");
    }

    // ── persistent-tail prefill ──────────────────────────────────────
    //
    // Does a sliding layer need every prefill row materialised in the
    // persistent cache, or only the terminal tail? These decide it. The
    // reference arm is a layer with `sliding_window = 0` (persist
    // everything, the previous behaviour); the candidate is the same
    // prefill through a windowed layer.

    /// A window comfortably smaller than the prompt, so most rows fall
    /// outside it. A fixture where the two behaviours coincide would be
    /// an absent test, not a weak one.
    const EXP_WINDOW: usize = 8;
    const EXP_SEQ_LEN: usize = 48;

    #[test]
    fn persisted_rows_keeps_the_window_or_everything() {
        // Unbounded sentinel — global layers persist the whole prefix.
        assert_eq!(persisted_rows(48, 0), 48);
        // A window wider than the prompt cannot bind.
        assert_eq!(persisted_rows(8, 64), 8);
        // The case that matters.
        assert_eq!(persisted_rows(48, 8), 8);
        // Persist W, not COMPACTION_SLACK * W — prefill has no history
        // to amortise, so the slack is capacity rather than occupancy.
        assert_eq!(persisted_rows(48, 8), EXP_WINDOW);
    }

    /// The decisive comparison: the rows attention will read after
    /// prefill are byte-identical whether the whole prefix or only the
    /// tail was persisted, and the stream position agrees.
    ///
    /// Rows are stamped with their absolute position, so this asserts
    /// the two arms hold the *same absolute positions in the same
    /// order* — the same reduction of the parity claim used for the
    /// residency contract.
    #[test]
    fn tail_only_prefill_leaves_attention_reading_identical_rows() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let bufs = metal.bufs();
        let (head_dim, num_kv_heads) = (64usize, 4usize);
        let row = num_kv_heads * head_dim;
        let total = EXP_SEQ_LEN * row;

        // Stamp every scratch row with its absolute position.
        let stamp = |base: f32| -> Vec<f32> {
            (0..total)
                .map(|i| base + (i / row) as f32)
                .collect::<Vec<f32>>()
        };
        let k_pattern = stamp(0.0);
        let v_pattern = stamp(1000.0);

        // Run one prefill through a layer with the given window and
        // return (visible K rows, visible V rows, current_len, abs_position),
        // where "visible" is the newest EXP_WINDOW rows — what a decode
        // step's `attention_span` will read.
        let run = |window: usize| {
            let mut layer = synth_layer(8, num_kv_heads, head_dim);
            layer.sliding_window = window;
            let layers = vec![layer];
            let lb = LayerBuffers::allocate(
                bufs,
                &layers,
                &[0.0; 64],
                64,
                256,
                EXP_SEQ_LEN,
                8 * head_dim,
            );
            write_metal_f32(&lb.k_out[0], &k_pattern);
            write_metal_f32(&lb.v_out[0], &v_pattern);

            let mut kv = KVCache::new(bufs, 1, DEFAULT_KV_CACHE_MAX_SEQ, num_kv_heads, head_dim);
            populate_kv_after_commit(Some(&mut kv), bufs, &lb, &layers, EXP_SEQ_LEN);

            let len = kv.layers[0].current_len;
            let start = (len - EXP_WINDOW) * row;
            let k_all = read_metal_f32(&kv.layers[0].k_cache, len * row);
            let v_all = read_metal_f32(&kv.layers[0].v_cache, len * row);
            (
                k_all[start..].to_vec(),
                v_all[start..].to_vec(),
                len,
                kv.layers[0].abs_position,
            )
        };

        let (ref_k, ref_v, ref_len, ref_pos) = run(0);
        let (tail_k, tail_v, tail_len, tail_pos) = run(EXP_WINDOW);

        assert_eq!(ref_len, EXP_SEQ_LEN, "the reference persists everything");
        assert_eq!(tail_len, EXP_WINDOW, "the candidate persists only the tail");
        assert_eq!(
            ref_pos, tail_pos,
            "abs_position is the prompt length on both arms — it is what RoPE reads"
        );
        assert_eq!(ref_pos, EXP_SEQ_LEN);

        assert_eq!(
            ref_k, tail_k,
            "attention would read different K rows after a tail-only prefill"
        );
        assert_eq!(
            ref_v, tail_v,
            "attention would read different V rows after a tail-only prefill"
        );

        // And they are the newest EXP_WINDOW absolute positions, in order —
        // not merely equal to each other.
        let expected_first_pos = (EXP_SEQ_LEN - EXP_WINDOW) as f32;
        assert_eq!(tail_k[0], expected_first_pos);
        assert_eq!(tail_k[tail_k.len() - row], (EXP_SEQ_LEN - 1) as f32);
    }

    /// A global layer is untouched: window `0` still persists the whole
    /// prefix, so the change cannot silently truncate global history.
    #[test]
    fn a_global_layer_still_persists_the_whole_prefix() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let bufs = metal.bufs();
        let (head_dim, num_kv_heads) = (64usize, 4usize);
        let layers = vec![synth_layer(8, num_kv_heads, head_dim)];
        assert_eq!(layers[0].sliding_window, 0, "synth layer is global");
        let lb = LayerBuffers::allocate(
            bufs,
            &layers,
            &[0.0; 64],
            64,
            256,
            EXP_SEQ_LEN,
            8 * head_dim,
        );
        let mut kv = KVCache::new(bufs, 1, DEFAULT_KV_CACHE_MAX_SEQ, num_kv_heads, head_dim);
        populate_kv_after_commit(Some(&mut kv), bufs, &lb, &layers, EXP_SEQ_LEN);
        assert_eq!(kv.layers[0].current_len, EXP_SEQ_LEN);
        assert_eq!(kv.layers[0].abs_position, EXP_SEQ_LEN);
    }

    /// Cache empty (or shorter than num_layers) → grows on demand to
    /// match. Catches the prefill-grow path that runs when a smaller
    /// model decoded first and a larger one hits the same backend.
    #[test]
    fn populate_kv_after_commit_grows_undersized_cache() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let bufs = metal.bufs();

        let layers = vec![
            synth_layer(8, 4, 64),
            synth_layer(8, 4, 64),
            synth_layer(8, 4, 64),
        ];
        let lb = LayerBuffers::allocate(bufs, &layers, &[0.0; 64], 64, 256, 1, 8 * 64);

        // Cache starts empty.
        let mut kv = KVCache { layers: vec![] };
        populate_kv_after_commit(Some(&mut kv), bufs, &lb, &layers, 1);
        assert_eq!(kv.layers.len(), 3, "cache must grow to num_layers");
        for l in 0..3 {
            assert_eq!(kv.layers[l].current_len, 1);
            assert_eq!(kv.layers[l].num_kv_heads, 4);
            assert_eq!(kv.layers[l].head_dim, 64);
        }
    }

    // ── populate_kv_one_layer ─────────────────────────────────────────────────

    /// `populate_kv_one_layer` targets exactly one layer — other layers in the
    /// cache must be untouched. This is the per-layer variant used in the
    /// batched MoE prefill commit loop.
    #[test]
    fn populate_kv_one_layer_updates_only_target_layer() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let bufs = metal.bufs();

        let head_dim = 64usize;
        let num_kv_heads = 4usize;
        let seq_len = 3usize;
        let total_kv = seq_len * num_kv_heads * head_dim;

        let layers = vec![
            synth_layer(8, num_kv_heads, head_dim),
            synth_layer(8, num_kv_heads, head_dim),
        ];
        let lb = LayerBuffers::allocate(bufs, &layers, &[0.0; 64], 64, 256, seq_len, 8 * head_dim);

        // Stamp a distinct pattern into layer 1's K/V scratch buffers.
        let k_pat: Vec<f32> = (0..total_kv).map(|i| 50.0 + i as f32 * 0.1).collect();
        let v_pat: Vec<f32> = (0..total_kv).map(|i| 60.0 + i as f32 * 0.1).collect();
        write_metal_f32(&lb.k_out[1], &k_pat);
        write_metal_f32(&lb.v_out[1], &v_pat);

        let mut kv = KVCache::new(bufs, 2, DEFAULT_KV_CACHE_MAX_SEQ, num_kv_heads, head_dim);
        assert_eq!(kv.layers[0].current_len, 0);
        assert_eq!(kv.layers[1].current_len, 0);

        populate_kv_one_layer(&mut kv, bufs, &lb, &layers[1], 1, seq_len);

        // Layer 0 must be untouched.
        assert_eq!(kv.layers[0].current_len, 0, "layer 0 must not be updated");

        // Layer 1 must reflect the stamped K/V.
        assert_eq!(
            kv.layers[1].current_len, seq_len,
            "layer 1 current_len updated"
        );
        let k_got = read_metal_f32(&kv.layers[1].k_cache, total_kv);
        let v_got = read_metal_f32(&kv.layers[1].v_cache, total_kv);
        assert_eq!(k_got, k_pat, "K cache mismatch");
        assert_eq!(v_got, v_pat, "V cache mismatch");
    }

    /// `populate_kv_one_layer` grows an empty cache on demand (same as the
    /// `populate_kv_after_commit` grow path, but per layer).
    #[test]
    fn populate_kv_one_layer_grows_empty_cache() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        let bufs = metal.bufs();

        let layers = vec![synth_layer(8, 4, 64), synth_layer(8, 4, 64)];
        let lb = LayerBuffers::allocate(bufs, &layers, &[0.0; 64], 64, 256, 1, 8 * 64);

        let mut kv = KVCache { layers: vec![] };
        // Populate layer 1 into an empty cache — must grow to at least 2 layers.
        populate_kv_one_layer(&mut kv, bufs, &lb, &layers[1], 1, 1);
        assert!(
            kv.layers.len() >= 2,
            "cache must grow to hold the target layer"
        );
        assert_eq!(kv.layers[1].current_len, 1);
    }
}
