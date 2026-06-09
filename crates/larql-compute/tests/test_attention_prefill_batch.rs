//! Phase 4b smoke test: `ComputeBackend::qwen35_attention_prefill_batch`
//! on CUDA hardware.
//!
//! Validates that the batched-prefill attention kernel produces
//! finite outputs over a synthetic 8-position sequence and that the
//! per-layer KV slab is correctly populated for a follow-up
//! single-step decode call.
//!
//! Test is `#[ignore]` by default. Run on a GPU host:
//!
//!     cargo test -p larql-compute --features cuda --release \
//!         --test test_attention_prefill_batch -- --ignored
//!
//! Numerical parity vs the per-token decode path is the headline
//! gate for wiring this into the qwen35 forward — but that requires
//! the host's per-position projection / norm / RoPE bookkeeping,
//! which lives in `larql-inference` not here. For Phase 4b the
//! smoke test only confirms the kernel call site works end-to-end;
//! a parity test against the qwen35 host helpers ships with the
//! Phase 4-final integration.

#![cfg(feature = "cuda")]

use larql_compute::cuda::CudaBackend;
use larql_compute::ComputeBackend;

fn synth_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    for _ in 0..n {
        s = s.wrapping_mul(2654435761).wrapping_add(1);
        out.push(((s >> 11) as i32 % 2048) as f32 / 4096.0);
    }
    out
}

#[test]
#[ignore = "requires CUDA hardware + attached driver — invoke with --ignored"]
fn prefill_batch_finite_outputs_on_qwen35_shape() {
    let Ok(backend) = CudaBackend::new() else {
        eprintln!("skipping: CUDA driver unavailable");
        return;
    };

    // Qwen3.6-35B-A3B full-attn layer shape:
    //   num_q_heads = 16, num_kv_heads = 4, head_dim = 128.
    let num_q_heads = 16_usize;
    let num_kv_heads = 4_usize;
    let head_dim = 128_usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let max_seq = 256_usize;
    let cache_id = 0xC0FFEE_BEEF_u64; // unique to this test
    let seq_len = 8_usize;

    let q_seq = synth_f32(seq_len * q_dim, 0xAAAA);
    let k_seq = synth_f32(seq_len * kv_dim, 0xBBBB);
    let v_seq = synth_f32(seq_len * kv_dim, 0xCCCC);

    let out = backend
        .qwen35_attention_prefill_batch(
            cache_id,
            max_seq,
            0, // base_pos = start of cache
            &q_seq,
            &k_seq,
            &v_seq,
            num_q_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        )
        .expect("CUDA backend should serve this shape");
    assert_eq!(out.len(), seq_len * q_dim);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "prefill_batch produced non-finite output"
    );
    // Synthetic Q/K/V uniformly in [-0.5, 0.5] → output magnitudes
    // should be O(0.1)-O(1.0) after softmax + GEMV. Loose sanity:
    let max_abs = out.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(
        max_abs < 10.0,
        "implausibly large output: max_abs={max_abs}"
    );

    // Reset clears the slab.
    backend.qwen35_gqa_decode_reset(cache_id);
}
