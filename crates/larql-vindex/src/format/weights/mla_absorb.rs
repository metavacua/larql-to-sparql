/// MLA absorption — fuse DS-V3 low-rank attention projections into standard Q/K/V tensors.
///
/// DS-V3 stores attention as four weight matrices:
///
///   kv_a  shape (kv_lora_rank + qk_rope, hidden)   — KV compressor + shared RoPE-K
///   kv_b  shape (num_kv*(qk_nope+v_hd), kv_lora)   — KV decompressor (K_nope interleaved with V)
///   q_a   shape (q_lora, hidden)                    — Q compressor
///   q_b   shape (num_q*qk_head_dim, q_lora)         — Q decompressor
///
/// After absorption the caller obtains three dense tensors in LARQL convention:
///
///   Q  shape (num_q  * qk_head_dim, hidden)   per-head layout [rope | nope]
///   K  shape (num_kv * qk_head_dim, hidden)   per-head layout [rope | nope] (rope replicated)
///   V  shape (num_kv * v_head_dim,  hidden)
///
/// The absorbed tensors feed directly into `gqa_attention_asym` because they have the
/// asymmetric qk_head_dim / v_head_dim that function expects.
use ndarray::{Array2, ArrayView2, s};

pub struct MlaGeometry {
    pub num_q: usize,
    pub num_kv: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_hd: usize,
    pub kv_lora: usize,
    pub q_lora: usize,
}

impl MlaGeometry {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
}

/// Absorb MLA projections into standard dense Q/K/V weight matrices.
///
/// Returns `(Q, K, V)` with shapes as documented above.
///
/// # Panics
/// Panics on shape mismatch (programming error, not runtime input error).
pub fn absorb(
    kv_a: &Array2<f32>,
    kv_b: &Array2<f32>,
    q_a: &Array2<f32>,
    q_b: &Array2<f32>,
    g: &MlaGeometry,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let MlaGeometry {
        num_q,
        num_kv,
        qk_nope,
        qk_rope,
        v_hd,
        kv_lora,
        q_lora,
    } = *g;
    let qk_head_dim = qk_nope + qk_rope;
    let hidden = kv_a.ncols();

    // Dimension assertions
    assert_eq!(
        kv_a.nrows(),
        kv_lora + qk_rope,
        "kv_a rows = kv_lora + qk_rope (MQA: single rope-K)"
    );
    assert_eq!(
        kv_b.nrows(),
        num_kv * (qk_nope + v_hd),
        "kv_b rows = num_kv * (qk_nope + v_hd)"
    );
    assert_eq!(kv_b.ncols(), kv_lora);
    assert_eq!(q_a.nrows(), q_lora);
    assert_eq!(q_a.ncols(), hidden);
    assert_eq!(q_b.nrows(), num_q * qk_head_dim);
    assert_eq!(q_b.ncols(), q_lora);

    let kv_compress: ArrayView2<f32> = kv_a.slice(s![..kv_lora, ..]);
    // MQA: single rope-K row shared across all KV heads
    let k_rope_row: ArrayView2<f32> = kv_a.slice(s![kv_lora.., ..]);
    assert_eq!(k_rope_row.nrows(), qk_rope);

    // ── Q ──────────────────────────────────────────────────────────────────
    // Absorbed Q = q_b @ q_a  (shape: num_q*qk_head_dim × hidden)
    // DS-V3 native per-head layout: [nope_dims | rope_dims]
    // LARQL convention: [rope_dims | nope_dims] — swap within each head
    let q_native = q_b.dot(q_a); // (num_q*qk_head_dim, hidden)
    let mut q_out = Array2::<f32>::zeros((num_q * qk_head_dim, hidden));
    for h in 0..num_q {
        let src_base = h * qk_head_dim;
        let dst_base = h * qk_head_dim;
        // rope part: native[nope..qk_head_dim] → dst[0..qk_rope]
        q_out
            .slice_mut(s![dst_base..dst_base + qk_rope, ..])
            .assign(&q_native.slice(s![src_base + qk_nope..src_base + qk_head_dim, ..]));
        // nope part: native[0..qk_nope] → dst[qk_rope..qk_head_dim]
        q_out
            .slice_mut(s![dst_base + qk_rope..dst_base + qk_head_dim, ..])
            .assign(&q_native.slice(s![src_base..src_base + qk_nope, ..]));
    }

    // ── K ──────────────────────────────────────────────────────────────────
    // K_nope[h] = kv_b[h*(nope+v_hd) .. h*(nope+v_hd)+nope, :] @ kv_compress  → (qk_nope, hidden)
    // K_rope    = k_rope_row @ identity                                          → (qk_rope, hidden) shared
    // Per head, LARQL layout: [rope_dims | nope_dims]
    let k_rope_dense = k_rope_row.dot(&Array2::eye(hidden)); // (qk_rope, hidden)
    let mut k_out = Array2::<f32>::zeros((num_kv * qk_head_dim, hidden));
    for h in 0..num_kv {
        let kv_base = h * (qk_nope + v_hd);
        let dst_base = h * qk_head_dim;
        // rope first (broadcast single MQA rope-K)
        k_out
            .slice_mut(s![dst_base..dst_base + qk_rope, ..])
            .assign(&k_rope_dense);
        // nope: absorb
        let k_nope_h = kv_b
            .slice(s![kv_base..kv_base + qk_nope, ..])
            .dot(&kv_compress);
        k_out
            .slice_mut(s![dst_base + qk_rope..dst_base + qk_head_dim, ..])
            .assign(&k_nope_h);
    }

    // ── V ──────────────────────────────────────────────────────────────────
    let mut v_out = Array2::<f32>::zeros((num_kv * v_hd, hidden));
    for h in 0..num_kv {
        let kv_base = h * (qk_nope + v_hd);
        let dst_base = h * v_hd;
        let v_h = kv_b
            .slice(s![kv_base + qk_nope..kv_base + qk_nope + v_hd, ..])
            .dot(&kv_compress);
        v_out
            .slice_mut(s![dst_base..dst_base + v_hd, ..])
            .assign(&v_h);
    }

    (q_out, k_out, v_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::attention::gqa::gqa_attention_asym;
    use ndarray::Array2;

    fn randn(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        // Simple deterministic "random" via LCG
        let mut state = seed;
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    /// Reference MLA forward pass (matches DS-V3 math, rope-first in output).
    ///
    /// Returns (q, k, v) projected activations for a single token x (shape 1×hidden).
    fn mla_reference_forward(
        x: &Array2<f32>,
        kv_a: &Array2<f32>,
        kv_b: &Array2<f32>,
        q_a: &Array2<f32>,
        q_b: &Array2<f32>,
        g: &MlaGeometry,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let MlaGeometry {
            num_q,
            num_kv,
            qk_nope,
            qk_rope,
            v_hd,
            kv_lora,
            ..
        } = *g;
        let qk_head_dim = qk_nope + qk_rope;
        let seq = x.nrows();
        let _hidden = x.ncols();

        // KV latent and shared rope-K
        let kv_latent = x.dot(&kv_a.slice(s![..kv_lora, ..]).t()); // (seq, kv_lora)
        let k_rope_global = x.dot(&kv_a.slice(s![kv_lora.., ..]).t()); // (seq, qk_rope)

        // Q: compress → decompress → reorder rope-first
        let q_latent = x.dot(&q_a.t()); // (seq, q_lora)
        let q_native = q_latent.dot(&q_b.t()); // (seq, num_q*qk_head_dim)
        let mut q_out = Array2::<f32>::zeros((seq, num_q * qk_head_dim));
        for h in 0..num_q {
            let src_base = h * qk_head_dim;
            let dst_base = h * qk_head_dim;
            // rope-first
            q_out
                .slice_mut(s![.., dst_base..dst_base + qk_rope])
                .assign(&q_native.slice(s![.., src_base + qk_nope..src_base + qk_head_dim]));
            q_out
                .slice_mut(s![.., dst_base + qk_rope..dst_base + qk_head_dim])
                .assign(&q_native.slice(s![.., src_base..src_base + qk_nope]));
        }

        // K: nope absorbed, rope replicated, rope-first
        let mut k_out = Array2::<f32>::zeros((seq, num_kv * qk_head_dim));
        for h in 0..num_kv {
            let kv_base = h * (qk_nope + v_hd);
            let dst_base = h * qk_head_dim;
            // rope (broadcast single shared K_rope)
            k_out
                .slice_mut(s![.., dst_base..dst_base + qk_rope])
                .assign(&k_rope_global);
            // nope
            let k_nope_h = kv_latent
                .dot(&kv_b.slice(s![kv_base..kv_base + qk_nope, ..]).t());
            k_out
                .slice_mut(s![.., dst_base + qk_rope..dst_base + qk_head_dim])
                .assign(&k_nope_h);
        }

        // V
        let mut v_out = Array2::<f32>::zeros((seq, num_kv * v_hd));
        for h in 0..num_kv {
            let kv_base = h * (qk_nope + v_hd);
            let dst_base = h * v_hd;
            let v_h = kv_latent.dot(&kv_b.slice(s![kv_base + qk_nope..kv_base + qk_nope + v_hd, ..]).t());
            v_out
                .slice_mut(s![.., dst_base..dst_base + v_hd])
                .assign(&v_h);
        }

        (q_out, k_out, v_out)
    }

    fn geometry() -> MlaGeometry {
        MlaGeometry {
            num_q: 4,
            num_kv: 2,
            qk_nope: 4,
            qk_rope: 2,
            v_hd: 4,
            kv_lora: 8,
            q_lora: 8,
        }
    }

    fn weights(g: &MlaGeometry) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let hidden = 16;
        let qk_head_dim = g.qk_head_dim();
        // kv_a: (kv_lora + qk_rope, hidden) — MQA: one shared rope-K
        let kv_a = randn(g.kv_lora + g.qk_rope, hidden, 1);
        // kv_b: (num_kv*(qk_nope+v_hd), kv_lora)
        let kv_b = randn(g.num_kv * (g.qk_nope + g.v_hd), g.kv_lora, 2);
        let q_a = randn(g.q_lora, hidden, 3);
        let q_b = randn(g.num_q * qk_head_dim, g.q_lora, 4);
        (kv_a, kv_b, q_a, q_b)
    }

    #[test]
    fn absorbed_forward_matches_reference() {
        let g = geometry();
        let (kv_a, kv_b, q_a, q_b) = weights(&g);
        let hidden = 16usize;
        let seq = 3usize;

        // Compute absorbed weight matrices
        let (w_q, w_k, w_v) = absorb(&kv_a, &kv_b, &q_a, &q_b, &g);

        // Random input sequence
        let x = randn(seq, hidden, 99);

        // Reference path: project each token through MLA, then run gqa_attention_asym
        let (q_ref, k_ref, v_ref) = mla_reference_forward(&x, &kv_a, &kv_b, &q_a, &q_b, &g);
        let qk_head_dim = g.qk_head_dim();
        let reps = g.num_q / g.num_kv;
        let scale = 1.0 / (qk_head_dim as f64).sqrt();
        let ref_out = gqa_attention_asym(&q_ref, &k_ref, &v_ref, g.num_q, qk_head_dim, g.v_hd, reps, scale, seq);

        // Absorbed path: project through absorbed weight matrices, then run gqa_attention_asym
        let q_abs = x.dot(&w_q.t());
        let k_abs = x.dot(&w_k.t());
        let v_abs = x.dot(&w_v.t());
        let abs_out = gqa_attention_asym(&q_abs, &k_abs, &v_abs, g.num_q, qk_head_dim, g.v_hd, reps, scale, seq);

        // Must match numerically (within float precision)
        let max_diff = ref_out
            .iter()
            .zip(abs_out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "absorbed forward must match reference, max_diff={max_diff}"
        );
    }

    #[test]
    fn absorbed_shapes() {
        let g = geometry();
        let (kv_a, kv_b, q_a, q_b) = weights(&g);
        let hidden = 16usize;
        let qk_head_dim = g.qk_head_dim();
        let (w_q, w_k, w_v) = absorb(&kv_a, &kv_b, &q_a, &q_b, &g);
        assert_eq!(w_q.shape(), &[g.num_q * qk_head_dim, hidden]);
        assert_eq!(w_k.shape(), &[g.num_kv * qk_head_dim, hidden]);
        assert_eq!(w_v.shape(), &[g.num_kv * g.v_hd, hidden]);
    }

    #[test]
    fn rope_k_is_broadcast_not_zero() {
        // The absorbed K rope section for each KV head must be non-zero
        // and identical across heads (proving the broadcast replicated correctly).
        let g = geometry();
        let (kv_a, kv_b, q_a, q_b) = weights(&g);
        let (_, w_k, _) = absorb(&kv_a, &kv_b, &q_a, &q_b, &g);
        let qk_head_dim = g.qk_head_dim();
        let head0_rope: Vec<f32> = w_k.slice(s![..g.qk_rope, ..]).iter().copied().collect();
        let head1_rope: Vec<f32> = w_k
            .slice(s![qk_head_dim..qk_head_dim + g.qk_rope, ..])
            .iter()
            .copied()
            .collect();
        assert!(head0_rope.iter().any(|v| v.abs() > 1e-6), "rope-K must be non-zero");
        for (a, b) in head0_rope.iter().zip(head1_rope.iter()) {
            assert!((a - b).abs() < 1e-6, "rope-K must be identical across heads: {a} vs {b}");
        }
    }
}
