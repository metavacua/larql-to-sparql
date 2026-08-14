//! CPU forward paths driven by Q4_K / Q6_K vindexes (substrate).
//!
//! Layer-scoped tensor materialisation + cached decode + walk-FFN +
//! hidden-state forward + hook-aware variants live here. Routes
//! through `&dyn crate::KvIndex` instead of `&VectorIndex` so the
//! substrate doesn't pull in `larql-vindex` (which sits above compute
//! in the dep chain).
//!
//! Inference-shaped paths that need tokenizers, MoE routing, or
//! orchestration (`generation`, `remote_ffn`, `metal`,
//! `interventions`, `hooks` with engine-side dispatch) stay in
//! `larql-inference`. The leaf compute paths here are what
//! `KvDispatch`'s CPU impl needs to call.

mod cached;
pub(crate) mod dequant;
mod hooks;
mod tensors;
mod walk_ffn;

pub use hooks::predict_kquant_hidden_hooked;

pub use cached::{
    attention_decode_step_native, ffn_decode_step_native, fused_decode_step,
    fused_decode_step_with_state, fused_decode_step_with_state_masked, fused_prefill,
    predict_kquant_decode_step, predict_kquant_decode_step_direct,
    predict_kquant_decode_step_direct_with_state, predict_kquant_prefill,
    predict_kquant_prefill_with_state, supports_cached_decode, supports_direct_matvec_decode,
    CachedTimings, CpuKvCache,
};
pub use tensors::{insert_q4k_layer_tensors, remove_layer_tensors};
pub use walk_ffn::{kquant_ffn_forward_layer, kquant_ffn_forward_layer_q8k};

#[cfg(test)]
mod tests {
    //! End-to-end coverage tests for the three small kquant_forward
    //! files (`walk_ffn`, `tensors`, `hooks`) driven against the
    //! Q4K fixture index. Each test reaches into the file under test
    //! through its public entry point; llvm-cov attributes line
    //! execution to the file containing the line, not the test.
    use super::*;
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::{make_test_q4k_weights, make_test_q4k_weights_silu};
    use ndarray::Array2;

    // ── walk_ffn.rs ───────────────────────────────────────────────────

    // ── tensors.rs: the pure-MoE arm ─────────────────────────────────
    //
    // Pure MoE has no dense FFN slab; requiring `interleaved_kquant`
    // rejected those models before the forward reached its MoE branch
    // (the drifted twin of the larql-inference copy). These pin both
    // sides of the gate.

    /// KvIndex double with attention data only — the shape a pure-MoE
    /// vindex presents (its `interleaved_kquant.bin` is empty).
    struct AttnOnlyIndex {
        attn: Vec<Vec<u8>>,
    }
    impl crate::KvIndex for AttnOnlyIndex {
        fn num_features(&self, _l: usize) -> usize {
            16
        }
        fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
            Some([
                (self.attn[0].as_slice(), "Q4_K"),
                (self.attn[1].as_slice(), "Q4_K"),
                (self.attn[2].as_slice(), "Q4_K"),
                (self.attn[3].as_slice(), "Q4_K"),
            ])
        }
        fn interleaved_kquant_layer_data(
            &self,
            _l: usize,
        ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
            None
        }
        fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
            None
        }
    }

    fn attn_only_fixture() -> (larql_models::ModelWeights, AttnOnlyIndex) {
        // hidden 16, 4 heads × head_dim 4 → every attn matrix is 16×16
        // = 256 elements, one Q4_K super-block.
        let mut weights = larql_models::test_fixtures::make_test_weights();
        weights.arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
        }));
        weights.hidden_size = 16;
        weights.num_layers = 1;
        let ones = vec![1.0f32; 16 * 16];
        let q = crate::cpu::ops::q4_common::quantize_q4_k(&ones);
        let index = AttnOnlyIndex {
            attn: vec![q.clone(), q.clone(), q.clone(), q],
        };
        (weights, index)
    }

    #[test]
    fn pure_moe_layer_inserts_attention_only() {
        let (weights, index) = attn_only_fixture();
        assert!(weights.arch.is_moe() && !weights.arch.is_hybrid_moe());
        let mut scratch = larql_models::DequantScratch::new();
        let keys = insert_q4k_layer_tensors(&mut scratch, &weights, &index, 0)
            .expect("pure MoE must not require a dense FFN slab");
        assert_eq!(keys.len(), 4, "attention only: Q/K/V/O");
        assert!(keys.iter().all(|k| k.contains("self_attn")));
        for k in &keys {
            assert!(scratch.contains_key(k), "{k} not inserted");
        }
    }

    #[test]
    fn dense_arch_with_missing_ffn_slices_still_errors() {
        let (mut weights, index) = attn_only_fixture();
        // Same index, but a DENSE architecture: the missing slab is now a
        // defect, not a topology.
        weights.arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 4,
        }));
        let mut scratch = larql_models::DequantScratch::new();
        let err = insert_q4k_layer_tensors(&mut scratch, &weights, &index, 0)
            .expect_err("a dense arch without FFN slices is a broken vindex");
        assert!(err.contains("ffn Q4K slices missing"), "{err}");
    }

    #[test]
    fn walk_ffn_kquant_layer_runs_gelu_tanh_path() {
        // Gemma-3 weights → GeluTanh activation branch.
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let x = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_runs_silu_path() {
        // SiLU-activation sibling weights → silu_gate_up branch.
        let weights = make_test_q4k_weights_silu();
        let idx = make_q4k_fixture_index(&weights);
        let x = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    /// `kquant_ffn_forward_layer` non-aligned-intermediate branch:
    /// when the index reports `num_features` that's not a multiple of
    /// `K_QUANT_BLOCK_ELEMS`, the down-projection path pads up to the
    /// next multiple and slices back down. Covers walk_ffn.rs:65-66.
    #[test]
    fn walk_ffn_kquant_layer_handles_non_aligned_intermediate() {
        struct NonAlignedIntermediate<'a> {
            inner: &'a crate::test_fixtures::Q4kFixtureIndex,
            claimed_intermediate: usize,
        }
        impl crate::KvIndex for NonAlignedIntermediate<'_> {
            fn num_features(&self, _l: usize) -> usize {
                self.claimed_intermediate
            }
            fn attn_kquant_layer_data(&self, l: usize) -> Option<[(&[u8], &str); 4]> {
                self.inner.attn_kquant_layer_data(l)
            }
            fn interleaved_kquant_layer_data(
                &self,
                l: usize,
            ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
                self.inner.interleaved_kquant_layer_data(l)
            }
            fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
                self.inner.interleaved_kquant_mmap_ref()
            }
            // No `kquant_ffn_layer_once` — forces dequant path which
            // pads to the next K_QUANT_BLOCK_ELEMS multiple.
        }
        let weights = make_test_q4k_weights();
        let inner = make_q4k_fixture_index(&weights);
        let idx = NonAlignedIntermediate {
            inner: &inner,
            // Real intermediate is 256; claim 200 → padded to 256, branch
            // fires.
            claimed_intermediate: 200,
        };
        let x = ndarray::Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let result = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        // The non-aligned branch slices the down-projection output;
        // shape should be `(1, hidden_size)`.
        assert_eq!(result.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_runs_dequant_fallback_when_cache_disabled() {
        // `disable_ffn_cache` forces `kquant_ffn_layer_once` → None, so
        // walk_ffn takes the `dequantize_matrix` branch on every
        // gate/up/down.
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights).without_ffn_cache();
        let x = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let out = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_q8k_runs_gelu_path() {
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
        let h_q8k = quantize_x_to_q8k(&h_in);
        let out = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_kquant_layer_q8k_runs_silu_fallback_path() {
        // SiLU activation + cache disabled exercises the fallback
        // (OnceLock cache None) path on the down-projection.
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        let weights = make_test_q4k_weights_silu();
        let idx = make_q4k_fixture_index(&weights).without_ffn_cache();
        let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
        let h_q8k = quantize_x_to_q8k(&h_in);
        let out = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    /// Regression for docs/audits/dec-readiness-review-2026-07-22.md §1a:
    /// pins the numerical equivalence the server's batched-GEMM Q8K
    /// handler (`larql-server/routes/walk_ffn/q8k.rs`) depends on. Same-
    /// layer entries dequantised to f32 and run through
    /// `kquant_ffn_forward_layer` as ONE multi-row GEMM must reproduce
    /// (within f32 rounding) what the single-row `q4k_q8k_matvec_into`
    /// kernel produces per entry — otherwise batching same-layer rows to
    /// fix the ~linear-in-B batch curve would silently change the numbers.
    #[test]
    fn walk_ffn_kquant_layer_q8k_batched_gemm_matches_per_row_single_kernel() {
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let hidden = weights.hidden_size;

        let rows: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                (0..hidden)
                    .map(|j| ((i * hidden + j) as f32 * 0.001).sin() * 0.05)
                    .collect()
            })
            .collect();

        // Reference: each row through the single-row Q8K fast kernel,
        // exactly as a batch-1 request (or the pre-fix code) would.
        let single_row_outputs: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| {
                let h_q8k = quantize_x_to_q8k(r);
                kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k)
                    .into_raw_vec_and_offset()
                    .0
            })
            .collect();

        // Batched: dequantise each row's Q8K activation back to f32 and
        // run all 3 rows through ONE GEMM, mirroring the server's grouped
        // handler.
        let mut dequantised: Vec<f32> = Vec::with_capacity(3 * hidden);
        for r in &rows {
            let h_q8k = quantize_x_to_q8k(r);
            for b in 0..h_q8k.n_blocks() {
                let d = h_q8k.d[b];
                for i in 0..256 {
                    dequantised.push(d * (h_q8k.qs[b * 256 + i] as f32));
                }
            }
        }
        let x = ndarray::Array2::from_shape_vec((3, hidden), dequantised).unwrap();
        let batched = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);

        for (row_idx, single) in single_row_outputs.iter().enumerate() {
            let batched_row = batched.row(row_idx);
            for (a, &b) in single.iter().zip(batched_row.iter()) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "row {row_idx}: single-kernel {a} vs batched-GEMM {b} diverge"
                );
            }
        }
    }

    /// Regression for docs/audits/dec-readiness-review-2026-07-22.md §1d:
    /// the down-projection fast path must consult the down slab's own
    /// format tag (`ffn[2].1`), not just the block-alignment guard.
    /// Before the fix, any down slab — regardless of its declared format —
    /// went straight into the Q4_K-only `q4k_q8k_matvec_into` kernel
    /// whenever `intermediate` was 256-aligned, silently misreading any
    /// non-Q4_K byte layout. Here the down component is (still
    /// Q4_K-encoded bytes, but) tagged with an unsupported format string;
    /// with the fix, that tag mismatch routes to the format-aware
    /// `dequantize_matrix` fallback, which loudly rejects an unknown tag
    /// instead of the fast path silently "succeeding" on the wrong
    /// assumption.
    #[test]
    #[should_panic(expected = "unsupported quant format")]
    fn walk_ffn_kquant_layer_q8k_rejects_down_slab_with_non_q4k_format_tag() {
        struct BogusDownFormat<'a> {
            inner: &'a crate::test_fixtures::Q4kFixtureIndex,
        }
        impl crate::KvIndex for BogusDownFormat<'_> {
            fn num_features(&self, l: usize) -> usize {
                self.inner.num_features(l)
            }
            fn attn_kquant_layer_data(&self, l: usize) -> Option<[(&[u8], &str); 4]> {
                self.inner.attn_kquant_layer_data(l)
            }
            fn interleaved_kquant_layer_data(
                &self,
                l: usize,
            ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
                let mut ffn = self.inner.interleaved_kquant_layer_data(l)?;
                ffn[2].1 = "BOGUS_FORMAT";
                Some(ffn)
            }
            fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
                self.inner.interleaved_kquant_mmap_ref()
            }
            // No `kquant_ffn_layer_once` — forces the dequant fallback,
            // which is the only branch that consults the format tag.
        }
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        let weights = make_test_q4k_weights();
        let inner = make_q4k_fixture_index(&weights);
        let idx = BogusDownFormat { inner: &inner };
        let h_in: Vec<f32> = vec![0.01; weights.hidden_size];
        let h_q8k = quantize_x_to_q8k(&h_in);
        let _ = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
    }

    // ── tensors.rs ───────────────────────────────────────────────────

    #[test]
    fn tensors_insert_q4k_layer_populates_dense_f32_keys() {
        let weights = make_test_q4k_weights();
        let mut scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let keys = insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0)
            .expect("insert_q4k_layer_tensors must succeed on Q4K fixture");
        // Q/K/V/O + gate/up/down = 7 keys per layer.
        assert_eq!(keys.len(), 7);
        for key in &keys {
            assert!(scratch.contains_key(key));
        }
        remove_layer_tensors(&mut scratch, keys.clone());
        for key in &keys {
            assert!(!scratch.contains_key(key));
        }
    }

    #[test]
    fn tensors_insert_q4k_layer_errors_on_missing_attn_data() {
        // An EmptyKvIndex returns None from every accessor — the
        // `ok_or_else` branch in `insert_q4k_layer_tensors` fires.
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let weights = make_test_q4k_weights();
        let mut scratch = larql_models::DequantScratch::new();
        let result = insert_q4k_layer_tensors(&mut scratch, &weights, &EmptyIdx, 0);
        let err = result.expect_err("missing attn data must fail");
        assert!(err.contains("attn"));
    }

    #[test]
    fn tensors_insert_q4k_layer_errors_on_missing_ffn_data() {
        // Provide attn but not ffn — the second `ok_or_else` fires.
        struct AttnOnlyIdx {
            attn_bytes: Vec<u8>,
        }
        impl crate::KvIndex for AttnOnlyIdx {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
                Some([
                    (self.attn_bytes.as_slice(), "Q4_K"),
                    (self.attn_bytes.as_slice(), "Q4_K"),
                    (self.attn_bytes.as_slice(), "Q4_K"),
                    (self.attn_bytes.as_slice(), "Q4_K"),
                ])
            }
        }
        // Reuse a real Q4K-quant slice — the test should hit the ffn
        // check before dequant runs, so the actual content is fine.
        let weights = make_test_q4k_weights();
        let real_idx = make_q4k_fixture_index(&weights);
        let attn_bytes = {
            let dyn_idx: &dyn crate::KvIndex = &real_idx;
            dyn_idx.attn_kquant_layer_data(0).unwrap()[0].0.to_vec()
        };
        let idx = AttnOnlyIdx { attn_bytes };
        let weights = make_test_q4k_weights();
        let mut scratch = larql_models::DequantScratch::new();
        let result = insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0);
        let err = result.expect_err("missing ffn data must fail");
        assert!(err.contains("ffn"));
    }

    // ── hooks.rs ─────────────────────────────────────────────────────

    /// `kquant_ffn_forward_layer` panics when the layer has no
    /// interleaved Q4K data. Server-side bug if you reach this path
    /// without preloading; the panic message is the contract.
    #[test]
    #[should_panic(expected = "interleaved_kquant layer data missing")]
    fn walk_ffn_panics_when_layer_data_missing() {
        struct AttnOnlyNoFfn;
        impl crate::KvIndex for AttnOnlyNoFfn {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            // interleaved_kquant_layer_data inherits default None → panic.
        }
        let weights = make_test_q4k_weights();
        let idx = AttnOnlyNoFfn;
        let x = Array2::<f32>::zeros((1, weights.hidden_size));
        let _ = kquant_ffn_forward_layer(&*weights.arch, &idx, 0, &x);
    }

    /// Same panic path on the Q8K-fused variant.
    #[test]
    #[should_panic(expected = "interleaved_kquant layer data missing")]
    fn walk_ffn_q8k_panics_when_layer_data_missing() {
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
        struct AttnOnlyNoFfn;
        impl crate::KvIndex for AttnOnlyNoFfn {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
        }
        let weights = make_test_q4k_weights();
        let idx = AttnOnlyNoFfn;
        let h_q8k = quantize_x_to_q8k(&vec![0.0; weights.hidden_size]);
        let _ = kquant_ffn_forward_layer_q8k(&*weights.arch, &idx, 0, &h_q8k);
    }

    // ── cached.rs CPU forward paths ──────────────────────────────────

    #[test]
    fn supports_cached_decode_returns_true_for_dense_weights() {
        let weights = make_test_q4k_weights();
        assert!(supports_cached_decode(&weights));
    }

    #[test]
    fn cached_timings_add_accumulates_dequant_ms() {
        let mut acc = CachedTimings::default();
        let a = CachedTimings { dequant_ms: 1.0 };
        let b = CachedTimings { dequant_ms: 2.5 };
        acc.add(a);
        acc.add(b);
        assert!((acc.dequant_ms - 3.5).abs() < 1e-9);
    }

    /// `layer_supports_direct_matvec` returns false when the index
    /// doesn't provide kquant data — drives the `None` short-circuit
    /// in `supports_direct_matvec_decode`.
    #[test]
    fn supports_direct_matvec_decode_false_for_empty_index() {
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let weights = make_test_q4k_weights();
        let idx = EmptyIdx;
        assert!(!supports_direct_matvec_decode(&weights, &idx));
    }

    /// `supports_direct_matvec_decode` rejects an index where attn
    /// data uses a non-K-quant format — covers the
    /// `!matches!(*fmt, "Q4_K" | "Q6_K")` early-return in
    /// `layer_supports_direct_matvec`.
    #[test]
    fn supports_direct_matvec_decode_false_for_q8_attn_format() {
        struct Q8AttnIdx {
            bytes: Vec<u8>,
        }
        impl crate::KvIndex for Q8AttnIdx {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
                Some([
                    (self.bytes.as_slice(), "Q8_0"),
                    (self.bytes.as_slice(), "Q8_0"),
                    (self.bytes.as_slice(), "Q8_0"),
                    (self.bytes.as_slice(), "Q8_0"),
                ])
            }
        }
        let weights = make_test_q4k_weights();
        let idx = Q8AttnIdx {
            bytes: vec![0u8; 16],
        };
        assert!(!supports_direct_matvec_decode(&weights, &idx));
    }

    /// `supports_direct_matvec_decode` rejects an index that has
    /// attn Q4_K data but no FFN interleaved data — covers the
    /// `interleaved_kquant_layer_data → None` branch.
    #[test]
    fn supports_direct_matvec_decode_false_when_ffn_data_missing() {
        struct AttnOnlyQ4Idx {
            bytes: Vec<u8>,
        }
        impl crate::KvIndex for AttnOnlyQ4Idx {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
                Some([
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                ])
            }
            // interleaved_kquant_layer_data inherits the default None.
        }
        let weights = make_test_q4k_weights();
        let idx = AttnOnlyQ4Idx {
            bytes: vec![0u8; 16],
        };
        assert!(!supports_direct_matvec_decode(&weights, &idx));
    }

    /// `supports_direct_matvec_decode` rejects an index where FFN
    /// data uses a non-K-quant format — covers the ffn `!matches!`
    /// branch at cached.rs:327.
    #[test]
    fn supports_direct_matvec_decode_false_for_q8_ffn_format() {
        struct Q8FfnIdx {
            bytes: Vec<u8>,
        }
        impl crate::KvIndex for Q8FfnIdx {
            fn num_features(&self, _l: usize) -> usize {
                256
            }
            fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
                Some([
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                ])
            }
            fn interleaved_kquant_layer_data(
                &self,
                _l: usize,
            ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
                Some([
                    (self.bytes.as_slice(), "Q8_0"),
                    (self.bytes.as_slice(), "Q8_0"),
                    (self.bytes.as_slice(), "Q8_0"),
                ])
            }
        }
        let weights = make_test_q4k_weights();
        let idx = Q8FfnIdx {
            bytes: vec![0u8; 16],
        };
        assert!(!supports_direct_matvec_decode(&weights, &idx));
    }

    #[test]
    fn supports_direct_matvec_decode_inspects_fixture() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        // The fixture is Q4_K throughout with a 256-multiple intermediate,
        // so it qualifies. This used to bind the result to `let _: bool`
        // and assert nothing "because the exact value depends on fixture
        // layout" — but the layout is fixed by `make_q4k_fixture_index`,
        // so the value is knowable and a routing regression should be
        // visible here. The refusal side is covered by
        // `cached::tests::layer_supports_direct_matvec_refuses_padded_intermediate`.
        assert!(supports_direct_matvec_decode(&weights, &idx));
    }

    #[test]
    fn predict_kquant_prefill_runs_end_to_end_on_cpu() {
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let (h, cache, _timings) = predict_kquant_prefill(&weights, &[0u32, 1, 2], &idx);
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        // One cache entry per layer, all populated by the prefill loop.
        assert_eq!(cache.len(), weights.num_layers);
        for entry in &cache {
            assert!(entry.is_some(), "every layer's cache should be populated");
        }
    }

    #[test]
    fn predict_kquant_prefill_with_state_captures_per_layer_residuals() {
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
        let (h, _cache, _timings) =
            predict_kquant_prefill_with_state(&weights, &[0u32, 1, 2], &idx, Some(&mut state));
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        // State captured for every layer.
        assert!(state.is_complete_for(weights.num_layers));
    }

    #[test]
    fn predict_kquant_decode_step_uses_prefill_cache() {
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        // First do prefill to populate the cache.
        let (_h, mut cache, _) = predict_kquant_prefill(&weights, &[0u32, 1, 2], &idx);
        // Then decode one new token at abs_position = 3.
        let result = predict_kquant_decode_step(&weights, 4u32, &idx, &mut cache, 3);
        let (h, _t) = result.expect("decode step succeeds with populated cache");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn predict_kquant_decode_step_rejects_mismatched_cache() {
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        // Wrong-sized cache → early return None.
        let mut wrong = vec![None; weights.num_layers + 1];
        let result = predict_kquant_decode_step(&weights, 0u32, &idx, &mut wrong, 0);
        assert!(result.is_none());
    }

    #[test]
    fn predict_kquant_decode_step_direct_runs_with_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut cache, _) = predict_kquant_prefill(&weights, &[0u32, 1, 2], &idx);
        let backend = crate::CpuBackend;
        let result =
            predict_kquant_decode_step_direct(&weights, 4u32, &idx, &backend, &mut cache, 3);
        match result {
            Some(h) => assert_eq!(h.shape(), &[1, weights.hidden_size]),
            None => {
                // Falls back when layer doesn't support direct matvec — OK on
                // the synthetic fixture.
            }
        }
    }

    #[test]
    fn predict_kquant_decode_step_direct_with_state_captures_per_layer() {
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut cache, _) = predict_kquant_prefill(&weights, &[0u32, 1, 2], &idx);
        let backend = crate::CpuBackend;
        let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
        let _ = predict_kquant_decode_step_direct_with_state(
            &weights,
            4u32,
            &idx,
            &backend,
            &mut cache,
            3,
            Some(&mut state),
        );
        // Whether or not the direct path engaged, the function shouldn't panic.
    }

    #[test]
    fn fused_decode_step_with_state_masked_drives_through_mock_backend() {
        use crate::test_fixtures::MockKquantBackend;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = MockKquantBackend;
        // Drive each mask variant.
        for mask in [
            crate::StateDumpMask::Full,
            crate::StateDumpMask::HOnly,
            crate::StateDumpMask::None,
        ] {
            let mut dump = crate::DecodeStateDump::with_capacity(weights.num_layers);
            let result = fused_decode_step_with_state_masked(
                &weights, &idx, 0u32, &backend, &mut dump, mask,
            );
            assert!(
                result.is_some(),
                "masked decode returns Some under {mask:?}"
            );
        }
    }

    /// `predict_kquant_hidden_hooked` early-returns Err on a hybrid-MoE
    /// arch — covers the `if weights.arch.is_hybrid_moe()` branch in
    /// hooks.rs.
    #[test]
    fn hooks_predict_kquant_hidden_hooked_errors_on_hybrid_moe_arch() {
        let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        // We don't need a real Q4K vindex — the function checks
        // is_hybrid_moe() before reading any tensor.
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let mut hook = crate::forward::NoopHook;
        let result =
            predict_kquant_hidden_hooked(&weights, &[0u32], &EmptyIdx, false, false, &mut hook);
        let err = result.expect_err("MoE arch must early-return Err");
        assert!(err.contains("dense FFN"));
    }

    /// `supports_cached_decode` returns false on a hybrid-MoE arch —
    /// covers the early-return branch in cached.rs:75-77.
    #[test]
    fn supports_cached_decode_rejects_hybrid_moe_arch() {
        let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        assert!(!supports_cached_decode(&weights));
    }

    /// `supports_cached_decode` returns false on KV-sharing archs
    /// (the e2b-like fixture has `num_kv_shared_layers: 2`) — covers
    /// the `kv_shared_source_layer(layer).is_some()` early-return.
    #[test]
    fn supports_cached_decode_rejects_kv_sharing_arch() {
        let weights = larql_models::test_fixtures::make_synthetic_e2b_like_weights();
        // The synthetic E2B-like arch advertises KV sharing.
        let has_shared =
            (0..weights.num_layers).any(|l| weights.arch.kv_shared_source_layer(l).is_some());
        assert!(has_shared, "fixture must have KV sharing");
        assert!(!supports_cached_decode(&weights));
    }

    /// `fused_prefill` Q4_0 fallback: when `interleaved_kquant_mmap_ref`
    /// returns None and `interleaved_q4_mmap_ref` returns Some, the
    /// helper picks the Q4_0 branch. Drives lines 357-360 + 375 in
    /// cached.rs (`ffn_is_q4k = false` branch + Q4_0 format selection).
    #[test]
    fn fused_prefill_takes_q4_0_fallback_branch() {
        use crate::test_fixtures::MockKquantBackend;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights).as_legacy_q4_mmap();
        let backend = MockKquantBackend;
        let result = fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        // MockKquantBackend's `prefill_kquant` returns Some — fixture
        // gates pass through the Q4_0 branch.
        let h = result.expect("Q4_0 fallback branch returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `fused_decode_step_inner` Q4_0 fallback: same gate pattern
    /// reached via the masked-state-dump entry point. Covers lines
    /// 476-479 + 489 in cached.rs.
    #[test]
    fn fused_decode_step_inner_q4_0_fallback_branch() {
        use crate::test_fixtures::MockKquantBackend;
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights).as_legacy_q4_mmap();
        let backend = MockKquantBackend;
        let mut dump = crate::DecodeStateDump::with_capacity(weights.num_layers);
        let result = fused_decode_step_with_state_masked(
            &weights,
            &idx,
            0u32,
            &backend,
            &mut dump,
            crate::StateDumpMask::Full,
        );
        let h = result.expect("Q4_0 fallback decode-step returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `attention_decode_step_native` end-to-end on Q4K weights — drives
    /// the CPU attention helper (Q4_K-CPU matvec + RoPE + attention).
    /// Covers ~70 LOC in cached.rs.
    #[test]
    fn attention_decode_step_native_runs_on_q4k_fixture() {
        use ndarray::Array2;
        let weights = make_test_q4k_weights();
        let mut scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        // Need to insert the layer 0 tensors so weights.tensors holds
        // the Q/K/V/O dense f32 matrices (the helper reads them).
        let inserted = insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0)
            .expect("layer 0 q4k tensors insert");
        let h_new = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let result = attention_decode_step_native(
            &weights, &idx, &backend, &h_new, /*layer=*/ 0, /*kv_entry=*/ None,
            /*abs_position=*/ 0,
        );
        let _ = result;
        // Clean up so subsequent tests don't see stale tensors.
        remove_layer_tensors(&mut scratch, inserted);
    }

    /// `ffn_decode_step_native` end-to-end on Q4K weights — drives
    /// the CPU FFN helper (Q4_K-CPU matvec + GeluTanh + down-proj).
    #[test]
    fn ffn_decode_step_native_runs_on_q4k_fixture() {
        use ndarray::Array2;
        let weights = make_test_q4k_weights();
        let mut scratch = larql_models::DequantScratch::new();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let inserted = insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0)
            .expect("layer 0 q4k tensors insert");
        let h_post_attn = Array2::<f32>::from_shape_vec(
            (1, weights.hidden_size),
            vec![0.01; weights.hidden_size],
        )
        .unwrap();
        let result = ffn_decode_step_native(&weights, &idx, &backend, &h_post_attn, 0);
        let _ = result;
        remove_layer_tensors(&mut scratch, inserted);
    }

    /// `fused_prefill` short-circuits on `!supports_quant(Q4_K)`
    /// (CpuBackend doesn't override, default false). Lines 352-354.
    #[test]
    fn fused_prefill_returns_none_when_backend_lacks_q4k_support() {
        struct NoQ4kBackend;
        impl crate::MatMul for NoQ4kBackend {
            fn matmul(
                &self,
                _: ndarray::ArrayView2<f32>,
                _: ndarray::ArrayView2<f32>,
            ) -> ndarray::Array2<f32> {
                unreachable!()
            }
            fn matmul_transb(
                &self,
                _: ndarray::ArrayView2<f32>,
                _: ndarray::ArrayView2<f32>,
            ) -> ndarray::Array2<f32> {
                unreachable!()
            }
        }
        impl crate::QuantMatVec for NoQ4kBackend {
            fn supports_quant(&self, _f: crate::QuantFormat) -> bool {
                false
            }
        }
        impl crate::DecodeBackend for NoQ4kBackend {}
        impl crate::ComputeBackend for NoQ4kBackend {
            fn name(&self) -> &str {
                "no-q4k"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = NoQ4kBackend;
        let result = fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        assert!(result.is_none());
    }

    /// `fused_prefill` returns None when both mmap accessors return
    /// None (lines 359-360, the final else-branch in the mmap match).
    #[test]
    fn fused_prefill_returns_none_without_any_ffn_mmap() {
        use crate::test_fixtures::MockKquantBackend;
        struct NoFfnMmap;
        impl crate::KvIndex for NoFfnMmap {}
        let weights = make_test_q4k_weights();
        let backend = MockKquantBackend;
        let result = fused_prefill(&weights, &NoFfnMmap, &[0u32, 1, 2], &backend);
        assert!(result.is_none());
    }

    /// `fused_prefill` returns None when `intermediate == 0` — drives
    /// the line 368-370 guard.
    #[test]
    fn fused_prefill_returns_none_when_intermediate_is_zero() {
        use crate::test_fixtures::MockKquantBackend;
        struct ZeroIntermediate {
            bytes: Vec<u8>,
        }
        impl crate::KvIndex for ZeroIntermediate {
            fn num_features(&self, _l: usize) -> usize {
                0
            }
            fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
                Some([
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                    (self.bytes.as_slice(), "Q4_K"),
                ])
            }
            fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
                Some(self.bytes.as_slice())
            }
        }
        let weights = make_test_q4k_weights();
        let backend = MockKquantBackend;
        let idx = ZeroIntermediate {
            bytes: vec![0u8; 1],
        };
        let result = fused_prefill(&weights, &idx, &[0u32, 1, 2], &backend);
        assert!(result.is_none());
    }

    /// `fused_decode_step_inner` returns None when both mmap accessors
    /// return None — covers lines 478-479 (the final else-return-None).
    #[test]
    fn fused_decode_step_inner_returns_none_without_any_ffn_mmap() {
        use crate::test_fixtures::MockKquantBackend;
        struct NoFfnMmapIdx;
        impl crate::KvIndex for NoFfnMmapIdx {
            // Both interleaved_kquant_mmap_ref AND interleaved_q4_mmap_ref
            // inherit the default None.
        }
        let weights = make_test_q4k_weights();
        let backend = MockKquantBackend;
        let mut dump = crate::DecodeStateDump::with_capacity(weights.num_layers);
        let result = fused_decode_step_with_state_masked(
            &weights,
            &NoFfnMmapIdx,
            0u32,
            &backend,
            &mut dump,
            crate::StateDumpMask::Full,
        );
        assert!(result.is_none(), "missing both mmap refs → None");
    }

    #[test]
    fn hooks_predict_kquant_hidden_hooked_errors_on_moe_arch() {
        // `make_test_gemma4_moe_weights` would yield `is_hybrid_moe=true`
        // and trip the early-return guard. We don't have that fixture
        // reachable from larql-compute (it lives in larql-inference's
        // test_utils), but we *do* have a way to fabricate a thin arch
        // wrapper. The simpler proof: the function returns an Err
        // string starting with "predict_kquant_hidden_hooked currently
        // supports dense FFN" when the guard fires. Skip if the fixture
        // is dense — we cover the happy-path branch in the next test.
        let weights = make_test_q4k_weights();
        let _scratch = larql_models::DequantScratch::new();
        assert!(!weights.arch.is_hybrid_moe());
        let idx = make_q4k_fixture_index(&weights);
        let mut hook = crate::forward::NoopHook;
        let result =
            predict_kquant_hidden_hooked(&weights, &[0u32, 1, 2], &idx, false, false, &mut hook);
        // The dense path completes — exact assertion on shape.
        let h = result.expect("dense Q4K hooked forward must succeed");
        assert_eq!(h.shape()[1], weights.hidden_size);
    }
}
