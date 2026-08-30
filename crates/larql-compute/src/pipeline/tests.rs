use super::*;

fn minimal_qw(data: &[u8]) -> QuantWeight<'_> {
    QuantWeight::new(QuantFormat::Q4_0, data, crate::QuantAux::None)
}

fn minimal_layer<'a>(
    data: &'a [u8],
    norms: &'a [f32],
    ffn_type: FfnType,
    moe: Option<MoeLayerWeights<'a>>,
) -> FullPipelineLayer<'a> {
    let qw = minimal_qw(data);
    // Spread `..Default::default()` collapses the 30-field boilerplate
    // to just the fields this test actually exercises. Pin this pattern
    // so any future test that wants a minimal layer copies it.
    FullPipelineLayer {
        wq: qw,
        wk: qw,
        wv: qw,
        wo: qw,
        gate: qw,
        up: qw,
        down: qw,
        input_norm: norms,
        post_attn_norm: norms,
        ffn_type,
        attn_scale: 0.5,
        head_dim: 4,
        num_q_heads: 1,
        num_kv_heads: 1,
        moe,
        ..FullPipelineLayer::default()
    }
}

#[test]
fn activation_from_bool() {
    assert_eq!(Activation::from(true), Activation::GeluTanh);
    assert_eq!(Activation::from(false), Activation::Silu);
}

#[test]
fn registry_tag_round_trips_every_variant() {
    // registry_tag() is the exact inverse of from_registry_tag() for
    // every variant — including the new ternary I2_S.
    for f in [
        QuantFormat::Q4_0,
        QuantFormat::Q4_K,
        QuantFormat::Q4_KF,
        QuantFormat::Q6_K,
        QuantFormat::Q8_0,
        QuantFormat::BF16,
        QuantFormat::F16,
        QuantFormat::F32,
        QuantFormat::I2S,
    ] {
        assert_eq!(QuantFormat::from_registry_tag(f.registry_tag()), Some(f));
    }
}

#[test]
fn i2s_tag_and_family_predicates() {
    assert_eq!(
        QuantFormat::from_registry_tag("I2_S"),
        Some(QuantFormat::I2S)
    );
    assert_eq!(QuantFormat::I2S.registry_tag(), "I2_S");
    assert!(QuantFormat::I2S.is_ternary());
    assert!(!QuantFormat::Q4_K.is_ternary());
    // I2_S is none of the block-quant families and has no flat block layout.
    assert!(!QuantFormat::I2S.is_kquant_family());
    assert!(!QuantFormat::I2S.is_legacy_q8());
    assert!(QuantFormat::I2S.packed_block_layout().is_none());
}

#[test]
fn is_gated_matches_ffn_type() {
    let norms = [1.0f32; 4];
    let gated = minimal_layer(&[], &norms, FfnType::Gated, None);
    let standard = minimal_layer(&[], &norms, FfnType::Standard, None);
    assert!(gated.is_gated());
    assert!(!standard.is_gated());
}

#[test]
fn is_hybrid_moe_reflects_option() {
    let norms = [1.0f32; 4];
    let no_moe = minimal_layer(&[], &norms, FfnType::Gated, None);
    assert!(!no_moe.is_hybrid_moe());

    let moe = MoeLayerWeights {
        expert_scales: crate::MoeExpertScales::Inline,
        fused_row_layout: crate::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 2,
        top_k: 1,
        intermediate_size: 4,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: crate::MoeGateRule::Gated(Activation::Silu),
        expert_data_format: QuantFormat::BF16,
    };
    let with_moe = minimal_layer(&[], &norms, FfnType::Gated, Some(moe));
    assert!(with_moe.is_hybrid_moe());
}

#[test]
fn quant_format_equality() {
    assert_eq!(QuantFormat::Q4_K, QuantFormat::Q4_K);
    assert_ne!(QuantFormat::Q4_K, QuantFormat::Q6_K);
    assert_ne!(QuantFormat::Q4_0, QuantFormat::Q4_KF);
}

/// Pin the k-quant family taxonomy. Adding a new format requires
/// updating exactly one of these classifiers.
#[test]
fn quant_format_classifiers() {
    // k-quant family (256-element super-blocks)
    assert!(QuantFormat::Q4_K.is_kquant_family());
    assert!(QuantFormat::Q4_KF.is_kquant_family());
    assert!(QuantFormat::Q6_K.is_kquant_family());
    // Legacy block-32 Q8 path
    assert!(QuantFormat::Q4_0.is_legacy_q8());
    assert!(QuantFormat::Q8_0.is_legacy_q8());
    // Float-input formats are neither
    for fmt in [QuantFormat::BF16, QuantFormat::F16, QuantFormat::F32] {
        assert!(!fmt.is_kquant_family());
        assert!(!fmt.is_legacy_q8());
    }
    // Q4_KF is a subset of the k-quant family
    assert!(QuantFormat::Q4_KF.is_q4kf());
    assert!(!QuantFormat::Q4_K.is_q4kf());
    assert!(!QuantFormat::Q6_K.is_q4kf());
}

#[test]
fn quant_format_reports_packed_matrix_bytes() {
    assert_eq!(QuantFormat::Q4_0.packed_matrix_bytes(2, 32), Some(36));
    assert_eq!(QuantFormat::Q4_K.packed_matrix_bytes(2, 256), Some(288));
    // Q4_KF packs identically to Q4_K (144 B/super-block): the tag
    // selects the llama.cpp-exact kernels, not a storage layout. The
    // previous pinned value (320 = 2 x 160) described the experimental
    // pre-baked layout no kernel reads — capability audit F15.
    assert_eq!(QuantFormat::Q4_KF.packed_matrix_bytes(2, 256), Some(288));
    assert_eq!(QuantFormat::Q6_K.packed_matrix_bytes(2, 256), Some(420));
    assert_eq!(QuantFormat::F16.packed_matrix_bytes(2, 256), None);
}

#[test]
fn from_registry_tag_round_trips_known_tags_and_rejects_unknown() {
    for (tag, fmt) in [
        ("Q4_0", QuantFormat::Q4_0),
        ("Q4_K", QuantFormat::Q4_K),
        ("Q4_KF", QuantFormat::Q4_KF),
        ("Q6_K", QuantFormat::Q6_K),
        ("Q8_0", QuantFormat::Q8_0),
        ("BF16", QuantFormat::BF16),
        ("F16", QuantFormat::F16),
        ("F32", QuantFormat::F32),
    ] {
        assert_eq!(QuantFormat::from_registry_tag(tag), Some(fmt));
    }
    assert_eq!(QuantFormat::from_registry_tag("Q42_X"), None);
    assert_eq!(QuantFormat::from_registry_tag("q4_k"), None); // case-sensitive
    assert_eq!(QuantFormat::from_registry_tag(""), None);
}

/// The two string-keyed matvec dispatchers derive their packed row
/// stride as `(cols / block_elems) * block_bytes`; pin that this equals
/// what `from_registry_tag` + `packed_block_layout` produce for the
/// Q4_K / Q6_K tags they accept (256-multiple cols).
#[test]
fn registry_tag_block_layout_matches_dispatcher_stride() {
    let cols = 512usize;
    for (tag, raw_block_bytes) in [("Q4_K", 144usize), ("Q6_K", 210usize)] {
        let fmt = QuantFormat::from_registry_tag(tag).unwrap();
        let (block_elems, block_bytes) = fmt.packed_block_layout().unwrap();
        assert_eq!(block_elems, 256);
        assert_eq!(block_bytes, raw_block_bytes);
        assert_eq!(
            (cols / block_elems) * block_bytes,
            (cols / 256) * raw_block_bytes
        );
    }
}

/// `..Default::default()` must work with stack-local borrowed data —
/// the compiler reborrows the `'static` defaults at the caller's
/// shorter lifetime. Pin the pattern.
#[test]
fn default_layer_accepts_local_borrows_via_spread() {
    let data: Vec<u8> = vec![0, 1, 2];
    let norms: Vec<f32> = vec![1.0; 4];

    let layer = FullPipelineLayer {
        input_norm: &norms,
        post_attn_norm: &norms,
        wq: QuantWeight::new(QuantFormat::Q4_0, &data, crate::QuantAux::None),
        head_dim: 4,
        num_q_heads: 1,
        num_kv_heads: 1,
        ..Default::default()
    };

    // Defaulted fields carry through.
    assert_eq!(layer.eps, RMSNORM_EPSILON_DEFAULT);
    assert_eq!(layer.norm_type, NormType::RmsNorm);
    assert_eq!(layer.ffn_type, FfnType::Gated);
    assert_eq!(layer.activation, Activation::Silu);
    assert!(!layer.has_v_norm);
    assert!(layer.moe.is_none());

    // Explicit fields are honoured.
    assert_eq!(layer.input_norm.len(), 4);
    assert_eq!(layer.wq.data.len(), 3);
    assert_eq!(layer.head_dim, 4);
}

#[test]
fn layer_spec_views_preserve_flat_field_values() {
    let data: Vec<u8> = vec![0, 1, 2, 3];
    let norms: Vec<f32> = vec![1.0; 8];
    let q_norm: Vec<f32> = vec![0.5; 4];

    let qw = QuantWeight::new(QuantFormat::Q4_K, &data, crate::QuantAux::None);
    let layer = FullPipelineLayer {
        wq: qw,
        wk: qw,
        wv: qw,
        wo: qw,
        gate: qw,
        up: qw,
        down: qw,
        input_norm: &norms,
        post_attn_norm: &norms,
        norm_offset: 1.0,
        qk_norm_offset: 0.25,
        eps: 1e-5,
        has_post_norms: true,
        activation: Activation::GeluTanh,
        attn_scale: 0.125,
        head_dim: 4,
        num_q_heads: 2,
        num_kv_heads: 1,
        rope_base: ROPE_BASE_GLOBAL,
        rotary_dim: 2,
        rope_freq: crate::attention::rope::RopeFreqPlan::unscaled(
            4_usize,
            2_usize,
            ROPE_BASE_GLOBAL as f64,
        ),
        sliding_window: 32,
        has_v_norm: true,
        q_norm_weight: Some(&q_norm),
        ffn_is_remote: true,
        moe_combined_output_norm: true,
        moe_outer_post_norm: Some(&norms),
        ..FullPipelineLayer::default()
    };

    let weights = layer.weights();
    assert_eq!(weights.attention.wq.format(), QuantFormat::Q4_K);
    assert_eq!(weights.ffn.down.data.len(), data.len());

    let norms_view = layer.norms();
    assert_eq!(norms_view.input_norm.len(), norms.len());
    assert_eq!(norms_view.norm_offset, 1.0);
    assert_eq!(norms_view.qk_norm_offset, 0.25);
    assert_eq!(norms_view.eps, 1e-5);
    assert!(norms_view.has_post_norms);

    let attn = layer.attention_spec();
    assert_eq!(attn.head_dim, 4);
    assert_eq!(attn.num_q_heads, 2);
    assert_eq!(attn.num_kv_heads, 1);
    assert_eq!(attn.rope_base, ROPE_BASE_GLOBAL);
    assert_eq!(attn.rotary_dim, 2);
    assert_eq!(attn.sliding_window, 32);
    assert!(attn.has_v_norm);
    assert!(attn.q_norm_enabled);
    assert!(!attn.k_norm_enabled);

    assert_eq!(layer.ffn_spec().activation, Activation::GeluTanh);
    assert!(layer.remote_ffn_spec().is_remote);
    assert!(layer.moe_spec().combined_output_norm);
    assert_eq!(layer.moe_spec().outer_post_norm.unwrap().len(), norms.len());
}

/// `MoeWeightLayout` constructors + both `down_cols` policy arms.
#[test]
fn moe_weight_layout_down_cols_policies() {
    let unpadded = MoeWeightLayout::unpadded();
    assert_eq!(unpadded.down_cols(704, QuantFormat::Q4_K), 704);

    let padded = MoeWeightLayout::quant_block_padded_down();
    // 704 rounds up to the next 256-elem Q4_K super-block boundary.
    assert_eq!(padded.down_cols(704, QuantFormat::Q4_K), 768);
    // Already block-aligned stays put.
    assert_eq!(padded.down_cols(512, QuantFormat::Q4_K), 512);
}

/// `has_dense_ffn` is a representation fact: present up/down weights ⇒
/// dense branch exists; empty slices (the pure-MoE extraction shape) ⇒
/// it doesn't, and consumers must not encode the dense kernels.
#[test]
fn has_dense_ffn_reflects_weight_presence() {
    let data = [0u8; 18];
    let norms = [1.0f32; 4];
    let dense = minimal_layer(&data, &norms, FfnType::Gated, None);
    assert!(dense.has_dense_ffn());

    let empty = FullPipelineLayer {
        input_norm: &norms,
        post_attn_norm: &norms,
        ..FullPipelineLayer::default()
    };
    assert!(
        !empty.has_dense_ffn(),
        "empty up/down slices are the pure-MoE shape — no dense branch"
    );

    // One present, one absent is still not a runnable dense branch.
    let half = FullPipelineLayer {
        up: minimal_qw(&data),
        input_norm: &norms,
        post_attn_norm: &norms,
        ..FullPipelineLayer::default()
    };
    assert!(!half.has_dense_ffn());
}

// ── Expert-bank storage description ──

#[test]
fn interleaved_row_walk_agrees_with_the_convention_owner() {
    use larql_models::quant::mxfp4::FusedHalf;

    // `MoeFusedRowLayout` restates FusedHalf's convention as a (base,
    // stride) pair because a kernel walks rows rather than indexing them.
    // Pin the two against each other so the restatement cannot drift: it
    // is agreement with the OWNER that is being asserted, not that both
    // happen to produce the same numbers today.
    for half in [FusedHalf::Gate, FusedHalf::Up] {
        let (base, stride) = MoeFusedRowLayout::Interleaved.row_walk(half, 64);
        for row in 0..64 {
            assert_eq!(
                base + row * stride,
                half.fused_row(row),
                "{half:?} row {row} disagrees with FusedHalf"
            );
        }
    }
}

#[test]
fn contiguous_halves_puts_up_one_half_region_in() {
    use larql_models::quant::mxfp4::FusedHalf;

    const INTER: usize = 64;
    assert_eq!(
        MoeFusedRowLayout::ContiguousHalves.row_walk(FusedHalf::Gate, INTER),
        (0, 1)
    );
    assert_eq!(
        MoeFusedRowLayout::ContiguousHalves.row_walk(FusedHalf::Up, INTER),
        (INTER, 1)
    );

    // The two layouts must not agree anywhere past row 0, or a test that
    // claims to distinguish them would be pinning a coincidence.
    let (ci_base, ci_stride) = MoeFusedRowLayout::ContiguousHalves.row_walk(FusedHalf::Up, INTER);
    let (in_base, in_stride) = MoeFusedRowLayout::Interleaved.row_walk(FusedHalf::Up, INTER);
    assert_ne!((ci_base, ci_stride), (in_base, in_stride));
}

#[test]
fn inline_scales_have_no_partner_stream() {
    let scales = MoeExpertScales::Inline;
    assert!(!scales.is_paired());
    // `None` is "no such stream exists", not "the stream is empty" — the
    // distinction the enum exists to keep.
    assert!(scales.gate_up(0).is_none());
    assert!(scales.down(0).is_none());
}

#[test]
fn paired_scales_are_index_aligned_with_their_payloads() {
    let gu: Vec<&[u8]> = vec![&[1, 2], &[3, 4]];
    let dn: Vec<&[u8]> = vec![&[5], &[6]];
    let scales = MoeExpertScales::Paired {
        gate_up: gu,
        down: dn,
    };
    assert!(scales.is_paired());
    assert_eq!(scales.gate_up(1), Some(&[3u8, 4][..]));
    assert_eq!(scales.down(1), Some(&[6u8][..]));
}

#[test]
#[should_panic(expected = "too short for expert 2")]
fn a_short_paired_table_panics_rather_than_reporting_inline() {
    let scales = MoeExpertScales::Paired {
        gate_up: vec![&[1u8][..]],
        down: vec![&[2u8][..]],
    };
    let _ = scales.gate_up(2);
}

#[test]
#[should_panic(expected = "too short for expert 3")]
fn a_short_paired_down_table_panics_too() {
    let scales = MoeExpertScales::Paired {
        gate_up: vec![&[1u8][..]],
        down: vec![&[2u8][..]],
    };
    let _ = scales.down(3);
}
