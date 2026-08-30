//! Coverage for the simple per-arch helpers (kv shapes, format
//! parsing, routing policy). The big MoE branches in
//! `build_moe_weights` need a Gemma 4 MoE fixture and live in the
//! `larql-inference` integration tests where that fixture is
//! reachable.
use super::moe_build::moe_routing_policy;
use super::*;
use larql_models::test_fixtures::make_test_weights;

mod moe_store;

/// Capacity must accommodate the compaction trigger, not merely the
/// window: occupancy is allowed to reach `SLACK * W` before
/// compaction reclaims it, so a capacity of `W` would be an overrun
/// rather than a smaller allocation.
#[test]
fn kv_capacity_accommodates_the_compaction_trigger() {
    assert_eq!(
        kv_capacity_for_window(1024, 4096),
        1024 * KV_COMPACTION_SLACK
    );
    assert_eq!(kv_capacity_for_window(1024, 4096), 2048);
}

/// `0` is the unbounded sentinel — a global layer keeps the full
/// default because nothing bounds what it may read.
#[test]
fn kv_capacity_leaves_unbounded_layers_at_the_default() {
    assert_eq!(kv_capacity_for_window(0, 4096), 4096);
}

/// A window wider than the default cannot ask for more than it, and
/// a window near `usize::MAX` must not overflow into a tiny capacity.
#[test]
fn kv_capacity_is_clamped_by_the_default() {
    assert_eq!(kv_capacity_for_window(8192, 4096), 4096);
    assert_eq!(kv_capacity_for_window(usize::MAX, 4096), 4096);
}

/// A window small enough that the trigger stays under the default
/// gets the smaller allocation, which is the whole point.
#[test]
fn kv_capacity_shrinks_a_narrow_window() {
    assert_eq!(kv_capacity_for_window(64, 4096), 128);
}

#[test]
fn kv_capacities_for_arch_returns_one_entry_per_layer() {
    let weights = make_test_weights();
    let caps = kv_capacities_for_arch(&weights, 4096);
    assert_eq!(caps.len(), weights.num_layers);
    // Every entry is a real allocation and never exceeds the default.
    for c in &caps {
        assert!(*c > 0 && *c <= 4096, "capacity {c} out of range");
    }
    // And each agrees with the per-window rule applied to that layer.
    for (layer, &c) in caps.iter().enumerate() {
        let window =
            crate::forward_overrides::effective_attention_window_for_layer(&*weights.arch, layer)
                .unwrap_or(0);
        assert_eq!(c, kv_capacity_for_window(window, 4096), "layer {layer}");
    }
}

#[test]
fn kv_cache_shapes_for_arch_returns_one_pair_per_layer() {
    let weights = make_test_weights();
    let shapes = kv_cache_shapes_for_arch(&weights);
    assert_eq!(shapes.len(), weights.num_layers);
    for (num_kv, head_dim) in &shapes {
        assert!(*num_kv > 0);
        assert!(*head_dim > 0);
    }
}

#[test]
fn attn_str_to_format_maps_known_tags() {
    assert_eq!(attn_str_to_format("Q4_K"), QuantFormat::Q4_K);
    assert_eq!(attn_str_to_format("Q6_K"), QuantFormat::Q6_K);
}

#[test]
#[should_panic(expected = "no compute::QuantFormat mapping")]
fn attn_str_to_format_panics_on_unknown_tag() {
    let _ = attn_str_to_format("Q42_X");
}

#[test]
fn ffn_str_to_format_maps_known_tags() {
    assert_eq!(
        ffn_str_to_format("Q4_K", QuantFormat::Q4_K),
        QuantFormat::Q4_K
    );
    assert_eq!(
        ffn_str_to_format("Q6_K", QuantFormat::Q4_K),
        QuantFormat::Q6_K
    );
    assert_eq!(
        ffn_str_to_format("Q4_0", QuantFormat::Q4_K),
        QuantFormat::Q4_0
    );
    // Empty tag falls through to the caller's fallback.
    assert_eq!(ffn_str_to_format("", QuantFormat::Q4_0), QuantFormat::Q4_0);
    assert_eq!(ffn_str_to_format("", QuantFormat::Q4_K), QuantFormat::Q4_K);
}

#[test]
#[should_panic(expected = "no compute::QuantFormat mapping")]
fn ffn_str_to_format_panics_on_unknown_tag() {
    let _ = ffn_str_to_format("unknown", QuantFormat::Q4_K);
}

/// Each router kind must map to a *distinct* policy.
///
/// The predecessor of this test called `moe_routing_policy` twice and
/// asserted nothing, so it passed while the string `match` silently sent
/// GPT-OSS's router to the default arm.
#[test]
fn every_router_kind_maps_to_its_own_policy() {
    use larql_models::MoeRouterKind::*;
    let gemma4 = moe_routing_policy(Gemma4Hybrid);
    let plain = moe_routing_policy(TopKSoftmax);
    let selected = moe_routing_policy(TopKThenSoftmax);
    assert_ne!(gemma4, plain);
    assert_ne!(
        plain, selected,
        "top-k-then-softmax must not equal the default"
    );
    assert_ne!(gemma4, selected);
}

/// The distinction that was being lost: selected weights summing to 1
/// versus keeping the raw softmax mass. Confusing them rescales the whole
/// expert branch.
#[test]
fn top_k_then_softmax_renormalises_where_the_default_does_not() {
    use larql_models::MoeRouterKind::*;
    assert_eq!(
        moe_routing_policy(TopKThenSoftmax).selected_weight,
        crate::MoeTopKWeightPolicy::RenormalizedSoftmax
    );
    assert_eq!(
        moe_routing_policy(TopKSoftmax).selected_weight,
        crate::MoeTopKWeightPolicy::RawSoftmax
    );
}

/// GPT-OSS's router lives inside the MLP block in the reference, so BOTH
/// the router and the experts read the pre-experts-normed hidden — on the
/// quantised path that norm is the policy's to apply, since the caller
/// hands the MoE the raw residual. Serving with `Residual` here ran every
/// router and expert on the un-normed stream: structurally sane routing,
/// incoherent generation, no crash. This pins the topology so a revert
/// fails a test instead of a generation.
#[test]
fn top_k_then_softmax_routes_and_runs_on_the_pre_experts_normed_input() {
    use larql_models::MoeRouterKind::TopKThenSoftmax;
    let policy = moe_routing_policy(TopKThenSoftmax);
    assert_eq!(policy.expert_input, crate::MoeInputSource::PreExpertsNorm);
    assert_eq!(policy.router_input, crate::MoeInputSource::PreExpertsNorm);
}

/// `resolve_attn_weights` falls through to the Q8 branch when the
/// index returns Q8 data instead of Q4_K.
#[test]
fn resolve_attn_weights_uses_q8_branch_when_index_returns_q8() {
    struct Q8Idx {
        bytes: Vec<u8>,
        scales: Vec<f32>,
    }
    impl crate::KvIndex for Q8Idx {
        fn attn_q8_layer_data(&self, _l: usize) -> Option<[(&[u8], &[f32]); 4]> {
            Some([
                (self.bytes.as_slice(), self.scales.as_slice()),
                (self.bytes.as_slice(), self.scales.as_slice()),
                (self.bytes.as_slice(), self.scales.as_slice()),
                (self.bytes.as_slice(), self.scales.as_slice()),
            ])
        }
    }
    let idx = Q8Idx {
        bytes: vec![0u8; 16],
        scales: vec![1.0f32; 4],
    };
    let result = resolve_attn_weights(&idx, 0);
    let (q, _k, _v, _o) = result.expect("Q8 fallback returns Some");
    assert_eq!(q.format(), QuantFormat::Q8_0);
}

/// `build_arch_params` rotary_dim branch fires when `rotary_fraction`
/// is < 1.0 (partial-rotary archs like StarCoder2).
#[test]
fn build_arch_params_handles_partial_rotary_fraction() {
    let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    // The partial-rotary branch is shape-dependent on the arch; what
    // we want is just to ensure no panic on a non-full-rotary arch.
    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    let _ = layer.rotary_dim;
}

/// `build_arch_params` on Llama2-style (Silu activation) fixture —
/// covers the Silu fallback branch in the activation match.
#[test]
fn build_arch_params_handles_silu_activation() {
    let weights = make_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    assert!(matches!(layer.activation, crate::Activation::Silu));
}

/// `build_arch_params` on Starcoder2-style fixture covers the
/// LayerNorm branch and the Standard (non-gated) FFN type.
#[test]
fn build_arch_params_handles_layernorm_and_standard_ffn() {
    let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    assert!(matches!(layer.norm_type, crate::NormType::LayerNorm));
    assert!(matches!(layer.ffn_type, crate::FfnType::Standard));
}

/// `build_moe_weights` happy path on the Gemma 4 hybrid-MoE fixture
/// — exercises the per-layer FFN router + packed expert slicing,
/// the BF16-stride math, and the routing-policy assignment.
#[test]
fn build_moe_weights_succeeds_on_hybrid_moe_fixture() {
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    assert!(weights.arch.is_hybrid_moe());
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        let result = build_moe_weights(&weights, arch, layer);
        assert!(
            result.is_some(),
            "MoE weights should resolve for layer {layer} on Gemma 4 hybrid-MoE"
        );
    }
}

/// `build_moe_weights` returns None on a non-MoE arch — covers the
/// `arch.moe_router_key(layer)?` short-circuit.
#[test]
fn build_moe_weights_returns_none_on_non_moe_arch() {
    let weights = make_test_weights();
    assert!(!weights.arch.is_hybrid_moe());
    assert!(build_moe_weights(&weights, &*weights.arch, 0).is_none());
}

/// `patch_pipeline_layers_for_remote_moe` injects MoE stubs on
/// MoE-capable layers when the local moe slot is still None.
#[test]
fn patch_pipeline_layers_for_remote_moe_injects_stubs() {
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    // Build pipeline layers with no MoE locally — simulates the
    // remote-MoE client deployment.
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..weights.num_layers)
        .map(|_| crate::FullPipelineLayer {
            wq: dummy,
            wk: dummy,
            wv: dummy,
            wo: dummy,
            gate: dummy,
            up: dummy,
            down: dummy,
            ..crate::FullPipelineLayer::default()
        })
        .collect();
    // Pre-patch: every layer has moe = None.
    for l in &layers {
        assert!(l.moe.is_none());
    }
    patch_pipeline_layers_for_remote_moe(&mut layers, &weights);
    // Post-patch: every MoE-capable layer has Some moe stub.
    let mut any_patched = false;
    for l in &layers {
        if l.moe.is_some() {
            any_patched = true;
        }
    }
    assert!(any_patched, "patch must inject at least one MoE stub");
}

#[test]
fn patch_pipeline_layers_for_remote_ffn_sets_remote_flag() {
    // Build a 1-layer pipeline and patch to remote FFN.
    let layer = crate::FullPipelineLayer::default();
    let mut layers = vec![layer];
    assert!(!layers[0].ffn_is_remote);
    patch_pipeline_layers_for_remote_ffn(&mut layers);
    for l in &layers {
        assert!(l.ffn_is_remote, "patch should set ffn_is_remote = true");
    }
}

/// Pure-MoE vindexes (e.g. Gemma-4 26B A4B) ship a zero-byte
/// `interleaved_q4k.bin` because there is no dense FFN. Before the
/// `is_empty()` guard, `resolve_ffn_weights` would panic on the first
/// `q4_ffn_mmap[fs..fs + q4_ffn_per_matrix]` slice. The guard returns
/// empty `QuantWeight` stubs — `patch_pipeline_layers_for_remote_moe`
/// overwrites the per-layer MoE weights afterward and the dense FFN
/// path is bypassed entirely by `moe_fn` during decode, so the empty
/// slices are never read.
#[test]
fn resolve_ffn_weights_returns_empty_stubs_when_q4_ffn_mmap_is_empty() {
    struct EmptyIdx;
    impl crate::KvIndex for EmptyIdx {}
    let idx = EmptyIdx;
    let empty_mmap: &[u8] = &[];
    // q4_ffn_per_matrix is irrelevant on this path — what we're pinning
    // is "no slice happens against the empty mmap" (i.e. no panic).
    let (gate, up, down) = resolve_ffn_weights(&idx, 7, empty_mmap, 1_115_136, QuantFormat::Q4_K);
    assert!(gate.data.is_empty());
    assert!(up.data.is_empty());
    assert!(down.data.is_empty());
    assert_eq!(gate.format(), QuantFormat::Q4_K);
    assert_eq!(up.format(), QuantFormat::Q4_K);
    assert_eq!(down.format(), QuantFormat::Q4_K);
}

/// The ordinary dense case: a non-empty interleaved mmap is cut into three
/// equal per-matrix slices at `layer * 3 * per_matrix`.
///
/// Pinning the *offsets* rather than just the lengths is the point — the
/// gate/up/down order and the per-layer stride are what a reader has to
/// agree with, and a stride bug yields correctly-sized garbage.
#[test]
fn resolve_ffn_weights_slices_gate_up_down_at_the_layer_stride() {
    struct EmptyIdx;
    impl crate::KvIndex for EmptyIdx {}
    let idx = EmptyIdx;
    let per_matrix = 4usize;
    // Three matrices per layer, two layers' worth.
    let mmap: Vec<u8> = (0..(per_matrix * 3 * 2) as u8).collect();

    let (gate, up, down) = resolve_ffn_weights(&idx, 1, &mmap, per_matrix, QuantFormat::Q4_K);

    // Layer 1 starts one full layer (12 bytes) in.
    assert_eq!(gate.data, &[12u8, 13, 14, 15]);
    assert_eq!(up.data, &[16u8, 17, 18, 19]);
    assert_eq!(down.data, &[20u8, 21, 22, 23]);

    // Layer 0 is the first stride, and the format is carried through rather
    // than re-derived per matrix.
    let (gate0, up0, down0) = resolve_ffn_weights(&idx, 0, &mmap, per_matrix, QuantFormat::Q6_K);
    assert_eq!(gate0.data, &[0u8, 1, 2, 3]);
    assert_eq!(up0.data, &[4u8, 5, 6, 7]);
    assert_eq!(down0.data, &[8u8, 9, 10, 11]);
    assert_eq!(gate0.format(), QuantFormat::Q6_K);
    assert_eq!(up0.format(), QuantFormat::Q6_K);
    assert_eq!(down0.format(), QuantFormat::Q6_K);
}

// ── moe_build.rs: the Gemma 4 MoE fixture reaches the real branches ──
//
// The header note above predates `make_test_gemma4_moe_weights` being
// reachable from this crate; the fixture IS available, so the MoE
// construction paths are pinned here rather than only via larql-inference
// integration tests.

#[test]
fn build_moe_weights_resolves_the_gemma4_fixture() {
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    let arch = &*weights.arch;
    let moe = build_moe_weights(&weights, arch, 0).expect("fixture layer 0 is MoE");

    assert_eq!(
        moe.num_experts,
        larql_models::test_fixtures::GEMMA4_MOE_NUM_EXPERTS
    );
    // Gemma declares no MoE biases; the resolution must produce the empty
    // slices (pre-bias behaviour), not phantom vectors.
    assert!(moe.router_bias.is_empty());
    assert!(moe.experts_gate_up_bias.is_empty());
    assert!(moe.experts_down_bias.is_empty());
    assert_eq!(
        moe.gate_rule,
        crate::MoeGateRule::Gated(crate::Activation::GeluTanh)
    );
    // Packed BF16 monolith: stored row width == hidden (no block padding).
    assert_eq!(moe.gate_up_cols(weights.hidden_size), weights.hidden_size);
    // The block actually runs: finite output of the right width.
    let h = vec![0.1f32; weights.hidden_size];
    let out = crate::cpu::ops::moe::cpu_moe_forward(&h, &moe, 1.0, 1e-6);
    assert_eq!(out.len(), weights.hidden_size);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn remote_moe_patching_builds_stubs_for_unserved_layers() {
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    // A remote-MoE client slice has no local expert bytes, so the builder
    // leaves `moe: None`; patching must synthesise norm-carrying stubs for
    // every routed layer rather than leaving the combine step un-normed.
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..weights.num_layers)
        .map(|l| build_arch_params(&weights, l, dummy, dummy, dummy, dummy, dummy, dummy, dummy))
        .collect();
    for layer in &mut layers {
        layer.moe = None;
    }
    patch_pipeline_layers_for_remote_moe(&mut layers, &weights);
    for (l, layer) in layers.iter().enumerate() {
        let moe = layer
            .moe
            .as_ref()
            .unwrap_or_else(|| panic!("layer {l}: stub not patched in"));
        assert!(
            moe.experts_gate_up.is_empty(),
            "stubs must carry no local expert bytes"
        );
        assert_eq!(
            moe.gate_rule,
            crate::MoeGateRule::Gated(crate::Activation::GeluTanh)
        );
    }
}

/// The attention projection biases resolve at the arch boundary exactly
/// like the sinks: `arch.attn_*_bias_key` → `weights.vectors` → layer
/// fields. StarCoder2 declares all four keys, so inserting vectors under
/// those names must surface on the built layer — and a fixture without
/// the vectors must yield `None`, never a fabricated bias.
#[test]
fn build_arch_params_threads_attention_projection_biases() {
    let mut weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);

    let q_key = weights
        .arch
        .attn_q_bias_key(0)
        .expect("starcoder2 declares q bias");
    let k_key = weights
        .arch
        .attn_k_bias_key(0)
        .expect("starcoder2 declares k bias");
    let v_key = weights
        .arch
        .attn_v_bias_key(0)
        .expect("starcoder2 declares v bias");
    let o_key = weights
        .arch
        .attn_o_bias_key(0)
        .expect("starcoder2 declares o bias");

    // Fixture without the vectors: all four stay None — the builder must
    // never fabricate a bias.
    if !weights.vectors.contains_key(&q_key) {
        let bare = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        assert!(bare.attn_q_bias.is_none());
    }
    weights.vectors.insert(q_key, vec![0.25f32; 8]);
    weights.vectors.insert(k_key, vec![0.5f32; 4]);
    weights.vectors.insert(v_key, vec![0.75f32; 4]);
    weights.vectors.insert(o_key, vec![1.25f32; 8]);

    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    assert_eq!(layer.attn_q_bias.unwrap()[0], 0.25);
    assert_eq!(layer.attn_k_bias.unwrap()[0], 0.5);
    assert_eq!(layer.attn_v_bias.unwrap()[0], 0.75);
    assert_eq!(layer.attn_o_bias.unwrap()[0], 1.25);
}
