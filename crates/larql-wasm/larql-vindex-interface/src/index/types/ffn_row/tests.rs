//! Coverage for the unified `ffn_row_*` dispatch chain. Each test
//! pins one branch of the priority cascade (FP4 → native f32 → Q4_K)
//! against a stub that lights up exactly one backend so we can see
//! which path the dispatch took.

use super::super::StorageBucket;
use super::super::{Fp4FfnAccess, NativeFfnAccess, QuantizedFfnAccess};
use super::test_support::{one_row, Stub};
use super::FfnRowAccess;

// ── ffn_row_dot ─────────────────────────────────────────────

#[test]
fn dot_routes_through_fp4_first() {
    let s = Stub {
        fp4_dot: Some(99.0),
        gate: Some(one_row(&[1.0, 1.0, 1.0])),
        q4k_dot: Some(7.0),
        ..Default::default()
    };
    let x = [1.0, 1.0, 1.0];
    assert_eq!(s.ffn_row_dot(0, 0, 0, &x), Some(99.0));
}

#[test]
fn dot_falls_through_to_native_gate_when_fp4_declines() {
    let s = Stub {
        fp4_into_returns: true,
        fp4_dot: None,
        gate: Some(one_row(&[2.0, 3.0, 4.0])),
        q4k_dot: Some(7.0),
        ..Default::default()
    };
    let x = [1.0, 1.0, 1.0];
    assert_eq!(s.ffn_row_dot(0, 0, 0, &x), Some(9.0));
}

#[test]
fn dot_native_up_then_q4k_fallback() {
    let s = Stub {
        up: Some(one_row(&[5.0])),
        q4k_dot: Some(99.0),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 1, 0, &[2.0]), Some(10.0));
}

#[test]
fn dot_native_down_via_interleaved() {
    let s = Stub {
        down: Some(one_row(&[3.0, 3.0])),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 1.0]), Some(6.0));
}

#[test]
fn dot_q4k_used_when_no_native() {
    let s = Stub {
        q4k_dot: Some(42.0),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 0, 0, &[1.0]), Some(42.0));
}

#[test]
fn dot_returns_none_when_nothing_loaded() {
    let s = Stub::default();
    assert!(s.ffn_row_dot(0, 0, 0, &[1.0]).is_none());
}

#[test]
fn dot_invalid_component_returns_none() {
    let s = Stub {
        gate: Some(one_row(&[1.0])),
        ..Default::default()
    };
    assert!(s.ffn_row_dot(0, 99, 0, &[1.0]).is_none());
}

#[test]
fn dot_native_shape_mismatch_falls_through_to_q4k() {
    let s = Stub {
        gate: Some(one_row(&[1.0, 1.0, 1.0])),
        q4k_dot: Some(123.0),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 0, 0, &[1.0, 1.0]), Some(123.0));
}

#[test]
fn dot_component_1_falls_through_to_up_layer_matrix() {
    let s = Stub {
        up_layer: Some(one_row(&[3.0, 3.0])),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 1, 0, &[1.0, 2.0]), Some(9.0));
}

#[test]
fn dot_component_2_uses_down_feature_vector_first() {
    let s = Stub {
        down_feature: Some(vec![2.0, 3.0]),
        down: Some(one_row(&[99.0, 99.0])),
        down_layer: Some(one_row(&[88.0, 88.0])),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 1.0]), Some(5.0));
}

#[test]
fn dot_component_2_falls_through_to_down_layer_matrix() {
    let s = Stub {
        down_layer: Some(one_row(&[4.0, 4.0])),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 0.5]), Some(6.0));
}

#[test]
fn dot_component_2_skips_down_feature_on_shape_mismatch() {
    let s = Stub {
        down_feature: Some(vec![1.0, 2.0, 3.0]),
        down: Some(one_row(&[5.0, 5.0])),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 2.0]), Some(15.0));
}

#[test]
fn dot_component_1_native_shape_mismatch_falls_through_to_q4k() {
    let s = Stub {
        up: Some(one_row(&[1.0, 1.0, 1.0])),
        q4k_dot: Some(77.0),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 1, 0, &[1.0, 1.0]), Some(77.0));
}

#[test]
fn dot_component_2_all_native_shape_mismatch_falls_through_to_q4k() {
    let s = Stub {
        down_feature: Some(vec![1.0, 2.0, 3.0]),
        down: Some(one_row(&[1.0, 1.0, 1.0])),
        down_layer: Some(one_row(&[1.0, 1.0, 1.0])),
        q4k_dot: Some(55.0),
        ..Default::default()
    };
    assert_eq!(s.ffn_row_dot(0, 2, 0, &[1.0, 2.0]), Some(55.0));
}

// ── ffn_row_scaled_add ──────────────────────────────────────

#[test]
fn scaled_add_native_gate_writes_alpha_times_row() {
    let s = Stub {
        gate: Some(one_row(&[2.0, 4.0])),
        ..Default::default()
    };
    let mut out = [10.0_f32, 10.0];
    assert!(s.ffn_row_scaled_add(0, 0, 0, 0.5, &mut out));
    assert_eq!(out, [11.0, 12.0]);
}

#[test]
fn scaled_add_down_prefers_feature_major_q4k() {
    let s = Stub {
        q4k_dot: Some(0.0),
        q4k_down_feature_returns: true,
        q4k_scaled_add_returns: false,
        ..Default::default()
    };
    let mut out = [0.0_f32; 4];
    assert!(s.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
}

#[test]
fn scaled_add_falls_back_to_cache_when_feature_major_declines() {
    let s = Stub {
        q4k_into_returns: true,
        q4k_down_feature_returns: false,
        q4k_scaled_add_returns: true,
        ..Default::default()
    };
    let mut out = [0.0_f32; 4];
    assert!(s.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
}

#[test]
fn scaled_add_returns_false_when_no_backend_covers() {
    let s = Stub::default();
    let mut out = [0.0_f32; 4];
    assert!(!s.ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
    assert_eq!(out, [0.0_f32; 4], "must not modify out on failure");
}

#[test]
fn scaled_add_invalid_component_returns_false_early() {
    let s = Stub {
        q4k_scaled_add_returns: true,
        ..Default::default()
    };
    let mut out = [0.0_f32; 4];
    assert!(!s.ffn_row_scaled_add(0, 99, 0, 1.0, &mut out));
}

#[test]
fn scaled_add_component_1_via_up_layer_matrix() {
    let s = Stub {
        up_layer: Some(one_row(&[1.0, 2.0])),
        ..Default::default()
    };
    let mut out = [10.0_f32, 10.0];
    assert!(s.ffn_row_scaled_add(0, 1, 0, 2.0, &mut out));
    assert_eq!(out, [12.0, 14.0]);
}

#[test]
fn scaled_add_component_2_via_down_feature_vector() {
    let s = Stub {
        down_feature: Some(vec![3.0, 4.0]),
        ..Default::default()
    };
    let mut out = [0.0_f32; 2];
    assert!(s.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
    assert_eq!(out, [3.0, 4.0]);
}

#[test]
fn scaled_add_component_2_via_down_layer_matrix() {
    let s = Stub {
        down_layer: Some(one_row(&[1.0, 1.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 2];
    assert!(s.ffn_row_scaled_add(0, 2, 0, 0.5, &mut out));
    assert_eq!(out, [0.5, 0.5]);
}

#[test]
fn scaled_add_component_0_falls_through_to_q4k() {
    let s = Stub {
        q4k_scaled_add_returns: true,
        q4k_dot: Some(0.0),
        ..Default::default()
    };
    let mut out = [0.0_f32; 4];
    assert!(s.ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
}

#[test]
fn scaled_add_component_1_falls_through_to_q4k() {
    let s = Stub {
        q4k_scaled_add_returns: true,
        q4k_dot: Some(0.0),
        ..Default::default()
    };
    let mut out = [0.0_f32; 4];
    assert!(s.ffn_row_scaled_add(0, 1, 0, 1.0, &mut out));
}

// ── ffn_row_into ────────────────────────────────────────────

#[test]
fn into_native_up_copies_row_verbatim() {
    let s = Stub {
        up: Some(one_row(&[1.0, 2.0, 3.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 3];
    assert!(s.ffn_row_into(0, 1, 0, &mut out));
    assert_eq!(out, [1.0, 2.0, 3.0]);
}

#[test]
fn into_falls_through_to_q4k_when_native_declines() {
    let s = Stub {
        q4k_into_returns: true,
        ..Default::default()
    };
    let mut out = [0.0_f32; 3];
    assert!(s.ffn_row_into(0, 0, 0, &mut out));
}

#[test]
fn into_returns_false_with_no_backend() {
    let s = Stub::default();
    let mut out = [0.0_f32; 3];
    assert!(!s.ffn_row_into(0, 0, 0, &mut out));
}

#[test]
fn into_shape_mismatch_returns_false() {
    let s = Stub {
        gate: Some(one_row(&[1.0, 2.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 3];
    assert!(!s.ffn_row_into(0, 0, 0, &mut out));
}

#[test]
fn into_component_1_via_up_layer_matrix() {
    let s = Stub {
        up_layer: Some(one_row(&[7.0, 8.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 2];
    assert!(s.ffn_row_into(0, 1, 0, &mut out));
    assert_eq!(out, [7.0, 8.0]);
}

#[test]
fn into_component_2_via_down_feature_vector() {
    let s = Stub {
        down_feature: Some(vec![1.5, 2.5, 3.5]),
        ..Default::default()
    };
    let mut out = [0.0_f32; 3];
    assert!(s.ffn_row_into(0, 2, 0, &mut out));
    assert_eq!(out, [1.5, 2.5, 3.5]);
}

#[test]
fn into_component_2_via_interleaved_down() {
    let s = Stub {
        down: Some(one_row(&[9.0, 9.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 2];
    assert!(s.ffn_row_into(0, 2, 0, &mut out));
    assert_eq!(out, [9.0, 9.0]);
}

#[test]
fn into_component_2_via_down_layer_matrix() {
    let s = Stub {
        down_layer: Some(one_row(&[2.0, 4.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 2];
    assert!(s.ffn_row_into(0, 2, 0, &mut out));
    assert_eq!(out, [2.0, 4.0]);
}

#[test]
fn into_invalid_component_returns_false() {
    let s = Stub {
        gate: Some(one_row(&[1.0])),
        ..Default::default()
    };
    let mut out = [0.0_f32; 1];
    assert!(!s.ffn_row_into(0, 99, 0, &mut out));
}

#[test]
fn into_feat_out_of_range_falls_through_to_q4k() {
    let s = Stub {
        up: Some(one_row(&[1.0])),
        q4k_into_returns: true,
        ..Default::default()
    };
    let mut out = [0.0_f32; 1];
    assert!(s.ffn_row_into(0, 1, 99, &mut out));
}

// ── primary_storage_bucket ──────────────────────────────────

#[test]
fn bucket_fp4_when_fp4_storage_present() {
    let s = Stub {
        fp4_into_returns: true,
        ..Default::default()
    };
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Fp4);
}

#[test]
fn bucket_exact_when_native_present() {
    let s = Stub {
        gate: Some(one_row(&[1.0])),
        ..Default::default()
    };
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
}

#[test]
fn bucket_quantized_when_only_q4k() {
    let s = Stub {
        q4k_into_returns: true,
        ..Default::default()
    };
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Quantized);
}

#[test]
fn bucket_exact_when_nothing_loaded() {
    let s = Stub::default();
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
}

#[test]
fn bucket_priority_fp4_over_exact() {
    let s = Stub {
        fp4_into_returns: true,
        gate: Some(one_row(&[1.0])),
        ..Default::default()
    };
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Fp4);
}

#[test]
fn bucket_exact_when_full_mmap_ffn_only() {
    let s = Stub {
        up_layer: Some(one_row(&[1.0])),
        ..Default::default()
    };
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
}

#[test]
fn bucket_exact_when_down_features_only() {
    let s = Stub {
        down_feature: Some(vec![1.0]),
        ..Default::default()
    };
    assert_eq!(s.primary_storage_bucket(), StorageBucket::Exact);
}

#[test]
fn bucket_quantized_when_only_q4_legacy() {
    // QuantizedFfnAccess::has_interleaved_q4 default is false; we
    // can't trigger it via the Stub since we only override q4k.
    // Use a wrapper type that overrides only `has_interleaved_q4`.
    struct OnlyQ4;
    impl NativeFfnAccess for OnlyQ4 {}
    impl QuantizedFfnAccess for OnlyQ4 {
        fn has_interleaved_q4(&self) -> bool {
            true
        }
    }
    impl Fp4FfnAccess for OnlyQ4 {}
    assert_eq!(OnlyQ4.primary_storage_bucket(), StorageBucket::Quantized);
}
