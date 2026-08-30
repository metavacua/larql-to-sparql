//! `mock_gpu_backend_tests` for [`super`].
//!
//! Split out of `test_utils.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::*;
use larql_compute::backend::Capability;
use larql_compute::prelude::*;

#[test]
fn mock_advertises_decode_token_capability() {
    let mock = MockGpuBackend::new();
    assert!(mock.supports(Capability::DecodeToken));
    assert!(mock.supports(Capability::PrefillQ4));
    assert!(mock.supports(Capability::DecodeQ4KMoe));
    assert_eq!(mock.name(), "mock-gpu");
}

#[test]
fn mock_decode_token_returns_hidden_sized_vector() {
    let mock = MockGpuBackend::new();
    let out = mock.decode_token(&[], &[], 8, 16).expect("Some");
    assert_eq!(out.len(), 8);
    assert_eq!(mock.kv_cache_len(), 1);
}

#[test]
fn mock_prefill_q4_returns_seq_x_hidden_vector() {
    let mock = MockGpuBackend::new();
    let out = mock
        .prefill_kquant(&[], &[], 4, 16, 3, false, 0.0)
        .expect("Some");
    assert_eq!(out.len(), 3 * 4);
    assert_eq!(mock.kv_cache_len(), 3);
}

#[test]
fn mock_reset_clears_kv_len() {
    let mock = MockGpuBackend::new();
    let _ = mock.prefill_kquant(&[], &[], 4, 16, 5, false, 0.0);
    assert_eq!(mock.kv_cache_len(), 5);
    mock.reset_kv_cache();
    assert_eq!(mock.kv_cache_len(), 0);
}

#[test]
fn mock_truncate_sets_kv_len() {
    let mock = MockGpuBackend::new();
    let _ = mock.prefill_kquant(&[], &[], 4, 16, 10, false, 0.0);
    mock.truncate_kv_cache(3);
    assert_eq!(mock.kv_cache_len(), 3);
}

#[test]
fn mock_decode_with_moe_invokes_callback() {
    let mock = MockGpuBackend::new();
    let mut callback_fired = false;
    let mut moe_fn = |_layer: usize, _h: &[f32]| -> Vec<f32> {
        callback_fired = true;
        vec![0.0f32; 8]
    };
    let _ = mock.decode_token_with_moe(&[], &[], 8, 16, &mut moe_fn);
    assert!(callback_fired);
}

#[test]
fn mock_decode_q4k_moe_invokes_expert_lookup() {
    let mock = MockGpuBackend::new();
    let lookup_count = std::cell::Cell::new(0);
    let bytes = [0u8; 16];
    let get_expert = |_layer: usize, _expert: usize| -> Option<(&[u8], &[u8])> {
        lookup_count.set(lookup_count.get() + 1);
        Some((&bytes[..], &bytes[..]))
    };
    let _ = mock.decode_token_q4k_moe(&[], &[], 8, 16, 1e-6, &get_expert);
    assert!(lookup_count.get() >= 1);
}
