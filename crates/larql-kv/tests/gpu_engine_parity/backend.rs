//! What the backend reports about itself, plus the dense (no-vindex)
//! forward through a non-CPU backend.

use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_inference::{cpu_engine_backend, default_engine_backend};

use super::support::{build, default_backend_is_host_delegating};

/// `per_layer_is_host_delegated` is what stops a bench row labelled
/// `[metal (GPU)]` from silently describing a CPU measurement. Pin that
/// Metal keeps answering `true` for as long as its per-layer methods
/// delegate to `CpuBackend` — when native per-layer kernels land, this
/// test is the one that should fail and be updated deliberately.
#[test]
fn host_delegation_flag_matches_the_backend_family() {
    let backend = default_engine_backend();
    let name = backend.name().to_string();
    let delegated =
        larql_compute::kv_dispatch::KvDispatch::per_layer_is_host_delegated(backend.as_ref());

    if name.contains("metal") {
        assert!(
            delegated,
            "MetalBackend's per-layer surface still forwards to CpuBackend; \
             reporting otherwise makes every windowed bench row a lie"
        );
    } else {
        assert!(
            !delegated,
            "backend {name:?} claims host delegation but is already the host"
        );
    }

    // CPU is never "delegating" — it IS the host.
    assert!(
        !larql_compute::kv_dispatch::KvDispatch::per_layer_is_host_delegated(
            cpu_engine_backend().as_ref()
        ),
        "CpuBackend must not report host delegation"
    );
}

/// Guards the suite itself: if the default backend is host-delegating
/// (Metal present and the `gpu` feature on) then the numeric test really
/// did compare two different execution paths. Without this the whole
/// suite could pass on a CPU-only build while appearing to cover the GPU.
#[test]
fn reports_whether_this_run_actually_exercised_a_gpu_backend() {
    let name = default_engine_backend().name().to_string();
    if default_backend_is_host_delegating() {
        assert!(
            name.contains("metal"),
            "unexpected host-delegating backend {name:?}"
        );
    } else {
        eprintln!(
            "[gpu_engine_parity] default backend is {name:?} — this run covered \
             the CPU path only. Build with `--features gpu` on macOS for GPU coverage."
        );
    }
}

/// The no-vindex path, through the GPU-capable backend. Cheap, but it is
/// the only coverage that the dense forward survives a non-CPU backend
/// being threaded through it.
#[test]
fn dense_prefill_and_decode_run_on_the_default_backend() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    for spec in ["standard", "standard:window=2", "markov-rs", "no-cache"] {
        let mut engine = build(spec, true);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .unwrap_or_else(|e| panic!("{spec}: dense prefill failed: {e}"));
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let d = engine
            .decode_step(&weights, &ffn, 3)
            .unwrap_or_else(|e| panic!("{spec}: dense decode failed: {e}"));
        assert!(
            d.iter().all(|v| v.is_finite()),
            "{spec}: dense decode produced non-finite values"
        );
    }
}
