//! Regression gate: native `gate_knn` vs Wasmi-boundary `gate_knn`.
//!
//! Both paths use the same inputs and must produce bit-for-bit identical
//! results. This test requires the compiled wasm binary at:
//!   target/wasm32v1-none/debug/larql-wasm32v1-none.wasm
//!
//! Build it first with:
//!   cargo build --target wasm32v1-none -p larql-wasm32v1-none-bin

use larql_wasmi_host::{Dtype, KnnResult, LarqlCoreRuntime, LayerData};
use larql_wasm32v1_none_lib::gate::{decode::StorageDtype, index::GateIndex, knn::gate_knn};

/// Load the compiled wasm binary. Panics with instructions if not built yet.
fn load_module(runtime: &LarqlCoreRuntime) -> wasmi::Module {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/wasm32v1-none/debug/larql-wasm32v1-none.wasm");
    let bytes = std::fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "wasm binary not found at {path:?}\n\
             Run: cargo build --target wasm32v1-none -p larql-wasm32v1-none-bin"
        )
    });
    runtime.compile(&bytes).expect("compile wasm module")
}

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn two_feature_gate_f32() -> Vec<u8> {
    [1.0f32, 0.0, 0.0, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

fn native_gate_knn_results(
    gate_bytes: &[u8],
    hidden_size: usize,
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut idx = GateIndex::new(1, hidden_size);
    idx.load_layer(0, gate_bytes, StorageDtype::F32);
    gate_knn(&idx, 0, query, k)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn dot_native_vs_wasm() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);
    let mut session = runtime.session(&module).unwrap();

    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];

    let native: f32 = larql_wasm32v1_none_lib::linalg::dot(&a, &b);
    let via_wasm = session.dot(&a, &b).expect("dot via wasm");

    assert!(
        (native - via_wasm).abs() < 1e-6,
        "native={native}, wasm={via_wasm}"
    );
}

#[test]
fn norm_native_vs_wasm() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);
    let mut session = runtime.session(&module).unwrap();

    let a = [3.0f32, 4.0];
    let native = larql_wasm32v1_none_lib::linalg::norm(&a);
    let via_wasm = session.norm(&a).expect("norm via wasm");

    assert!((native - via_wasm).abs() < 1e-5, "native={native}, wasm={via_wasm}");
}

#[test]
fn cosine_native_vs_wasm() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);
    let mut session = runtime.session(&module).unwrap();

    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];

    let native = larql_wasm32v1_none_lib::linalg::cosine(&a, &b);
    let via_wasm = session.cosine(&a, &b).expect("cosine via wasm");

    assert!((native - via_wasm).abs() < 1e-6, "native={native}, wasm={via_wasm}");
}

#[test]
fn gate_knn_empty_index_native_vs_wasm() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);
    let mut session = runtime.session(&module).unwrap();

    let query = [1.0f32, 0.0];
    let native = native_gate_knn_results(&[], 2, &query, 5);
    let via_wasm = session
        .gate_knn(2, &[None], 0, &query, 5)
        .expect("gate_knn empty via wasm");

    assert!(native.is_empty());
    assert!(via_wasm.is_empty());
}

#[test]
fn gate_knn_two_features_top1_native_vs_wasm() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);
    let mut session = runtime.session(&module).unwrap();

    let gate_bytes = two_feature_gate_f32();
    let query = [1.0f32, 0.0];
    let k = 1;

    // Native
    let native = native_gate_knn_results(&gate_bytes, 2, &query, k);

    // Wasm
    let via_wasm = session
        .gate_knn(
            2,
            &[Some(LayerData { bytes: &gate_bytes, num_features: 2, dtype: Dtype::F32 })],
            0,
            &query,
            k as u32,
        )
        .expect("gate_knn via wasm");

    assert_eq!(native.len(), via_wasm.len(), "result count mismatch");
    for ((n_feat, n_score), KnnResult { feature: w_feat, score: w_score }) in
        native.iter().zip(via_wasm.iter())
    {
        assert_eq!(*n_feat as u32, *w_feat, "feature index mismatch");
        assert!(
            (n_score - w_score).abs() < 1e-6,
            "score mismatch: native={n_score}, wasm={w_score}"
        );
    }
}

#[test]
fn gate_knn_top_k_ordering_preserved() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);
    let mut session = runtime.session(&module).unwrap();

    // 4 features: rows [1,0], [0,1], [0.7,0.7], [-1,0]
    let gate_bytes: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0, 0.7, 0.7, -1.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let query = [1.0f32, 0.0];
    let k = 3;

    let native = native_gate_knn_results(&gate_bytes, 2, &query, k);
    let via_wasm = session
        .gate_knn(
            2,
            &[Some(LayerData { bytes: &gate_bytes, num_features: 4, dtype: Dtype::F32 })],
            0,
            &query,
            k as u32,
        )
        .expect("gate_knn via wasm");

    assert_eq!(native.len(), via_wasm.len());
    for ((n_feat, n_score), KnnResult { feature: w_feat, score: w_score }) in
        native.iter().zip(via_wasm.iter())
    {
        assert_eq!(*n_feat as u32, *w_feat);
        assert!((n_score - w_score).abs() < 1e-5);
    }
}

#[test]
fn multiple_sessions_are_isolated() {
    let runtime = LarqlCoreRuntime::new().unwrap();
    let module = load_module(&runtime);

    // Two independent sessions should not share static solution buffer state.
    let mut s1 = runtime.session(&module).unwrap();
    let mut s2 = runtime.session(&module).unwrap();

    let r1 = s1.dot(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
    let r2 = s2.dot(&[0.0, 1.0], &[0.0, 1.0]).unwrap();

    assert!((r1 - 1.0).abs() < 1e-6);
    assert!((r2 - 1.0).abs() < 1e-6);
}
