//! Shared fixtures, engine rosters and comparison helpers for the
//! GPU-parity suite.

use larql_inference::{cpu_engine_backend, default_engine_backend};
use larql_kv::EngineKind;

/// Engines benched unwindowed. Apollo is excluded (its prefill fails
/// closed without an attached boundary store); `no-cache` holds no K/V
/// to place on either side of the comparison.
pub(crate) const UNWINDOWED: &[&str] = &[
    "standard",
    "markov-rs",
    "markov-rs-codec",
    "boundary-per-layer:layers=2",
];

/// Windowed specs whose window bounds **attention**. These engines have
/// no cold tier — an evicted K/V row is gone — so attending past the
/// window would answer from data the engine claims not to have. They
/// must decline the window-less coarse surface.
pub(crate) const WINDOW_BOUNDS_ATTENTION: &[&str] = &["standard:window=2"];

/// Windowed specs whose window bounds **hot-tier memory only**. These
/// retain a cold tier and attend over full history by contract (see
/// `fix(kv,compute): the window must bound attention, not just storage`,
/// which drew exactly this distinction while fixing the engine that had
/// it wrong). They may legitimately take the coarse path.
pub(crate) const WINDOW_BOUNDS_MEMORY: &[&str] = &[
    "markov-rs:window=2",
    "markov-rs-codec:window=2",
    "boundary-per-layer:window=2,layers=2",
];

/// Every spec the numeric comparison sweeps.
pub(crate) fn all_specs() -> Vec<&'static str> {
    UNWINDOWED
        .iter()
        .chain(WINDOW_BOUNDS_ATTENTION.iter())
        .chain(WINDOW_BOUNDS_MEMORY.iter())
        .copied()
        .collect()
}

/// Build an engine on either the default (GPU-capable) backend or the
/// explicit CPU one.
pub(crate) fn build(spec: &str, gpu: bool) -> larql_kv::AnyEngine {
    let backend = if gpu {
        default_engine_backend()
    } else {
        cpu_engine_backend()
    };
    EngineKind::from_name(spec)
        .unwrap_or_else(|| panic!("spec {spec:?} failed to parse"))
        .build(backend)
}

/// Whether the selected default backend implements the per-layer surface
/// by forwarding to the host — Metal today.
pub(crate) fn default_backend_is_host_delegating() -> bool {
    larql_compute::kv_dispatch::KvDispatch::per_layer_is_host_delegated(
        default_engine_backend().as_ref(),
    )
}

/// Prefill runs a single fused pass on each side; agreement is tight
/// (measured ~1e-7 on the SWA-free fixture).
pub(crate) const PREFILL_TOL: f32 = 1e-4;

/// Per-step decode. Metal's fused decode kernel and the CPU cached-decode
/// path dequantise and accumulate Q4K in different orders, which lands at
/// a stable 2.6e-3 - 3.6e-3 on this fixture for every engine. The bound
/// is set above that band; the compounding check is what actually guards
/// correctness.
pub(crate) const DECODE_TOL: f32 = 1e-2;

/// How much the last decode step's divergence may exceed the first
/// before we call it drift rather than rounding.
pub(crate) const COMPOUNDING_FACTOR: f32 = 3.0;

/// Assert two hidden states agree within `tol`. Returns the relative L2
/// so callers can check its trend across steps.
pub(crate) fn assert_close(
    a: &ndarray::Array2<f32>,
    b: &ndarray::Array2<f32>,
    tol: f32,
    spec: &str,
    stage: &str,
) -> f32 {
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{spec}: {stage} produced non-finite values on the gpu path"
    );
    let rel = relative_l2(a, b);
    assert!(
        rel < tol,
        "{spec}: {stage} diverged between backends — relative L2 {rel:.3e} (tol {tol:.0e})"
    );
    rel
}

/// Relative L2 of `a - b` against `b`'s norm.
pub(crate) fn relative_l2(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt().max(NORM_FLOOR);
    num / den
}

/// Guards against dividing by a zero-norm reference.
const NORM_FLOOR: f32 = 1e-6;
