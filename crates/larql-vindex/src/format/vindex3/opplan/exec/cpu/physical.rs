//! One decision — format, kernel, and threading together.
//!
//! The hazard this file exists to remove is a loader that chooses BF16
//! while an executor separately guesses which kernel to run. Those are
//! two derivations of one fact, and two derivations drift. With two
//! formats the drift is a bug; with Q8 and Q4 as well it is a state
//! space nobody can hold in their head.
//!
//! So there is exactly one value, [`PhysicalProjectionPlan`], and both
//! halves are read off it. The loader asks [`PhysicalProjectionPlan::choose`]
//! what to make resident; the executor asks
//! [`PhysicalProjectionPlan::for_resident`] what is resident. The second
//! is not a second decision — it is an OBSERVATION of the first, total
//! over the representations a CPU kernel can consume, so the two cannot
//! disagree about a matrix even in principle.
//!
//! [`project_matrix`] and [`ExecutorProjections`] live here rather than
//! beside the backend for the same reason: they are the only readers of
//! the observation, and a projection helper that sat somewhere else would
//! be one refactor away from choosing its own kernel again.

use super::kernels::{BlasF32, FusedBf16, FusedQ4, FusedQ8, ScalarF32};
use super::projector::{DenseProjector, WeightRows};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightSlice};

/// Default performance-cluster L2, used where the machine does not
/// report one. The value this rung measured against (Apple M3 Max).
const DEFAULT_L2_BYTES: usize = 16 * 1024 * 1024;

/// How a dense projection is physically realised on the CPU.
///
/// A single enum rather than a `(format, kernel)` pair because the
/// pairing is not free: `FusedBf16` consumes [`WeightRows::Bf16`] and
/// nothing else, and a pair type would let a caller build the one
/// combination that cannot run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhysicalProjectionPlan {
    /// The literal scalar transcription over f32. The oracle: chosen by
    /// the reference backend, never by the policy.
    ScalarF32,
    /// Q8 resident, widened and scaled in registers, executor-threaded.
    ///
    /// The first LOSSY plan: the values it decodes are not the values the
    /// checkpoint stores. Worth 1.28x on the projections a token runs —
    /// half the bytes returning a third of the time, because at 8.5 bits
    /// the kernel stops waiting for memory and starts waiting for the
    /// widen.
    FusedQ8,
    /// Q4 resident, unpacked and scaled in registers.
    ///
    /// Reachable by OBSERVATION and not by [`Self::choose`]: CPU-4A asks
    /// only whether Q4 x f32 is worth making a model representation, and
    /// no `WeightFormat` names it, so a policy answering Q4 would refuse
    /// at load. Listed so `for_resident` stays total.
    FusedQ4,
    /// f32 resident, BLAS `sgemv`, threaded by the library.
    ///
    /// The right answer for a matrix whose widened image still fits
    /// cache — see [`compact_threshold_bytes`].
    BlasF32,
    /// bf16 resident, widened in registers, threaded by the executor.
    ///
    /// Halves the bytes a decoded token streams AND halves what the
    /// model occupies, because they are the same bytes.
    FusedBf16,
}

impl PhysicalProjectionPlan {
    /// The representation the loader must make resident for this plan.
    pub fn format(self) -> WeightFormat {
        match self {
            Self::ScalarF32 | Self::BlasF32 => WeightFormat::F32,
            Self::FusedBf16 => WeightFormat::Bf16,
            Self::FusedQ8 => WeightFormat::Q8,
            // No `WeightFormat` names Q4; see the variant's note.
            Self::FusedQ4 => WeightFormat::Q8,
        }
    }

    /// The kernel that consumes it, which in turn declares its threading.
    pub fn kernel(self) -> &'static dyn DenseProjector {
        match self {
            Self::ScalarF32 => &ScalarF32,
            Self::BlasF32 => &BlasF32,
            Self::FusedBf16 => &FusedBf16,
            Self::FusedQ8 => &FusedQ8,
            Self::FusedQ4 => &FusedQ4,
        }
    }

    /// **The policy.** What to make one matrix resident as.
    ///
    /// `elements` is the matrix's element count — `out_dim * in_dim` —
    /// so the question is asked per MATRIX, not per matrix class. That
    /// distinction is the whole point: Qwen3.8's `48 x 5120` delta gates
    /// and its `10240 x 5120` fused projection are both attention-class
    /// operands, and they want opposite answers.
    ///
    /// `stored_bf16` is a physical fact about the checkpoint, not a
    /// preference. A container holding f32 has no compact bytes to keep,
    /// and narrowing them here would ROUND — bf16 residency promises the
    /// stored bytes are the resident bytes, and a policy that quietly
    /// quantised to hit its own threshold would make that a lie.
    pub fn choose(elements: usize, stored_bf16: bool) -> Self {
        if !stored_bf16 || elements * F32_BYTES < compact_threshold_bytes() {
            return Self::BlasF32;
        }
        // **The same cache argument, one format further down.**
        //
        // BF16 beats BLAS f32 once the F32 image stops fitting L2; Q8
        // beats BF16 once the BF16 image does. Measured on the real
        // shapes: `1024 x 5120` is 10.5 MB as bf16, still L2-resident,
        // and runs 0.81x through Q8 — no traffic to halve and the extra
        // unpacking is pure cost. `5120 x 6144` is 62.9 MB, streams, and
        // wins 1.16x. Every measured shape falls on the side this
        // predicts.
        if elements * BF16_BYTES >= compact_threshold_bytes() && q8_permitted() {
            Self::FusedQ8
        } else {
            Self::FusedBf16
        }
    }

    /// **The observation.** What IS resident, read off the bytes.
    ///
    /// Deliberately not a second call to [`Self::choose`]: an executor
    /// that re-derived the policy could be handed a matrix the loader
    /// decided differently about — a fallback, a checkpoint that stores
    /// something else, a threshold read on a machine reporting a
    /// different cache — and would then run the wrong kernel over the
    /// right bytes. Reading the representation cannot be wrong about it.
    pub fn for_resident(rows: WeightRows<'_>) -> Self {
        match rows {
            WeightRows::F32(_) => Self::BlasF32,
            WeightRows::Bf16(_) => Self::FusedBf16,
            WeightRows::Q8 { .. } => Self::FusedQ8,
            WeightRows::Q4 { .. } => Self::FusedQ4,
        }
    }
}

/// One projection through LARQL's own CPU executor.
///
/// **The kernel is OBSERVED, not chosen again.**
/// [`PhysicalProjectionPlan::for_resident`] reads back the decision
/// `weight_format` already made at load, off the bytes themselves — so a
/// matrix the loader kept compact cannot reach a kernel that expects f32,
/// and a fallback (an f32 checkpoint, an overlaid operand, a machine
/// reporting a different cache) needs no second rule to stay consistent.
///
/// NOT bit-identical to the scalar transcription: both kernels reassociate
/// the sum, measured at rel_rms ~1.3e-6 for BLAS and 3.6e-7 for the fused
/// bf16 kernel. No weight VALUE changes — bf16 widens exactly — so the
/// only difference either kernel introduces is summation order, and the
/// parity gates are what judge it.
pub fn project_rows(
    weight: WeightRows<'_>,
    x: &[f32],
    out_dim: usize,
) -> Result<Vec<f32>, VindexError> {
    let plan = PhysicalProjectionPlan::for_resident(weight);
    Ok(super::shared()?.project(plan.kernel(), weight, x, out_dim))
}

/// The same, from a declared operand slice plus the geometry that says
/// how much of it is the matrix.
pub fn project_matrix(
    weight: &WeightSlice<'_>,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, VindexError> {
    project_rows(weight.rows(out_dim, in_dim)?, x, out_dim)
}

/// Gated DeltaNet's five dense projections, through the same executor
/// and the same observation as every other production matrix.
pub struct ExecutorProjections;

impl crate::format::vindex3::opplan::exec::gated_delta::DenseProjections for ExecutorProjections {
    fn project(&self, weight: WeightRows<'_>, x: &[f32], out_dim: usize) -> Vec<f32> {
        project_rows(weight, x, out_dim)
            .expect("the CPU executor pool is unavailable, so no projection can run")
    }
}

/// Caps the policy at a representation, for A/B'ing FORMATS in one
/// binary.
pub const MAX_FORMAT_ENV: &str = "LARQL_CPU_MAX_FORMAT";

/// Whether the policy may reach Q8.
///
/// Exists so a lossy format can be compared against the exact one it
/// replaces WITHOUT rebuilding. Q8 changes logits, so the comparison has
/// to be against the same binary's own bf16 answer — a rebuild moved an
/// untouched function 14% in CPU-2D, and a numerical A/B across builds
/// would be arguing with a compiler as much as with a format.
///
/// Only `bf16` caps anything; every other value (and no value) leaves the
/// measured policy in force, so a typo cannot silently disable Q8 in
/// production and be mistaken for a regression.
fn q8_permitted() -> bool {
    !matches!(
        std::env::var(MAX_FORMAT_ENV).ok().as_deref().map(str::trim),
        Some("bf16")
    )
}

/// f32 bytes per element — what the BLAS alternative must read.
pub(super) const F32_BYTES: usize = 4;

/// bf16 bytes per element — what the Q8 alternative must read.
pub(super) const BF16_BYTES: usize = 2;

/// The f32 footprint at or above which compact-to-registers wins.
///
/// **Not a fitted constant: it is the performance cluster's L2 size.**
/// BLAS `sgemv` reads its weights from cache while the widened matrix
/// fits (measured 291 GB/s at 7.9 MB) and from RAM once it does not (117
/// GB/s at 21 MB), so the crossover between the two kernels IS the cache
/// boundary — the fused kernel has no such cliff because it streams
/// either way.
///
/// Swept at Qwen3.8's `in_dim` of 5120, the two cross at 832 rows =
/// 17.04 MB against this machine's 16 MiB L2: fused loses 0.60x at 768
/// rows below the boundary and wins 1.82x at 896 rows above it. The
/// transition is a cliff, not a slope, which is why the constant is read
/// from the hardware rather than tuned.
///
/// Every real Qwen3.8 matrix sits far from it — the nearest below is
/// `48 x 5120` at 0.98 MB (BLAS by 3.8x) and the nearest above is
/// `1024 x 5120` at 20.97 MB (fused by 1.99x), a factor of 21 apart —
/// so this model would decode identically under any threshold inside
/// that bracket. A future model with a matrix in the gap is what the
/// boundary is for.
pub(super) fn compact_threshold_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = super::executor::sysctl_usize("hw.perflevel0.l2cachesize") {
            return bytes.max(1);
        }
    }
    DEFAULT_L2_BYTES
}
