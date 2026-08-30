//! KV-B1 sequence-parallel dispatch policy.
//!
//! Owns one decision: given what the operator asked for (`LARQL_KV_SEQPAR`),
//! the layer's `head_dim`, and the effective attention span, how many
//! sequence slices should phase 3 dispatch — or should it refuse and leave
//! the shipped serial kernel in place.
//!
//! The kernels themselves live in `crate::shaders::kv_attention` and
//! `crate::shaders::kv_append_attend_fused`; dispatch lives in
//! `crate::ops::kv_cache`. This module is policy only, so it stays testable
//! without a Metal device.
//!
//! # Programme invariant (KV-B1 and KV-B2)
//!
//! **The sequence-parallel arms are an execution-order change and nothing
//! else.** KV stays f32, slice partials accumulate in f32, and the merge
//! runs in f32. The only permitted difference from the serial kernel is
//! reassociation of the weighted-V sum.
//!
//! This is what licenses the parity gates in
//! `tests/test_kernel_kv_attention_seqpar.rs` and its two siblings, which
//! assert `max_rel < 1e-4` against the serial f32 kernel with negative
//! controls calibrated at ~1e-1. That tolerance separates *reassociation*
//! from *defect*; it cannot separate reassociation from *approximation*.
//! An f16 KV cache or an f16 accumulator blows past 1e-4 by construction,
//! so narrowing any width here would not fail these gates loudly — it would
//! quietly convert them from a proof of exactness into a loose bound, and
//! fold a representation win into the B1/B2 latency number.
//!
//! Representation width is KV-C's subject, and KV-C needs its own oracle:
//! an f32-KV reference scored in predictive units on the deployment path,
//! with a quality budget fixed before any latency is measured. Do not
//! borrow these gates for it.

/// Threadgroup width ceiling for the KV-B1 seqpar kernels. Matches
/// `tg_partial[1024]` in `shaders::kv_attention` and
/// `shaders::kv_append_attend_fused`, and Metal's own 1024-thread
/// threadgroup limit — so this is the hard ceiling, not a tuning knob.
pub const SEQPAR_MAX_THREADS: usize = 1024;

/// What the operator asked of `LARQL_KV_SEQPAR`.
///
/// [`SeqparRequest::Unset`] is deliberately distinct from
/// [`SeqparRequest::Off`]: "the operator said nothing" resolves per
/// geometry (see [`default_is_auto`]), while "the operator said `off`" is
/// an override that must hold on every geometry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SeqparRequest {
    /// `LARQL_KV_SEQPAR` absent, or set to something unparseable — the
    /// latter deliberately does not inherit the geometry default, since
    /// guessing at a typo is worse than keeping the shipped kernel.
    Unset,
    /// `off`, or `0`.
    Off,
    /// `auto` — the measured span policy, on any geometry the caller asks
    /// for, subject to the usual refusal and clamping.
    Auto,
    /// An explicit slice count.
    Slices(usize),
}

/// head_dims where `auto` is the *default*, i.e. where the span policy in
/// [`auto_threads`] has been measured rather than merely extrapolated.
///
/// **Widening this list is the entire mechanism for broadening the
/// default, and it is gated on evidence, not on the code supporting the
/// geometry.** Adding an entry requires a `examples/bench_attention_span`
/// sweep at that head_dim on an idle GPU showing the same tier ordering.
/// Until then a new geometry can still opt in explicitly with `auto`.
///
/// 64 is gpt-oss-20b, the only geometry the KV-B1 sweep covered, enabled
/// on the A/B/C gate in `docs/kv-attention-scaling.md`.
const SEQPAR_DEFAULT_ON_HEAD_DIMS: &[usize] = &[64];

/// Whether an unset `LARQL_KV_SEQPAR` means `auto` for this geometry.
///
/// Narrow by construction: the alternative — defaulting on wherever the
/// kernel *can* run — would ship an unmeasured policy, and at head_dim 512
/// would present a silent no-op (see [`slices_for`]) as a defaulted
/// optimisation.
pub fn default_is_auto(head_dim: usize) -> bool {
    SEQPAR_DEFAULT_ON_HEAD_DIMS.contains(&head_dim)
}

/// Threadgroup widths the KV-B1 sweep found best, by span tier.
///
/// Expressed in **threads, not slices**, on the hypothesis that the
/// optimum tracks threadgroup occupancy rather than the slice count
/// itself, so converting via `head_dim` keeps the policy meaningful for
/// other attention geometries instead of pinning it to gpt-oss's 64.
///
/// **That hypothesis is unfalsified, not measured.** Every row below was
/// taken at head_dim 64, where width and slice count are collinear —
/// 8/12/16 slices *are* 512/768/1024 threads — so the sweep cannot
/// distinguish an occupancy optimum from a slice-count optimum. The two
/// read differently everywhere else: at head_dim 256 the short tier means
/// 2 slices, and at the 512 of Gemma 3's global layers it means 1, i.e.
/// [`slices_for`] refuses and `auto` silently does nothing below span
/// 1024. A second sweep at head_dim 128 separates them (4 slices under
/// occupancy, 8 under slice-count) and is cheap. This is why
/// [`SEQPAR_DEFAULT_ON_HEAD_DIMS`] holds only 64.
///
/// Measured with `examples/bench_attention_span` on M3 Max, gpt-oss-20b
/// geometry, 2026-08-14. Only rows whose bracketing baselines agreed
/// within 5% are cited here:
///
/// ```text
/// span   8x    12x   16x    best
///  192  2.41  2.44  2.07    8~12
///  256  2.92  2.76  2.32    8
///  384  3.17  3.19  2.85    8~12
///  768  4.32  5.25  5.15    12
/// 1024  4.68  5.30  5.41    16
/// 2048  5.68  6.73  6.90    16
/// ```
///
/// Those establish that **the widest threadgroup is genuinely harmful at
/// short span** — at span 256 it costs 21% against 8 slices — so this is a
/// real optimum rather than a clamp against the 1024-thread hardware
/// limit, and that 16 still wins at 2048, i.e. intra-threadgroup
/// parallelism is exhausted there rather than merely sufficient.
const SEQPAR_THREADS_SHORT: usize = 512;
const SEQPAR_THREADS_MID: usize = 768;
const SEQPAR_THREADS_LONG: usize = 1024;

/// Span tier boundaries — **engineering policy, not measured phase
/// transitions**, and worth keeping that distinction.
///
/// The sweep locates the ordering at 192-384 (8 wins), 768 (12) and
/// 1024-2048 (16); it does NOT locate where each crossover actually falls.
/// The rows nearest these boundaries are exactly the ones that breached
/// the drift rule — span 384 and 512 read -10.1% and -12.8% on their
/// bracketing baselines, and negative drift inflates the arms measured
/// between them — so 512 and 1024 are round numbers interpolated between
/// trustworthy neighbours.
///
/// Do not treat them as discovered constants or tune against them. A
/// re-sweep on an idle machine, or any other head_dim, may move them; what
/// should survive is the shape (narrow at short span, widest at long).
pub const SEQPAR_SPAN_TIER_MID: u32 = 512;
pub const SEQPAR_SPAN_TIER_LONG: u32 = 1024;

/// Measured-best threadgroup width for `span`. See [`SEQPAR_THREADS_SHORT`].
fn auto_threads(span: u32) -> usize {
    if span < SEQPAR_SPAN_TIER_MID {
        SEQPAR_THREADS_SHORT
    } else if span < SEQPAR_SPAN_TIER_LONG {
        SEQPAR_THREADS_MID
    } else {
        SEQPAR_THREADS_LONG
    }
}

/// Sequence slices the KV-B1 kernel may use for this request and geometry.
///
/// Bounded by the kernel's `tg_partial`, which holds `n_slices * head_dim`
/// partials, and by the caller contract that `tg_sz` be a multiple of
/// `head_dim`. Returns 0 when the request cannot be honoured, which the
/// caller reads as "use the shipped kernel" — a refusal rather than a
/// silently different geometry.
pub fn slices_for(request: SeqparRequest, head_dim: usize, span: u32) -> usize {
    if head_dim == 0 {
        return 0;
    }
    let requested = match request {
        SeqparRequest::Off => return 0,
        SeqparRequest::Unset if !default_is_auto(head_dim) => return 0,
        SeqparRequest::Unset | SeqparRequest::Auto => auto_threads(span) / head_dim,
        SeqparRequest::Slices(n) => n,
    };
    if requested <= 1 {
        return 0;
    }
    // `tg_partial[SEQPAR_MAX_THREADS]` and `tg_sg_vals[32]` in the seqpar
    // kernels both bound the threadgroup at SEQPAR_MAX_THREADS; the
    // partial bound is the tighter one for head_dim >= 32. The ceiling was
    // raised from 512 after the KV-B1 sweep found the optimum sitting
    // exactly ON the old ceiling at every span, i.e. unmeasured above it.
    let max_by_partial = SEQPAR_MAX_THREADS / head_dim;
    let n = requested.min(max_by_partial);
    if n <= 1 {
        0
    } else {
        n
    }
}

#[cfg(test)]
mod tests;
