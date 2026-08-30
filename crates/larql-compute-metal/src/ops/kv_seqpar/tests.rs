//! Policy tests for KV-B1 slice selection. No Metal device required.

use super::*;

/// gpt-oss-20b's head_dim, the geometry the sweep was run on.
const HD: usize = 64;

/// Gemma 3's two attention geometries — sliding and global layers.
const HD_GEMMA_SLIDING: usize = 256;
const HD_GEMMA_GLOBAL: usize = 512;

/// The common Llama/Qwen/Mistral geometry, and the head_dim that
/// separates the occupancy hypothesis from the slice-count one.
const HD_LLAMA: usize = 128;

#[test]
fn auto_follows_the_measured_span_tiers() {
    // Below MID the sweep's clean rows put 8 slices ahead of 12 and 16.
    assert_eq!(slices_for(SeqparRequest::Auto, HD, 64), 8);
    assert_eq!(slices_for(SeqparRequest::Auto, HD, 256), 8);
    assert_eq!(
        slices_for(SeqparRequest::Auto, HD, SEQPAR_SPAN_TIER_MID - 1),
        8
    );
    // MID..LONG: 12.
    assert_eq!(
        slices_for(SeqparRequest::Auto, HD, SEQPAR_SPAN_TIER_MID),
        12
    );
    assert_eq!(slices_for(SeqparRequest::Auto, HD, 768), 12);
    assert_eq!(
        slices_for(SeqparRequest::Auto, HD, SEQPAR_SPAN_TIER_LONG - 1),
        12
    );
    // At and past LONG: the ceiling, where the sweep was still climbing.
    assert_eq!(
        slices_for(SeqparRequest::Auto, HD, SEQPAR_SPAN_TIER_LONG),
        16
    );
    assert_eq!(slices_for(SeqparRequest::Auto, HD, 4096), 16);
}

/// The policy is stated in threads, so a different attention geometry
/// gets the same threadgroup width rather than the same slice count —
/// otherwise it would silently dispatch 2x the threads on head_dim 128.
#[test]
fn auto_is_expressed_in_threads_not_slices() {
    assert_eq!(slices_for(SeqparRequest::Auto, HD_LLAMA, 2048), 8);
    assert_eq!(slices_for(SeqparRequest::Auto, 32, 2048), 32);
    for hd in [32usize, 64, 128] {
        for span in [64u32, 512, 2048] {
            let slices = slices_for(SeqparRequest::Auto, hd, span);
            assert!(
                slices * hd <= SEQPAR_MAX_THREADS,
                "head_dim {hd} span {span}: {slices} slices exceed the \
                 {SEQPAR_MAX_THREADS}-thread bound"
            );
        }
    }
}

/// An explicit count still overrides the policy, and is still clamped —
/// the assert in `encode_kv_attend_seqpar` is a panic, so the clamp has
/// to happen here.
#[test]
fn explicit_request_overrides_but_is_still_clamped() {
    assert_eq!(slices_for(SeqparRequest::Slices(4), HD, 2048), 4);
    assert_eq!(
        slices_for(SeqparRequest::Slices(999), HD, 64),
        SEQPAR_MAX_THREADS / HD
    );
    assert_eq!(
        slices_for(SeqparRequest::Slices(999), HD_LLAMA, 64),
        SEQPAR_MAX_THREADS / HD_LLAMA
    );
}

/// 0 and 1 are refusals, not slice counts: the caller reads them as
/// "dispatch the shipped serial kernel".
#[test]
fn zero_one_and_degenerate_head_dim_refuse() {
    assert_eq!(slices_for(SeqparRequest::Slices(0), HD, 2048), 0);
    assert_eq!(slices_for(SeqparRequest::Slices(1), HD, 2048), 0);
    assert_eq!(slices_for(SeqparRequest::Auto, 0, 2048), 0);
    assert_eq!(slices_for(SeqparRequest::Unset, 0, 2048), 0);
    // head_dim past the whole thread budget cannot be sliced at all.
    assert_eq!(slices_for(SeqparRequest::Auto, SEQPAR_MAX_THREADS, 2048), 0);
    assert_eq!(
        slices_for(SeqparRequest::Slices(8), SEQPAR_MAX_THREADS, 2048),
        0
    );
}

// ---------------------------------------------------------------------
// Geometry-scoped default. The claim under test is narrow on purpose:
// `auto` is the default only where the span policy was measured.
// ---------------------------------------------------------------------

/// `Unset` resolves *through* the default list — it is neither hardcoded
/// off nor hardcoded to the policy. Stated as an invariant over every
/// geometry so it holds unchanged when the list gains an entry: today it
/// pins "the list is consulted", after the flip it pins "64 gets the auto
/// policy, tracking the span tiers rather than a fixed slice count".
#[test]
fn unset_resolves_through_the_default_list() {
    for hd in [32usize, 64, 128, 256, 512] {
        for span in [64u32, 256, 511, 512, 768, 1023, 1024, 4096] {
            let expected = if default_is_auto(hd) {
                slices_for(SeqparRequest::Auto, hd, span)
            } else {
                0
            };
            assert_eq!(
                slices_for(SeqparRequest::Unset, hd, span),
                expected,
                "unset at head_dim {hd}, span {span} must follow \
                 default_is_auto({hd}) = {}",
                default_is_auto(hd)
            );
        }
    }
}

/// The pending state. KV-B1's short-context block is clean at head_dim 64,
/// but the long-prompt and deep-context blocks have not been run on an
/// exclusively owned GPU, so nothing defaults on yet.
///
/// **This test flips with the enablement commit, not before it.** If it
/// starts failing, either the default list grew without the A/B/C gate or
/// the gate closed and this expectation is stale — check which.
#[test]
fn head_dim_64_defaults_on_and_nothing_else_does() {
    // The measured geometry gets the span-tracking policy, not a fixed count.
    assert_eq!(slices_for(SeqparRequest::Unset, HD, 256), 8);
    assert_eq!(slices_for(SeqparRequest::Unset, HD, 2048), 16);

    for hd in [32usize, 128, 256, 512] {
        for span in [64u32, 512, 2048] {
            assert_eq!(
                slices_for(SeqparRequest::Unset, hd, span),
                0,
                "head_dim {hd} is unmeasured and must not default on"
            );
        }
    }
}

/// head_dim 128 is supported by the kernel and would produce a legal
/// 4-slice dispatch — that is exactly why it must stay off by default.
/// The sweep has never run there, so `auto` would be an extrapolation.
#[test]
fn unset_on_head_dim_128_stays_off_even_though_it_would_work() {
    for span in [64u32, 512, 2048] {
        assert_eq!(
            slices_for(SeqparRequest::Unset, HD_LLAMA, span),
            0,
            "head_dim {HD_LLAMA} is unmeasured; unset must not opt it in"
        );
    }
    // The refusal is a policy choice, not a capability limit: asking
    // explicitly still works.
    assert_eq!(slices_for(SeqparRequest::Auto, HD_LLAMA, 2048), 8);
}

/// head_dim 512 is the case the narrow default exists to prevent: `auto`
/// there resolves to 1 slice below span 1024 and is refused, so a broad
/// default would present a silent no-op as a defaulted optimisation.
#[test]
fn unset_on_head_dim_512_stays_off_and_would_have_been_a_no_op() {
    for span in [64u32, 512, 2048] {
        assert_eq!(slices_for(SeqparRequest::Unset, HD_GEMMA_GLOBAL, span), 0);
    }
    // What a broad default would have got: nothing at all below the long
    // tier, and 2 slices above it.
    assert_eq!(slices_for(SeqparRequest::Auto, HD_GEMMA_GLOBAL, 512), 0);
    assert_eq!(slices_for(SeqparRequest::Auto, HD_GEMMA_GLOBAL, 2048), 2);
}

/// Gemma 3's sliding layers are the other half of that model and are also
/// unmeasured, so the whole architecture stays off by default rather than
/// half-on.
#[test]
fn unset_on_head_dim_256_stays_off() {
    for span in [64u32, 512, 2048] {
        assert_eq!(slices_for(SeqparRequest::Unset, HD_GEMMA_SLIDING, span), 0);
    }
    assert_eq!(slices_for(SeqparRequest::Auto, HD_GEMMA_SLIDING, 2048), 4);
}

/// `off` is an operator override and must stay off on every geometry.
///
/// Vacuous while the default list is empty — `Unset` is also off — and it
/// is here for the state after the flip, when `Off` and `Unset` diverge at
/// head_dim 64 and this becomes the only thing pinning the override.
#[test]
fn explicit_off_stays_off_on_every_geometry() {
    for hd in [32usize, 64, 128, 256, 512] {
        for span in [64u32, 512, 2048] {
            assert_eq!(
                slices_for(SeqparRequest::Off, hd, span),
                0,
                "explicit off must beat any default at head_dim {hd}, span {span}"
            );
        }
    }
}

/// An explicit slice count is honoured on unmeasured geometries — the
/// default is narrow, the capability is not.
#[test]
fn explicit_slices_are_honoured_on_unmeasured_geometries() {
    assert_eq!(slices_for(SeqparRequest::Slices(4), HD_LLAMA, 2048), 4);
    assert_eq!(
        slices_for(SeqparRequest::Slices(2), HD_GEMMA_GLOBAL, 2048),
        2
    );
}

/// The default list is the single widening point, and it should not grow
/// by accident. This test is expected to be edited *with* the evidence
/// that justifies the new entry — never on its own.
#[test]
fn the_default_list_holds_the_measured_geometry() {
    assert_eq!(
        SEQPAR_DEFAULT_ON_HEAD_DIMS,
        &[64],
        "adding a head_dim requires the A/B/C gate in \
         docs/kv-attention-scaling.md (for 64) or a bench_attention_span \
         sweep on an idle GPU (for any other); see the const's doc comment"
    );
    assert!(default_is_auto(HD));
    for hd in [32usize, 128, 256, 512] {
        assert!(!default_is_auto(hd), "head_dim {hd} is unmeasured");
    }
}
