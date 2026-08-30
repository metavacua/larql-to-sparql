//! Colocated tests for `residency`.
//!
//! Page attribution is measurement code, and measurement code that is wrong
//! reports confidently. These pin the boundary behaviour the probe's headline
//! number depends on.

use crate::runtime::residency::{account, ExpertRegion, PageSpan, ResidencyAccount};

const PAGE: usize = 4_096;

fn expert(expert_id: u32, start: usize, len: usize) -> ExpertRegion {
    ExpertRegion {
        expert_id,
        bytes: start..start + len,
    }
}

// ── Page spans ─────────────────────────────────────────────────────────────

#[test]
fn a_range_inside_one_page_spans_that_page() {
    let span = PageSpan::of(&(10..20), PAGE);
    assert_eq!((span.first, span.end), (0, 1));
    assert_eq!(span.len(), 1);
}

#[test]
fn a_range_ending_mid_page_still_occupies_it() {
    // Rounding down here would under-report residency by one page per region.
    let span = PageSpan::of(&(0..PAGE + 1), PAGE);
    assert_eq!(span.end, 2);
    assert_eq!(span.len(), 2);
}

#[test]
fn a_page_aligned_range_does_not_claim_the_next_page() {
    let span = PageSpan::of(&(0..PAGE), PAGE);
    assert_eq!(span.end, 1);
}

#[test]
fn an_empty_range_spans_nothing() {
    let span = PageSpan::of(&(PAGE..PAGE), PAGE);
    assert!(span.is_empty());
    assert_eq!(span.len(), 0);
}

#[test]
fn a_zero_page_size_is_handled_rather_than_dividing_by_zero() {
    assert!(PageSpan::of(&(0..100), 0).is_empty());
}

#[test]
fn containment_is_half_open() {
    let span = PageSpan::of(&(0..PAGE), PAGE);
    assert!(span.contains(0));
    assert!(!span.contains(1), "end is exclusive");
}

#[test]
fn an_offset_range_starts_at_its_own_page() {
    let span = PageSpan::of(&(3 * PAGE..4 * PAGE), PAGE);
    assert_eq!((span.first, span.end), (3, 4));
}

// ── Attribution ────────────────────────────────────────────────────────────

/// Router on page 0; experts on pages 1, 2 and 3.
fn layout() -> (std::ops::Range<usize>, Vec<ExpertRegion>) {
    (
        0..PAGE,
        vec![
            expert(0, PAGE, PAGE),
            expert(1, 2 * PAGE, PAGE),
            expert(2, 3 * PAGE, PAGE),
        ],
    )
}

fn account_for(resident: &[bool], selected: &[u32]) -> ResidencyAccount {
    let (router, experts) = layout();
    account(resident, PAGE, &router, &experts, selected, 2 * PAGE)
}

#[test]
fn a_sparse_read_attributes_only_the_router_and_selected_experts() {
    // Router + expert 1 resident; experts 0 and 2 cold.
    let a = account_for(&[true, false, true, false], &[1]);
    assert_eq!(a.resident_total, 2);
    assert_eq!(a.resident_router, 1);
    assert_eq!(a.resident_selected, 1);
    assert_eq!(a.resident_unselected, 0, "routing stayed sparse");
}

#[test]
fn pages_belonging_to_unselected_experts_are_counted_as_overshoot() {
    // The whole file resident while only expert 1 ran — readahead defeating
    // the sparsity, which is precisely what the probe exists to detect.
    let a = account_for(&[true, true, true, true], &[1]);
    assert_eq!(a.resident_total, 4);
    assert_eq!(a.resident_unselected, 2);
    assert!((a.overshoot_fraction() - 0.5).abs() < 1e-9);
}

#[test]
fn nothing_resident_attributes_nothing() {
    let a = account_for(&[false; 4], &[1]);
    assert_eq!(
        a,
        ResidencyAccount {
            predicted: 2,
            ..Default::default()
        }
    );
    assert_eq!(a.overshoot_fraction(), 0.0);
    assert_eq!(a.prediction_coverage(), 0.0);
}

#[test]
fn every_resident_page_is_counted_exactly_once() {
    let a = account_for(&[true, true, true, true], &[0, 1]);
    assert_eq!(
        a.resident_router + a.resident_selected + a.resident_unselected,
        a.resident_total
    );
}

#[test]
fn a_shared_boundary_page_resolves_in_favour_of_the_selected_expert() {
    // Two experts sharing page 1. Attributing it to the unselected one would
    // report overshoot for a page the selected expert genuinely needed.
    let router = 0..PAGE;
    let experts = vec![
        expert(0, PAGE, PAGE / 2),
        expert(1, PAGE + PAGE / 2, PAGE / 2),
    ];
    let a = account(&[false, true], PAGE, &router, &experts, &[0], PAGE);
    assert_eq!(a.resident_selected, 1);
    assert_eq!(a.resident_unselected, 0);
}

#[test]
fn the_router_wins_a_page_it_shares_with_an_expert() {
    let router = 0..PAGE / 2;
    let experts = vec![expert(0, PAGE / 2, PAGE / 2)];
    let a = account(&[true], PAGE, &router, &experts, &[], PAGE / 2);
    assert_eq!(a.resident_router, 1);
    assert_eq!(a.resident_unselected, 0);
}

#[test]
fn a_page_outside_every_region_counts_only_toward_the_total() {
    // Padding or an unmapped tail. Real, resident, attributable to nothing.
    let a = account_for(&[false, false, false, false, true], &[1]);
    assert_eq!(a.resident_total, 1);
    assert_eq!(
        a.resident_router + a.resident_selected + a.resident_unselected,
        0
    );
}

// ── Derived ratios ─────────────────────────────────────────────────────────

#[test]
fn resident_fraction_is_of_the_whole_mapping() {
    let a = account_for(&[true, true, false, false], &[0]);
    assert!((a.resident_fraction(4) - 0.5).abs() < 1e-9);
}

#[test]
fn prediction_coverage_compares_used_pages_against_the_plan() {
    // Predicted two pages (router + one expert); both became resident.
    let a = account_for(&[true, false, true, false], &[1]);
    assert!((a.prediction_coverage() - 1.0).abs() < 1e-9);
}

#[test]
fn coverage_below_one_is_normal_rather_than_a_failure() {
    // The plan predicts every byte of every operand; execution may finish
    // without faulting a region's last page.
    let (router, experts) = layout();
    let a = account(
        &[true, false, false, false],
        PAGE,
        &router,
        &experts,
        &[1],
        2 * PAGE,
    );
    assert!(a.prediction_coverage() < 1.0);
}

#[test]
fn ratios_are_zero_rather_than_nan_when_there_is_no_denominator() {
    let a = ResidencyAccount::default();
    assert_eq!(a.resident_fraction(0), 0.0);
    assert_eq!(a.prediction_coverage(), 0.0);
    assert_eq!(a.overshoot_fraction(), 0.0);
}
