//! Attributing resident pages to the operands that caused them.
//!
//! # The VINDEX3 memory question
//!
//! The incumbent cold-read probe asks whether sparse access to a large blob
//! pages acceptably. VINDEX3 makes a stronger claim available, because binding
//! happens before execution:
//!
//! > A bound operation knows its own footprint. The bytes it will touch are
//! > derivable from the plan, before a single page is faulted.
//!
//! That turns residency from something you observe afterwards into something
//! you can *predict and then check*. Which matters for placement, prefetch and
//! remote transfer: all three need the byte set in advance, and a predictor
//! that quietly disagrees with reality would mis-size every one of them.
//!
//! # Why page attribution, not a byte count
//!
//! Residency is quantised to pages, and expert regions do not begin on page
//! boundaries. An expert whose region starts mid-page shares that page with
//! its neighbour, so touching one expert makes part of another resident. That
//! is not a bug — it is the granularity — but it means "bytes the plan needs"
//! and "bytes that become resident" are different quantities, and comparing
//! them without accounting for the boundary makes a correct implementation
//! look wasteful.
//!
//! This module keeps the two apart: `predicted` is what the plan asks for,
//! `resident` is what the kernel actually holds, and the difference is
//! attributed rather than averaged away.
//!
//! # Zero overshoot is a reference-kernel result, not a universal rule
//!
//! The reference executor reads exactly the operands the plan names, one row
//! at a time, so its measured residency equals its prediction. That is a fact
//! about *this kernel*, and it must not harden into an acceptance criterion
//! for every kernel.
//!
//! A grouped or accelerated kernel may legitimately touch more:
//!
//! ```text
//! group-extent reads      one storage group fetched for several experts
//! aligned over-read       vectorised loads spanning a block boundary
//! separate scale blocks   quantisation metadata outside the weight extent
//! kernel metadata         headers, offset tables
//! prefetch                speculative or asynchronous reads
//! ```
//!
//! So three quantities are eventually distinct, and only the first two are
//! about correctness:
//!
//! ```text
//! required pages        what the operation cannot run without
//! touch envelope        what the bound kernel may legally read
//! prefetch pages        what it may speculatively read ahead
//! ```
//!
//! For the reference kernel the envelope coincides with the requirement. For a
//! grouped kernel the envelope is the union of the complete group extents
//! containing selected experts, and overshoot against the *requirement*
//! becomes intentional and predictable rather than a defect.
//!
//! The durable invariant, which survives both cases:
//!
//! > Observed pages must lie inside the bound kernel's declared touch
//! > envelope, and every required page must be covered.
//!
//! This module measures the reference case. Declaring envelopes belongs to
//! kernel binding and is deliberately not modelled here yet.

use core::ops::Range;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Page indices spanned by a byte range, inclusive of both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageSpan {
    pub first: usize,
    /// One past the last page — empty when `first == end`.
    pub end: usize,
}

impl PageSpan {
    /// The pages a byte range occupies at `page_size`.
    pub fn of(range: &Range<usize>, page_size: usize) -> Self {
        if page_size == 0 || range.is_empty() {
            return Self { first: 0, end: 0 };
        }
        Self {
            first: range.start / page_size,
            // Round up: a range ending mid-page still occupies that page.
            end: range.end.div_ceil(page_size),
        }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.first)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, page: usize) -> bool {
        page >= self.first && page < self.end
    }
}

/// What a routed execution was expected to touch, and what it did touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyAccount {
    /// Pages resident anywhere in the mapping.
    pub resident_total: usize,
    /// Resident pages belonging to the router.
    pub resident_router: usize,
    /// Resident pages belonging to an expert the router selected.
    pub resident_selected: usize,
    /// Resident pages belonging only to experts that did **not** run.
    ///
    /// The number that matters. Routing is supposed to be sparse; if this
    /// tracks the whole population, either readahead is defeating the sparsity
    /// or the executor is reading operands it does not need.
    ///
    /// Zero is the *reference kernel's* result. A grouped kernel that fetches
    /// whole storage groups will show non-zero overshoot legitimately — see
    /// the touch-envelope note in the module docs before treating this as a
    /// pass/fail gate.
    pub resident_unselected: usize,
    /// Pages the plan predicted it would need.
    pub predicted: usize,
}

impl ResidencyAccount {
    /// Resident pages as a fraction of the whole mapping.
    pub fn resident_fraction(&self, total_pages: usize) -> f64 {
        ratio(self.resident_total, total_pages)
    }

    /// How much of the prediction actually became resident.
    ///
    /// Below 1.0 is normal and healthy: a plan predicts every byte of every
    /// operand, while execution may finish without faulting the last page of
    /// a region it only partly reads.
    pub fn prediction_coverage(&self) -> f64 {
        ratio(
            self.resident_router + self.resident_selected,
            self.predicted,
        )
    }

    /// Resident pages that no selected operand asked for.
    pub fn overshoot_fraction(&self) -> f64 {
        ratio(self.resident_unselected, self.resident_total)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

/// One expert's byte extent, as the residency probe sees it.
#[derive(Debug, Clone)]
pub struct ExpertRegion {
    pub expert_id: u32,
    pub bytes: Range<usize>,
}

/// Attribute a residency bitmap to router, selected experts and the rest.
///
/// `resident[i]` is whether page `i` of the mapping is resident. A page is
/// counted once, and shared pages resolve in favour of *use*: router first,
/// then selected, then unselected. Counting a boundary page against the
/// unselected experts it partly covers would report overshoot for a page the
/// selected expert genuinely needed.
pub fn account(
    resident: &[bool],
    page_size: usize,
    router: &Range<usize>,
    experts: &[ExpertRegion],
    selected: &[u32],
    predicted_bytes: usize,
) -> ResidencyAccount {
    let router_pages = PageSpan::of(router, page_size);
    let selected_spans: Vec<PageSpan> = experts
        .iter()
        .filter(|e| selected.contains(&e.expert_id))
        .map(|e| PageSpan::of(&e.bytes, page_size))
        .collect();
    let unselected_spans: Vec<PageSpan> = experts
        .iter()
        .filter(|e| !selected.contains(&e.expert_id))
        .map(|e| PageSpan::of(&e.bytes, page_size))
        .collect();

    let mut account = ResidencyAccount {
        predicted: predicted_bytes.div_ceil(page_size.max(1)),
        ..Default::default()
    };
    for (page, &is_resident) in resident.iter().enumerate() {
        if !is_resident {
            continue;
        }
        account.resident_total += 1;
        if router_pages.contains(page) {
            account.resident_router += 1;
        } else if selected_spans.iter().any(|s| s.contains(page)) {
            account.resident_selected += 1;
        } else if unselected_spans.iter().any(|s| s.contains(page)) {
            account.resident_unselected += 1;
        }
    }
    account
}
