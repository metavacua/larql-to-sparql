//! Allocation guard for the VINDEX3 reference execution path.
//!
//! # Why this exists
//!
//! `decode_at` rebuilt its operand's name on every scalar read, allocating a
//! `String` per element. It was correct, it was covered by tests, and it cost
//! ~17× — nothing in the suite could see it, and the perf bench only caught it
//! because the noise happened to differ between two storage arrangements.
//!
//! # The invariant
//!
//! > Operand lookup and scalar access must not allocate in proportion to
//! > tensor dimensions, expert population, or number of elements read.
//!
//! Deliberately *not* zero-allocation. `execute` returns owned buffers and
//! allocates per-expert scratch, so a per-token count of a few dozen is
//! expected and fine. What must not happen is that count tracking the work.
//! When an `execute_into` path with caller-owned scratch arrives, this
//! tightens to zero per token.
//!
//! The counting allocator is declared here rather than in the library, so it
//! applies to this test binary alone and never to production builds.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use larql_vindex::runtime::execute;
use larql_vindex::runtime::fixtures::synthetic::{SyntheticLayer, SyntheticShape};
use larql_vindex::runtime::MoeInputs;

thread_local! {
    /// Allocations on *this* thread.
    ///
    /// Per-thread, not a global atomic. `cargo test` runs test functions
    /// concurrently, so a process-wide counter measures whatever else happens
    /// to be running and the numbers move between runs — which is exactly the
    /// failure this instrument produced before being fixed.
    ///
    /// `const`-initialised so that reading the counter cannot itself allocate
    /// and re-enter the allocator.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn allocations() -> usize {
    ALLOCATIONS.try_with(Cell::get).unwrap_or(0)
}

fn record() {
    // `try_with`: during thread teardown the TLS slot is gone, and panicking
    // inside the allocator would abort.
    let _ = ALLOCATIONS.try_with(|c| c.set(c.get().wrapping_add(1)));
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocations performed by one `execute`, measured after a warm-up call.
///
/// Binding happens in the caller, deliberately. Binding allocates — it builds
/// operand names, contracts and expert lists — and it is load-path work; doing
/// it inside the measurement would put load-path allocations in a token-path
/// number and make the result depend on call ordering.
fn allocations_per_token(layer: &SyntheticLayer) -> usize {
    let operation = layer.bind(&layer.bytes);
    let residual = layer.residual();
    // First-call effects are not what this measures.
    execute(&operation, MoeInputs::shared(&residual)).expect("synthetic layer executes");
    measure(&layer.bind(&layer.bytes), &residual)
}

/// Allocations attributable to a single already-bound execution.
fn measure(operation: &larql_vindex::runtime::BoundMoeOperation<'_>, residual: &[f32]) -> usize {
    let before = allocations();
    let out = execute(operation, MoeInputs::shared(residual)).expect("synthetic layer executes");
    let after = allocations();
    // Keep the result alive so its buffer is not freed inside the window.
    std::hint::black_box(&out);
    after - before
}

fn shape(population: usize, hidden: usize, intermediate: usize) -> SyntheticShape {
    SyntheticShape {
        population,
        hidden,
        intermediate,
        top_k: 2,
    }
}

/// Slack for allocations that are constant in count but conditional on size.
///
/// Measured at 18 / 18 / 19 for populations 8 / 32 / 512. The single extra
/// allocation above ~20 experts is `slice::sort_by`'s merge scratch: the
/// stable sort uses insertion sort below that threshold and allocates a buffer
/// above it. One buffer, whatever the length — a constant, not a rate.
const CONSTANT_ALLOCATION_SLACK: usize = 4;

#[test]
fn allocation_count_does_not_track_expert_population() {
    // 64× the experts. Router scoring grows; allocation must not.
    let small = SyntheticLayer::build(shape(8, 32, 24));
    let large = SyntheticLayer::build(shape(512, 32, 24));

    let small_allocs = allocations_per_token(&small);
    let large_allocs = allocations_per_token(&large);

    assert!(
        large_allocs <= small_allocs + CONSTANT_ALLOCATION_SLACK,
        "allocations tracked population: {small_allocs} at 8 experts, \
         {large_allocs} at 512 — more than the {CONSTANT_ALLOCATION_SLACK} \
         allowed for size-conditional constants"
    );
}

#[test]
fn allocation_count_is_nowhere_near_proportional_to_population() {
    // The assertion above is a bound; this one states the shape. Were the
    // count proportional, 64× the experts would give ~64× the allocations.
    let small = SyntheticLayer::build(shape(8, 32, 24));
    let large = SyntheticLayer::build(shape(512, 32, 24));

    let small_allocs = allocations_per_token(&small);
    let large_allocs = allocations_per_token(&large);
    let proportional = small_allocs * 64;

    assert!(
        large_allocs * 4 < proportional,
        "{large_allocs} allocations at 512 experts is within 4× of the \
         {proportional} a per-expert allocation would produce"
    );
}

#[test]
fn allocation_count_does_not_track_elements_read() {
    // 16× the elements per expert, same population and top-k. This is the
    // shape of the bug that motivated the guard: a per-element allocation
    // shows up here and nowhere else.
    let narrow = SyntheticLayer::build(shape(16, 16, 16));
    let wide = SyntheticLayer::build(shape(16, 64, 64));

    let narrow_allocs = allocations_per_token(&narrow);
    let wide_allocs = allocations_per_token(&wide);

    assert_eq!(
        narrow_allocs, wide_allocs,
        "allocations moved with tensor size: {narrow_allocs} at 16×16, \
         {wide_allocs} at 64×64 — this is exactly the shape of the \
         per-element `String` bug this guard exists for"
    );
}

#[test]
fn allocation_count_stays_small_and_constant_per_token() {
    // A loose absolute ceiling. The exact number is an implementation detail
    // and will drop when `execute_into` lands; what this pins is that it is a
    // small constant rather than a function of the work.
    const CEILING: usize = 64;
    let layer = SyntheticLayer::build(shape(64, 32, 24));
    let allocs = allocations_per_token(&layer);
    assert!(
        allocs <= CEILING,
        "{allocs} allocations per token exceeds the ceiling of {CEILING}"
    );
}

#[test]
fn repeated_tokens_allocate_identically() {
    // Growth across repeats would mean something accumulates per token — a
    // cache, a log, a memo — which is how a steady-state leak begins.
    let layer = SyntheticLayer::build(shape(32, 32, 24));
    let operation = layer.bind(&layer.bytes);
    let residual = layer.residual();
    execute(&operation, MoeInputs::shared(&residual)).expect("warm-up");

    let first = measure(&operation, &residual);
    for token in 1..16 {
        assert_eq!(
            measure(&operation, &residual),
            first,
            "token {token} allocated differently from the first"
        );
    }
}

#[test]
fn the_counting_allocator_is_actually_observing_allocations() {
    // A guard whose instrument is broken passes every test. This one would
    // fail if the global allocator were not installed.
    let before = allocations();
    let v: Vec<u8> = Vec::with_capacity(1_024);
    std::hint::black_box(&v);
    assert!(
        allocations() > before,
        "the counting allocator recorded nothing"
    );
}
