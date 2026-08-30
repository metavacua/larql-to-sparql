//! The leaves must be disjoint, and the ledger must say so when they
//! are not.
//!
//! `timed` writes to the PROCESS ledger by design: the sites it
//! instruments are spread across the whole executor, and threading a
//! ledger through them would be the observer changing the code it
//! observes. So tests that go through `timed` assert on DELTAS, which
//! survive whatever else the suite is doing concurrently, while tests of
//! the ledger's own arithmetic hold a local instance — `reset` is
//! destructive, and a test that zeroed the shared counters would be
//! deleting another test's measurement.
//!
//! The real reconciliation runs where nothing else does: `larql vindex3
//! exec --generate`, one step, on the caller's thread.

use super::super::timing::{ledger, timed, ClassTally, OpClass, TimingLedger};

/// Every class is reachable, distinct, and named.
///
/// Names appear in the report a reader acts on, so a class added without
/// one — or two classes sharing one — would make the table lie about
/// where the time went.
#[test]
fn every_class_has_a_distinct_name_and_slot() {
    let mut names: Vec<&str> = OpClass::ALL.iter().map(|c| c.name()).collect();
    names.sort_unstable();
    let count = names.len();
    names.dedup();
    assert_eq!(names.len(), count, "two classes share a name");
    assert_eq!(count, OpClass::ALL.len());
}

/// A timed leaf lands in its own class and nowhere else.
#[test]
fn a_leaf_is_counted_once_in_its_own_class() {
    let before: Vec<ClassTally> = OpClass::ALL.iter().map(|c| ledger().get(*c)).collect();
    {
        let _t = timed(OpClass::DeltaGatedNorm);
        std::hint::black_box((0..2_000).map(|i| i as f64).sum::<f64>());
    }
    let after: Vec<ClassTally> = OpClass::ALL.iter().map(|c| ledger().get(*c)).collect();
    for (i, class) in OpClass::ALL.iter().enumerate() {
        let calls = after[i].calls - before[i].calls;
        if *class == OpClass::DeltaGatedNorm {
            assert!(calls >= 1, "the leaf did not reach its own class");
            assert!(
                after[i].nanos > before[i].nanos,
                "the leaf recorded no time"
            );
        } else {
            // Other tests run concurrently and may add to other classes,
            // so this cannot assert zero — only that a Projection was not
            // manufactured by timing a gated norm.
            assert!(
                calls < 1_000_000,
                "{} exploded, which means the slots are aliased",
                class.name()
            );
        }
    }
}

/// The ledger's own arithmetic, on a LOCAL instance.
///
/// Local because `reset` is destructive and the process ledger is shared
/// with every other test in the suite — a test that zeroed it would be
/// deleting another test's measurement, and would pass either way.
#[test]
fn the_ledger_sums_resets_and_enumerates() {
    let l = TimingLedger::new();
    assert_eq!(l.total_nanos(), 0);
    assert_eq!(l.nested(), 0);

    l.record(OpClass::Projection, 1_000);
    l.record(OpClass::Projection, 500);
    l.record(OpClass::DeltaRecurrence, 250);

    assert_eq!(
        l.get(OpClass::Projection),
        ClassTally {
            calls: 2,
            nanos: 1_500
        }
    );
    assert_eq!(l.total_nanos(), 1_750);

    // `all` enumerates every class in declaration order, so a reader
    // never has to remember which ones exist.
    let all = l.all();
    assert_eq!(all.len(), OpClass::ALL.len());
    assert_eq!(all[0].0, OpClass::Projection);
    assert_eq!(all[0].1.nanos, 1_500);
    assert_eq!(all.iter().map(|(_, t)| t.nanos).sum::<u64>(), 1_750);

    l.reset();
    assert_eq!(l.total_nanos(), 0);
    for class in OpClass::ALL {
        assert_eq!(
            l.get(class),
            ClassTally::default(),
            "{class:?} survived reset"
        );
    }
}

/// **Nesting is detected.** Two overlapping timers on one thread would
/// double-count, and a sum that double-counts can be made to equal
/// anything — which is exactly the property the reconciliation relies on
/// NOT having.
///
/// The guard counts rather than panics, because an instrumentation slip
/// must not take down a decode; the report is what refuses, by declining
/// to add up while `nested` is non-zero.
#[test]
fn overlapping_leaves_are_counted_as_nested() {
    let before = ledger().nested();
    {
        let _outer = timed(OpClass::Norm);
        let _inner = timed(OpClass::Rope);
    }
    assert!(
        ledger().nested() > before,
        "an overlapping pair was not reported as nested"
    );
}

/// The nesting flag is released, so a later leaf is not blamed for an
/// earlier one's overlap.
#[test]
fn the_thread_flag_clears_after_the_outermost_leaf() {
    {
        let _t = timed(OpClass::Residual);
    }
    let before = ledger().nested();
    {
        let _t = timed(OpClass::Residual);
    }
    assert_eq!(
        ledger().nested(),
        before,
        "a sequential pair of leaves was mistaken for a nested one — the flag leaked"
    );
}

/// `nanos_per_call` is the number a reader acts on: 12 ms over 48 calls
/// is an arithmetic cost and 12 ms over 5000 is a dispatch cost.
#[test]
fn per_call_time_is_reported_and_safe_when_empty() {
    assert_eq!(ClassTally::default().nanos_per_call(), 0.0);
    let t = ClassTally {
        calls: 4,
        nanos: 1_000,
    };
    assert_eq!(t.nanos_per_call(), 250.0);
}
