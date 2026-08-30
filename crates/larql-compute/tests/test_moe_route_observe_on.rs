//! `moe_route_observe::observe` with the trace sink actually OPEN.
//!
//! Exactly ONE test lives in this binary, for the same reason as
//! `test_moe_route_trace_on.rs`: the sink resolves once per process from
//! the environment (`OnceLock`), so a process that has already observed
//! with tracing off can never reach the recording path afterwards.
//!
//! The existing on-binary drives `trace::record` directly, which leaves
//! `observe`'s own two arms — refuse-and-count, and record-under-a-scope —
//! unexecuted. Those arms are the entire point of the module: it exists
//! because the trace was once attached to a code path the served model
//! does not run, and produced an empty file on a model that was visibly
//! generating tokens. A test that never opens the sink cannot tell that
//! failure from success.

use larql_compute::moe_route_observe::{observe, refused, LayerScope};

#[test]
fn observe_refuses_without_a_scope_and_records_within_one() {
    let path = std::env::temp_dir().join(format!(
        "larql-route-observe-on-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    std::env::set_var(larql_compute::options::ENV_MOE_ROUTE_TRACE, &path);

    // Sanity: the sink must really be open, or every assertion below is
    // vacuous — `observe` returns at the top when tracing is off and both
    // arms would "pass" without executing.
    assert!(
        larql_compute::ffn::expert_weight::trace::buffer(1).is_some(),
        "trace sink did not open; this binary cannot test either arm"
    );

    // The warning names the bypassing executor via a backtrace, but only
    // under RUST_BACKTRACE and only on the FIRST refusal — so it has to be
    // set before the `observe` below or that arm never runs in any test.
    std::env::set_var("RUST_BACKTRACE", "1");

    // ── Arm 1: no scope. Refused and COUNTED, never attributed to a
    // guessed layer. The predecessor design recovered a layer index from
    // a global counter `% 30`, which produces a plausible trace from an
    // unsound attribution — the worst outcome for a measurement.
    let before = refused();
    observe(&[1, 2, 3]);
    assert_eq!(
        refused(),
        before + 1,
        "an unscoped observation must increment the refusal counter, so a \
         silently-bypassing executor is visible as a non-zero count rather \
         than as a short file nobody questions"
    );

    // ── Arm 2: inside a scope. Recorded against the scope's layer.
    let refusals_before_scoped = refused();
    {
        let _scope = LayerScope::new(7);
        observe(&[5, 9]);
    }
    assert_eq!(
        refused(),
        refusals_before_scoped,
        "a scoped observation must NOT be refused"
    );

    let raw = std::fs::read_to_string(&path).expect("trace file exists");
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "exactly the scoped observation is recorded; the refused one must \
         not appear. got: {lines:?}"
    );
    assert_eq!(lines[0], r#"{"layer":7,"seq":0,"experts":[[5,9]]}"#);

    let _ = std::fs::remove_file(&path);
}
