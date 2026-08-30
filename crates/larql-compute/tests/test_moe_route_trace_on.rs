//! End-to-end expert-routing trace: `LARQL_MOE_ROUTE_TRACE` names a writable
//! path, so the process-global sink resolves and every recorded block appends
//! one JSON line.
//!
//! Exactly ONE test lives in this binary: the sink is resolved once per
//! process from the environment (`OnceLock`), so on/off scenarios cannot
//! share a process — and `std::env::set_var` is only safe with no concurrent
//! `getenv` (see `options::ENV_OVERRIDES`), which a single-test binary
//! guarantees.
//!
//! This duplicates the `trace::tests` unit assertions through the public API
//! on purpose: the coverage gate merges this binary's instantiation of the
//! crate with the lib-test binary's, so the real env-driven path has to run
//! here for `trace.rs` to clear its per-file floor.

use larql_compute::ffn::expert_weight::trace;

#[test]
fn trace_end_to_end_writes_jsonl() {
    let path =
        std::env::temp_dir().join(format!("larql-route-trace-on-{}.jsonl", std::process::id()));
    std::env::set_var(larql_compute::options::ENV_MOE_ROUTE_TRACE, &path);

    let mut block = trace::buffer(2).expect("tracing is on");
    block.push(vec![3, 1]);
    block.push(vec![2, 0]);
    trace::record(4, Some(block));

    let empty = trace::buffer(0).expect("still on");
    trace::record(4, Some(empty)); // empty block still records, seq advances
    trace::record(4, None); // caller ran with tracing off — must be a no-op

    let raw = std::fs::read_to_string(&path).expect("trace file exists");
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 2, "two recorded blocks, the None ignored");
    assert_eq!(lines[0], r#"{"layer":4,"seq":0,"experts":[[3,1],[2,0]]}"#);
    assert_eq!(lines[1], r#"{"layer":4,"seq":1,"experts":[]}"#);

    let _ = std::fs::remove_file(&path);
}
