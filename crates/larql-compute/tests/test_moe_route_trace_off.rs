//! Expert-routing trace with an uncreatable path: tracing degrades to OFF
//! (with a stderr explanation) instead of failing the scoring run.
//!
//! One test per binary for the same reasons as `test_moe_route_trace_on.rs`:
//! the sink resolves once per process, and `set_var` needs a quiet process.

use larql_compute::ffn::expert_weight::trace;

#[test]
fn unwritable_trace_path_degrades_to_off() {
    // A path under a directory that does not exist cannot be created.
    let bad = std::env::temp_dir().join(format!(
        "larql-route-trace-missing-dir-{}/x/trace.jsonl",
        std::process::id()
    ));
    std::env::set_var(larql_compute::options::ENV_MOE_ROUTE_TRACE, &bad);

    assert!(trace::buffer(4).is_none(), "bad path must disable tracing");
    // Both record shapes are no-ops with no sink — neither may panic.
    trace::record(0, None);
    trace::record(0, Some(vec![vec![1, 2]]));
}
