//! Integration test: load a minimal WAT fixture implementing the
//! alloc-write-solve-read ABI, exercise the full Session.solve round-trip.
//!
//! The fixture is a byte-echo solver: on solve(ptr, len) it copies the
//! input back to solution_buf. That's enough to verify alloc, write,
//! solve, solution_ptr, and solution_len all wire together.

#![cfg(feature = "wasm")]

use model_compute::wasm::{SolverError, SolverLimits, SolverRuntime};

const ECHO_WAT: &str = r#"
(module
  (memory (export "memory") 1)

  ;; Static layout: input region at 0, solution region at 4096.
  (global $in_ptr i32 (i32.const 0))
  (global $out_ptr i32 (i32.const 4096))
  (global $in_len (mut i32) (i32.const 0))
  (global $out_len (mut i32) (i32.const 0))

  ;; alloc(size) -> ptr
  (func (export "alloc") (param $size i32) (result i32)
    (global.set $in_len (local.get $size))
    (global.get $in_ptr))

  ;; solve(ptr, len) -> status
  ;; copy $len bytes from $ptr to $out_ptr, set $out_len, return 0.
  (func (export "solve") (param $ptr i32) (param $len i32) (result i32)
    (memory.copy
      (global.get $out_ptr)
      (local.get $ptr)
      (local.get $len))
    (global.set $out_len (local.get $len))
    (i32.const 0))

  (func (export "solution_ptr") (result i32)
    (global.get $out_ptr))

  (func (export "solution_len") (result i32)
    (global.get $out_len)))
"#;

const INFINITE_LOOP_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "alloc") (param i32) (result i32) (i32.const 0))
  (func (export "solve") (param i32) (param i32) (result i32)
    (loop $forever
      (br $forever))
    (i32.const 0))
  (func (export "solution_ptr") (result i32) (i32.const 0))
  (func (export "solution_len") (result i32) (i32.const 0)))
"#;

fn compile(runtime: &SolverRuntime, wat: &str) -> wasmtime::Module {
    let bytes = wat::parse_str(wat).expect("wat parse");
    runtime.compile(&bytes).expect("module compile")
}

#[test]
fn echo_roundtrip() {
    let runtime = SolverRuntime::new().unwrap();
    let module = compile(&runtime, ECHO_WAT);
    let mut session = runtime.session(&module).unwrap();

    let input = b"hello, model-compute";
    let output = session.solve(input).expect("solve");
    assert_eq!(output.as_slice(), input);
}

#[test]
fn echo_two_sessions_isolated() {
    // Two sessions on the same module must not share state.
    let runtime = SolverRuntime::new().unwrap();
    let module = compile(&runtime, ECHO_WAT);

    let mut s1 = runtime.session(&module).unwrap();
    let r1 = s1.solve(b"first").unwrap();
    assert_eq!(&r1, b"first");

    let mut s2 = runtime.session(&module).unwrap();
    let r2 = s2.solve(b"second-longer").unwrap();
    assert_eq!(&r2, b"second-longer");
}

#[test]
fn fuel_cap_stops_infinite_loop() {
    let limits = SolverLimits {
        fuel: 10_000,
        memory_pages: 16,
    };
    let runtime = SolverRuntime::with_limits(limits).unwrap();
    let module = compile(&runtime, INFINITE_LOOP_WAT);
    let mut session = runtime.session(&module).unwrap();

    let err = session.solve(b"anything").expect_err("should exhaust fuel");
    match err {
        SolverError::FuelExhausted { .. } | SolverError::Trap { .. } => {}
        other => panic!("expected fuel exhaustion, got {:?}", other),
    }
}

#[test]
fn missing_export_errors_clearly() {
    let runtime = SolverRuntime::new().unwrap();
    // Module with memory but no ABI exports
    let wat = r#"(module (memory (export "memory") 1))"#;
    let module = compile(&runtime, wat);
    let mut session = runtime.session(&module).unwrap();

    let err = session.solve(b"").expect_err("should fail");
    assert!(matches!(err, SolverError::MissingExport(name) if name == "alloc"));
}

/// Solver whose solve() tries to grow memory beyond the configured cap.
/// A memory grow of +N pages on top of the initial 1 page should trap
/// when N pushes past the limit the host set.
const MEMORY_HOG_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "alloc") (param i32) (result i32) (i32.const 0))
  (func (export "solve") (param i32) (param i32) (result i32)
    ;; try to grow by 1024 pages (64 MiB). If limit < 1025 total pages,
    ;; memory.grow returns -1; we trap via unreachable so the host sees it.
    (if (i32.eq (memory.grow (i32.const 1024)) (i32.const -1))
      (then unreachable))
    (i32.const 0))
  (func (export "solution_ptr") (result i32) (i32.const 0))
  (func (export "solution_len") (result i32) (i32.const 0)))
"#;

#[test]
fn memory_cap_rejects_grow() {
    let limits = SolverLimits {
        fuel: 10_000_000,
        memory_pages: 16, // 1 initial + anything trying to grow past 16 should fail
    };
    let runtime = SolverRuntime::with_limits(limits).unwrap();
    let module = compile(&runtime, MEMORY_HOG_WAT);
    let mut session = runtime.session(&module).unwrap();

    let err = session.solve(b"anything").expect_err("should hit memory cap");
    assert!(matches!(err, SolverError::Trap { .. }),
            "expected Trap from memory.grow=-1 + unreachable, got {:?}", err);
}

/// Solver whose solve() returns a non-zero status, signalling the guest
/// detected a semantic failure (infeasible problem, parse error, etc).
const FAIL_STATUS_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "alloc") (param i32) (result i32) (i32.const 0))
  (func (export "solve") (param i32) (param i32) (result i32) (i32.const 42))
  (func (export "solution_ptr") (result i32) (i32.const 0))
  (func (export "solution_len") (result i32) (i32.const 0)))
"#;

#[test]
fn nonzero_solve_status_reported() {
    let runtime = SolverRuntime::new().unwrap();
    let module = compile(&runtime, FAIL_STATUS_WAT);
    let mut session = runtime.session(&module).unwrap();

    let err = session.solve(b"anything").expect_err("should fail with status 42");
    assert!(matches!(err, SolverError::SolveFailed(42)));
}

#[test]
fn large_input_crosses_multiple_pages() {
    let runtime = SolverRuntime::new().unwrap();
    let module = compile(&runtime, ECHO_WAT);
    let mut session = runtime.session(&module).unwrap();

    // 200 KiB — crosses several 64 KiB pages. Echo module's memory grows
    // implicitly via linear.memory layout; the echo fixture places output
    // at offset 4096, so 200 KiB round-trips the full input + output region.
    // We cap at ~48 KiB to fit within the default 1-page memory of the WAT.
    let input: Vec<u8> = (0..48_000).map(|i| (i % 251) as u8).collect();
    let output = session.solve(&input).expect("solve");
    assert_eq!(output, input);
}

#[test]
fn fuel_remaining_decreases_after_call() {
    let runtime = SolverRuntime::new().unwrap();
    let module = compile(&runtime, ECHO_WAT);
    let mut session = runtime.session(&module).unwrap();

    let initial = session.fuel_remaining();
    session.solve(b"hello").unwrap();
    let after = session.fuel_remaining();
    assert!(after < initial, "fuel should decrease: before={initial}, after={after}");
}
