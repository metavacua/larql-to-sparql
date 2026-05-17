// Ported from tests/wasm_roundtrip.rs.
// model-compute's `wasm` feature hosts a wasmi-based solver runtime.
// When this crate itself runs inside wasm32 (via wasm-pack test), wasmi
// runs as a WASM interpreter inside WASM — a legal but unusual nesting.
// Compile or runtime failures here reveal what blocks the double-hosted path.

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_node_experimental);

// ── Smoke: crate loads and native module is accessible ────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn native_module_accessible() {
    // Native arithmetic kernel is always compiled in; just confirm it exists.
    let _ = model_compute::native::arithmetic::ArithmeticKernel;
}

// ── WASM host round-trip (requires `wasm` feature) ───────────────────────────

#[cfg(feature = "wasm")]
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn wasm_echo_roundtrip() {
    use model_compute::wasm::{SolverLimits, SolverRuntime};

    const ECHO_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (global $in_ptr i32 (i32.const 0))
  (global $out_ptr i32 (i32.const 4096))
  (global $in_len (mut i32) (i32.const 0))
  (global $out_len (mut i32) (i32.const 0))
  (func (export "alloc") (param $size i32) (result i32)
    (global.set $in_len (local.get $size))
    (global.get $in_ptr))
  (func (export "solve") (param $ptr i32) (param $len i32) (result i32)
    (memory.copy (global.get $out_ptr) (local.get $ptr) (local.get $len))
    (global.set $out_len (local.get $len))
    (i32.const 0))
  (func (export "solution_ptr") (result i32) (global.get $out_ptr))
  (func (export "solution_len") (result i32) (global.get $out_len)))
"#;

    let wasm = wat::parse_str(ECHO_WAT).unwrap();
    let limits = SolverLimits::default();
    let rt = SolverRuntime::new(&wasm, limits).unwrap();
    let input = b"hello wasm";
    let result = rt.solve(input).unwrap();
    assert_eq!(result, input);
}
