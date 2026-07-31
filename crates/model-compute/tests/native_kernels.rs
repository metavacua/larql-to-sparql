//! wasm32 + host integration tests for the native compute kernels.
//!
//! Uses the dual cfg_attr pattern so the same test body runs as a
//! `#[test]` on the host and as a `#[wasm_bindgen_test]` on wasm32.
//! This gives wasm32 coverage of the arithmetic and datetime kernels
//! and verifies they behave identically across both targets.

#![cfg(feature = "native")]

// Default runner is Node.js (no configure macro needed).

use model_compute::native::{ArithmeticKernel, DateTimeKernel, Kernel, KernelRegistry};

// ── ArithmeticKernel ─────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_basic_ops() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("2 + 3").unwrap(), "5");
    assert_eq!(k.invoke("100 * 101 / 2").unwrap(), "5050");
    assert_eq!(k.invoke("(1 + 2) * 4").unwrap(), "12");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_gauss_sum() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("sum(1..101)").unwrap(), "5050");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_factorial() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("factorial(0)").unwrap(), "1");
    assert_eq!(k.invoke("factorial(5)").unwrap(), "120");
    assert_eq!(k.invoke("factorial(20)").unwrap(), "2432902008176640000");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_factorial_out_of_range() {
    use model_compute::native::KernelError;
    let k = ArithmeticKernel;
    assert!(matches!(
        k.invoke("factorial(21)").unwrap_err(),
        KernelError::OutOfRange(_)
    ));
    assert!(matches!(
        k.invoke("factorial(-1)").unwrap_err(),
        KernelError::OutOfRange(_)
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_product() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("product(1..6)").unwrap(), "120");
    assert_eq!(k.invoke("product(5..5)").unwrap(), "1");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_aggregate_composes() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("sum(1..11) * 2").unwrap(), "110");
    assert_eq!(k.invoke("sum(factorial(3)..factorial(4))").unwrap(), "261");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_float_ops() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("math::pow(2.0, 10.0)").unwrap(), "1024");
    assert_eq!(k.invoke("math::sqrt(16.0)").unwrap(), "4");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_empty_range_sum_zero() {
    let k = ArithmeticKernel;
    assert_eq!(k.invoke("sum(5..5)").unwrap(), "0");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_range_too_large_rejected() {
    use model_compute::native::KernelError;
    let k = ArithmeticKernel;
    assert!(matches!(
        k.invoke("sum(0..200000000)").unwrap_err(),
        KernelError::OutOfRange(_)
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_reversed_range_rejected() {
    use model_compute::native::KernelError;
    let k = ArithmeticKernel;
    assert!(matches!(
        k.invoke("sum(10..5)").unwrap_err(),
        KernelError::OutOfRange(_)
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn arith_identifier_prefix_no_match() {
    use model_compute::native::KernelError;
    let k = ArithmeticKernel;
    // "summary" must NOT trigger the sum() aggregate
    assert!(matches!(
        k.invoke("summary(1..10)").unwrap_err(),
        KernelError::Eval(_)
    ));
}

// ── DateTimeKernel ────────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_days_between() {
    let k = DateTimeKernel;
    assert_eq!(
        k.invoke("days_between(2026-01-01, 2026-04-16)").unwrap(),
        "105"
    );
    assert_eq!(
        k.invoke("days_between(2024-01-01, 2024-01-01)").unwrap(),
        "0"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_add_days() {
    let k = DateTimeKernel;
    assert_eq!(k.invoke("add_days(2026-01-01, 30)").unwrap(), "2026-01-31");
    assert_eq!(k.invoke("add_days(2026-01-31, -30)").unwrap(), "2026-01-01");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_weekday() {
    let k = DateTimeKernel;
    assert_eq!(k.invoke("weekday(2026-04-16)").unwrap(), "Thu");
    assert_eq!(k.invoke("weekday(2026-01-01)").unwrap(), "Thu");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_parse_date_valid() {
    let k = DateTimeKernel;
    assert_eq!(k.invoke("parse_date(2026-05-23)").unwrap(), "2026-05-23");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_parse_date_invalid() {
    use model_compute::native::KernelError;
    let k = DateTimeKernel;
    assert!(k.invoke("parse_date(not-a-date)").is_err());
    // Invalid calendar date
    assert!(matches!(
        k.invoke("parse_date(2026-02-30)").unwrap_err(),
        KernelError::Parse(_) | KernelError::OutOfRange(_)
    ));
}

// ── KernelRegistry ────────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn registry_defaults() {
    let r = KernelRegistry::with_defaults();
    let mut names = r.names();
    names.sort();
    assert_eq!(names, vec!["arithmetic", "datetime"]);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn registry_dispatch_arithmetic() {
    let r = KernelRegistry::with_defaults();
    assert_eq!(r.invoke("arithmetic", "sum(1..101)").unwrap(), "5050");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn registry_dispatch_datetime() {
    let r = KernelRegistry::with_defaults();
    assert_eq!(
        r.invoke("datetime", "days_between(2026-01-01, 2026-04-16)")
            .unwrap(),
        "105"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn registry_not_found() {
    use model_compute::native::KernelError;
    let r = KernelRegistry::with_defaults();
    assert!(matches!(
        r.invoke("nonexistent", "whatever").unwrap_err(),
        KernelError::NotFound(_)
    ));
}
