//! Compile-time resolution of the Gauss sum using the arithmetic kernel.
//!
//! The video demo ("Gemma says 4050, compiled says 5050") hinges on
//! resolving `sum(1..101)` into `"5050"` at compile time, then installing
//! that string as the answer-side of a compiled edge. This example only
//! shows the compute step.
//!
//! Run with: `cargo run --example gauss -p model-compute`

#[cfg(feature = "native")]
fn main() {
    use model_compute::native::KernelRegistry;

    let registry = KernelRegistry::with_defaults();
    let cases = [
        ("arithmetic", "sum(1..101)"),
        ("arithmetic", "100 * 101 / 2"),
        ("arithmetic", "factorial(10)"),
        ("arithmetic", "math::pow(2.0, 16.0)"),
        ("datetime", "days_between(2025-01-01, 2026-01-01)"),
        ("datetime", "weekday(2026-04-16)"),
    ];

    for (kernel, expr) in cases {
        match registry.invoke(kernel, expr) {
            Ok(out) => println!("{:12} {:40} → {}", kernel, expr, out),
            Err(e) => println!("{:12} {:40} ERR: {}", kernel, expr, e),
        }
    }
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("gauss example requires the `native` feature (default). Re-run with --features native.");
}
