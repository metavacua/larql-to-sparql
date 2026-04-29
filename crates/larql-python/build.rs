// SPDX-License-Identifier: Apache-2.0

fn main() {
    // pyo3 extension-module: libpython symbols resolve at runtime via the host
    // interpreter, but the macOS linker rejects undefined symbols by default.
    // Maturin handles this; for plain `cargo build -p larql-python`, opt in here.
    // Use CARGO_CFG_TARGET_OS for the build target, not compile-time detection.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
