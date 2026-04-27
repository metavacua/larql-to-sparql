use larql_platform::{Platform, OperatingSystem};

fn main() {
    let platform = Platform::detect();

    // pyo3 extension-module: libpython symbols resolve at runtime via the host
    // interpreter, but the macOS linker rejects undefined symbols by default.
    // Maturin handles this; for plain `cargo build -p larql-python`, opt in here.
    if platform.os() == OperatingSystem::MacOS {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
