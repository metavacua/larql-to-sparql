// On non-Windows hosts we bundle `protoc` via `protobuf_src`. On
// Windows we rely on a system protoc (CI installs it via
// `arduino/setup-protoc`) because cmake-built protoc fails to link
// against the debug UCRT runtime on the GitHub windows-latest runner.
#[cfg(not(windows))]
fn set_protoc() {
    // `tonic_build::compile_protos` discovers protoc via the PROTOC env
    // var — a locally-built `prost_build::Config` pointing at the bundled
    // binary never reaches it (the historical form of this function did
    // exactly that, and the build only worked while the CI runner images
    // happened to ship a system protoc). Respect an explicit PROTOC if
    // the caller set one.
    if std::env::var_os("PROTOC").is_none() {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }
}

#[cfg(windows)]
fn set_protoc() {
    // No-op: rely on the system `protoc` discovered via PROTOC / PATH.
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_protoc();
    tonic_build::compile_protos("proto/vindex.proto")?;
    Ok(())
}
