// On non-Windows hosts we bundle `protoc` via `protobuf_src` so the
// build needs no system install. On Windows the cmake link step fails
// against the debug UCRT (see Cargo.toml), so `protoc` must be on
// PATH — CI sets it via `arduino/setup-protoc`. `tonic_build` picks
// up `protoc` from the `PROTOC` env var or PATH automatically.
#[cfg(not(windows))]
fn set_protoc() {
    // `tonic_build::compile_protos` builds its own prost config and
    // discovers protoc via the PROTOC env var — a locally-built
    // `prost_build::Config` pointing at the bundled binary never reaches
    // it (the historical form of this function did exactly that, and the
    // build only worked while the CI runner images happened to ship a
    // system protoc). Respect an explicit PROTOC if the caller set one.
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
    tonic_build::compile_protos("proto/grid.proto")?;
    tonic_build::compile_protos("proto/expert.proto")?;
    tonic_build::compile_protos("proto/shard.proto")?;
    Ok(())
}
