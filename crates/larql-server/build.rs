fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use bundled protoc (no system install required).
    std::env::set_var("PROTOC", protobuf_src::protoc());
    tonic_build::compile_protos("proto/vindex.proto")?;
    Ok(())
}
