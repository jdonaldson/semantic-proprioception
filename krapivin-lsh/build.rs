use std::io::Result;

fn main() -> Result<()> {
    // Generate Rust code from protobuf schema
    prost_build::compile_protos(&["../krapivin_index.proto"], &["../"])?;
    Ok(())
}
