use larql_codebase_core::WeightRepr;
use larql_models::loading::gguf::{GgufTensor, GgufValue, GgufWriter};
use std::path::Path;

/// Write a [`WeightRepr`] to a GGUF v3 file.
///
/// Metadata written:
/// - `general.architecture` = `"bitnet"` (matches the `convert_cmd` dispatch)
/// - `general.name` = `name`
/// - `larql.hidden_size`, `larql.n_layers`, `larql.n_heads`
pub fn write_weight_repr_to_gguf(
    repr: WeightRepr,
    name: &str,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = GgufWriter::new();
    writer.meta("general.architecture", GgufValue::String("bitnet".into()));
    writer.meta("general.name", GgufValue::String(name.into()));
    writer.meta(
        "larql.hidden_size",
        GgufValue::U32(repr.arch.hidden_size as u32),
    );
    writer.meta("larql.n_layers", GgufValue::U32(repr.arch.n_layers as u32));
    writer.meta("larql.n_heads", GgufValue::U32(repr.arch.n_heads as u32));
    for t in repr.tensors {
        writer.tensor(GgufTensor {
            name: t.name,
            dims: t.dims,
            ggml_type: t.ggml_type,
            data: t.data,
        });
    }
    writer.write_to_file(path)?;
    Ok(())
}
