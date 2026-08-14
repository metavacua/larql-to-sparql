//! §9's "what this is" section: model dimensions plus the slice table.

use larql_vindex_spec::VindexManifest;

use crate::SliceSummary;

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Render the model-description section.
pub fn render_overview(manifest: &VindexManifest) -> String {
    format!(
        "# {model}\n\n\
         A [vindex](https://github.com/chrishayuk/chuk-larql-rs) — a re-layout \
         of `{model}`'s weights for mechanistic-interpretability queries and \
         remote-FFN inference, not a lossy transform.\n\n\
         - Architecture: `{family}`\n\
         - Layers: {num_layers}\n\
         - Hidden size: {hidden_size}\n\
         - FFN intermediate size: {intermediate_size}\n\
         - Vocab size: {vocab_size}",
        model = manifest.model,
        family = manifest.family,
        num_layers = manifest.num_layers,
        hidden_size = manifest.hidden_size,
        intermediate_size = manifest.intermediate_size,
        vocab_size = manifest.vocab_size,
    )
}

/// Render the slice-size table, or an empty string if there are no
/// slices to report yet.
pub fn render_slice_table(slices: &[SliceSummary]) -> String {
    if slices.is_empty() {
        return String::new();
    }
    let mut out = "## Slices\n\n| Output | Size |\n|---|---|\n".to_string();
    for slice in slices {
        out.push_str(&format!(
            "| `{}` | {} |\n",
            slice.preset,
            format_size(slice.size_bytes)
        ));
    }
    out.trim_end().to_string()
}

fn format_size(bytes: u64) -> String {
    let gib = bytes as f64 / BYTES_PER_GIB;
    if gib >= 1.0 {
        format!("{gib:.2} GiB")
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> VindexManifest {
        larql_vindex_spec::test_fixtures::sample_manifest()
    }

    #[test]
    fn overview_includes_model_and_dims() {
        let out = render_overview(&sample_manifest());
        assert!(out.contains("# google/gemma-3-4b-it"));
        assert!(out.contains("Layers: 34"));
        assert!(out.contains("Hidden size: 2560"));
    }

    #[test]
    fn slice_table_empty_when_no_slices() {
        assert_eq!(render_slice_table(&[]), "");
    }

    #[test]
    fn slice_table_lists_every_slice_with_formatted_size() {
        let slices = vec![
            SliceSummary {
                preset: "full".into(),
                size_bytes: 3 * 1024 * 1024 * 1024,
            },
            SliceSummary {
                preset: "client".into(),
                size_bytes: 512 * 1024 * 1024,
            },
        ];
        let out = render_slice_table(&slices);
        assert!(out.contains("| `full` | 3.00 GiB |"));
        assert!(out.contains("| `client` | 512.0 MiB |"));
    }
}
