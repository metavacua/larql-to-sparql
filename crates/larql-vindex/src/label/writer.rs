use std::io::Write;

/// Write `feature_labels.json` (JSONL `{l,f,t}`) into `dir`, in the format
/// `load_feature_labels` reads. Each entry is one `((layer, feature), label)`.
pub fn write_feature_labels(
    dir: &std::path::Path,
    labels: &[((usize, usize), String)],
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(dir.join("feature_labels.json"))?;
    for ((l, feat), t) in labels {
        writeln!(f, "{}", serde_json::json!({"l": l, "f": feat, "t": t}))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_jsonl_that_load_feature_labels_reads() {
        let dir = tempfile::tempdir().unwrap();
        let labels = vec![
            ((5usize, 1usize), "capital".to_string()),
            ((7, 3), "official language".to_string()),
        ];
        write_feature_labels(dir.path(), &labels).unwrap();
        // round-trip through the REAL engine reader:
        let back = crate::format::load::load_feature_labels(
            &dir.path().join("feature_labels.json"),
        )
        .unwrap();
        assert_eq!(back.get(&(5, 1)).map(String::as_str), Some("capital"));
        assert_eq!(
            back.get(&(7, 3)).map(String::as_str),
            Some("official language")
        );
    }
}
