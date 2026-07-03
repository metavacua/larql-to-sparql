/// Write `feature_labels.json` into `dir` as a single pretty-printed JSON
/// OBJECT keyed `"L{layer}_F{feature}": "{relation}"` — the format the
/// DESCRIBE/STATS `RelationClassifier::from_vindex` reader parses.
///
/// Each entry is one `((layer, feature), relation)`. If two entries collide on
/// the same `(layer, feature)`, the last one wins.
///
/// (Note: this is NOT the JSONL `{l,f,t}` "down_meta top-token" format that
/// `larql_vindex::load_feature_labels` reads — that reader consumes a different
/// file, `down_meta.jsonl` / `ffn_gate.vectors.jsonl`, carrying per-feature top
/// *tokens*, not per-feature *relation* labels.)
pub fn write_feature_labels(
    dir: &std::path::Path,
    labels: &[((usize, usize), String)],
) -> std::io::Result<()> {
    let mut map = serde_json::Map::new();
    for ((layer, feat), relation) in labels {
        map.insert(
            format!("L{layer}_F{feat}"),
            serde_json::Value::String(relation.clone()),
        );
    }
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(map))
        .map_err(std::io::Error::other)?;
    std::fs::write(dir.join("feature_labels.json"), json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_object_format_relation_classifier_reads() {
        let dir = tempfile::tempdir().unwrap();
        let labels = vec![
            ((5usize, 1usize), "capital".to_string()),
            ((7, 3), "official language".to_string()),
        ];
        write_feature_labels(dir.path(), &labels).unwrap();
        // Assert the on-disk shape is a JSON OBJECT keyed "L{layer}_F{feat}"
        // (larql-vindex cannot depend on larql-lql, so assert the structure
        // the RelationClassifier reads directly here; the true round-trip
        // through RelationClassifier lives in larql-lql's integration test).
        let text = std::fs::read_to_string(dir.path().join("feature_labels.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        let obj = value.as_object().expect("feature_labels.json is a JSON object");
        assert_eq!(obj.get("L5_F1").and_then(|v| v.as_str()), Some("capital"));
        assert_eq!(
            obj.get("L7_F3").and_then(|v| v.as_str()),
            Some("official language")
        );
    }

    #[test]
    fn last_entry_wins_on_layer_feature_collision() {
        let dir = tempfile::tempdir().unwrap();
        let labels = vec![
            ((5usize, 1usize), "capital".to_string()),
            ((5, 1), "language".to_string()),
        ];
        write_feature_labels(dir.path(), &labels).unwrap();
        let text = std::fs::read_to_string(dir.path().join("feature_labels.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["L5_F1"].as_str(), Some("language"));
    }
}
