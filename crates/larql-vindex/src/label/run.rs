//! Producer orchestration for `larql label`: load captured residuals and
//! drive the (already-tested) routing + frame-subtraction + match pieces to
//! emit `((layer, feature), relation)` labels.
//!
//! Two pure, model-free entry points live here so the producer's logic is
//! testable without loading a model:
//!
//! - [`load_subject_residuals`] — read a `residuals.vectors.jsonl` into a
//!   `subject -> layer -> vector` map.
//! - [`label_catalog`] — compose [`crate::label::contrastive`] over a whole
//!   [`Catalog`] of relations.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array1;
use serde_json::Value;

use crate::index::VectorIndex;
use crate::label::catalog::Catalog;
use crate::label::contrastive::{label_relation_from_routed, routed_features};

/// Read a `residuals.vectors.jsonl` (as written by the inference capturer)
/// into `subject -> (layer -> residual vector)`.
///
/// Each data line carries `id` ("<subject>_L<layer>"), `layer`, and `vector`.
/// The `_header` line — and any line missing `id`/`layer`/`vector` — is
/// skipped. The subject is `id` with its trailing `_L<layer>` stripped (using
/// the `layer` field, so subjects that themselves contain `_L` survive).
pub fn load_subject_residuals(
    path: &Path,
) -> std::io::Result<HashMap<String, HashMap<usize, Array1<f32>>>> {
    let text = std::fs::read_to_string(path)?;
    let mut out: HashMap<String, HashMap<usize, Array1<f32>>> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let (Some(id), Some(layer), Some(vec_json)) =
            (v.get("id").and_then(Value::as_str), v.get("layer").and_then(Value::as_u64), v.get("vector"))
        else {
            continue;
        };
        let Some(arr) = vec_json.as_array() else {
            continue;
        };
        let layer = layer as usize;
        let subject = id
            .strip_suffix(&format!("_L{layer}"))
            .unwrap_or(id)
            .to_string();
        let vector: Vec<f32> = arr
            .iter()
            .filter_map(|x| x.as_f64().map(|f| f as f32))
            .collect();
        out.entry(subject)
            .or_default()
            .insert(layer, Array1::from_vec(vector));
    }
    Ok(out)
}

/// Label every relation in `catalog`, composing the tested routing +
/// frame-subtraction pieces. For each relation, build the per-subject routed
/// features (only for subjects that have residuals), the `down_meta` token map
/// for the routed features, then run frame-subtraction + object-match. Returns
/// the union of labels across relations, de-duplicated.
///
/// `residuals` is keyed by `(relation_name, subject)` because a subject's
/// last-token residual is relation-prompt-specific (the residual of
/// "The capital of France is" differs from "The official language of France
/// is"). Each relation is therefore labeled off ITS OWN relation-prompt
/// residual; a subject shared across relations is not conflated.
pub fn label_catalog(
    index: &VectorIndex,
    catalog: &Catalog,
    residuals: &HashMap<(String, String), HashMap<usize, Array1<f32>>>,
    per_layer_k: usize,
    frame_frac: f32,
) -> Vec<((usize, usize), String)> {
    let mut labels: Vec<((usize, usize), String)> = Vec::new();
    for (rel_name, relation) in catalog.iter() {
        // Routed (layer,feat) per subject that has residuals for THIS relation.
        let mut routed: Vec<(String, Vec<(usize, usize)>)> = Vec::new();
        for (subject, _obj) in &relation.pairs {
            if let Some(resid) = residuals.get(&(rel_name.clone(), subject.clone())) {
                routed.push((subject.clone(), routed_features(index, resid, per_layer_k)));
            }
        }

        // down_meta top tokens for every routed (layer,feat).
        let mut down: HashMap<(usize, usize), String> = HashMap::new();
        for (_, feats) in &routed {
            for &(layer, feat) in feats {
                if down.contains_key(&(layer, feat)) {
                    continue;
                }
                if let Some(meta) = index.feature_meta(layer, feat) {
                    if !meta.top_token.is_empty() {
                        down.insert((layer, feat), meta.top_token);
                    }
                }
            }
        }

        labels.extend(label_relation_from_routed(
            &routed,
            &down,
            &relation.pairs,
            rel_name,
            frame_frac,
        ));
    }

    labels.sort();
    labels.dedup();
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::types::FeatureMeta;
    use std::io::Write;

    #[test]
    fn load_subject_residuals_skips_header_and_reads_layers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residuals.vectors.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Header line — must be skipped (no id/layer/vector).
        writeln!(
            f,
            r#"{{"_header":true,"component":"residuals","model":"m","dimension":3}}"#
        )
        .unwrap();
        // France at layer 5 and layer 7.
        writeln!(
            f,
            r#"{{"id":"France_L5","layer":5,"vector":[1.0,2.0,3.0]}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"id":"France_L7","layer":7,"vector":[4.0,5.0,6.0]}}"#
        )
        .unwrap();

        let map = load_subject_residuals(&path).unwrap();
        assert_eq!(map.len(), 1, "only France, header skipped");
        let france = map.get("France").expect("France present");
        assert_eq!(france.len(), 2, "both layers present");
        assert_eq!(france.get(&5).unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(france.get(&7).unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    fn meta(token: &str) -> FeatureMeta {
        FeatureMeta {
            top_token: token.into(),
            top_token_id: 0,
            c_score: 0.0,
            top_k: Vec::new(),
        }
    }

    /// Synthetic index: hidden=4, one layer. Gate feature `f` is the unit
    /// vector `e_(f%4)`, so a residual `e_i` routes (signed top-k) to the
    /// features with `f % 4 == i`. down_meta gives feature 1 → "Paris" and
    /// feature 2 → "Tokyo" so the object-match step can fire.
    fn synth_index() -> VectorIndex {
        let hidden = 4;
        let mut gate0 = ndarray::Array2::<f32>::zeros((4, hidden));
        for f in 0..4 {
            gate0[[f, f % 4]] = 1.0;
        }
        let down0 = vec![None, Some(meta("Paris")), Some(meta("Tokyo")), None];
        VectorIndex::new(vec![Some(gate0)], vec![Some(down0)], 1, hidden)
    }

    #[test]
    fn label_catalog_labels_subject_specific_features_full_round_trip() {
        let index = synth_index();
        // France routes to feature 1 (top_token "Paris"); Japan to feature 2
        // (top_token "Tokyo"). Disjoint features → neither is frame at n=2.
        let france = ndarray::Array1::from_vec(vec![0.0f32, 1.0, 0.0, 0.0]); // e_1
        let japan = ndarray::Array1::from_vec(vec![0.0f32, 0.0, 1.0, 0.0]); // e_2
        let mut residuals: HashMap<(String, String), HashMap<usize, Array1<f32>>> = HashMap::new();
        residuals.insert(
            ("capital".to_string(), "France".to_string()),
            [(0usize, france)].into_iter().collect(),
        );
        residuals.insert(
            ("capital".to_string(), "Japan".to_string()),
            [(0usize, japan)].into_iter().collect(),
        );

        let json = r#"{"capital":{"pid":"P36","template":"The capital of {entity} is","pairs":[["France","Paris"],["Japan","Tokyo"]]}}"#;
        let catalog = Catalog::from_json_str(json).unwrap();

        let labels = label_catalog(&index, &catalog, &residuals, 1, 0.5);
        assert!(
            labels.contains(&((0, 1), "capital".to_string())),
            "France's Paris feature labeled: {labels:?}"
        );
        assert!(
            labels.contains(&((0, 2), "capital".to_string())),
            "Japan's Tokyo feature labeled: {labels:?}"
        );
    }

    /// Synthetic index: hidden=4, one layer, four features, gate feature `f`
    /// is the unit vector `e_f`. down_meta gives 0 → "Paris", 1 → "Tokyo",
    /// 2 → "French", 3 → "German" — one feature per object across two relations.
    fn synth_index_two_relations() -> VectorIndex {
        let hidden = 4;
        let mut gate0 = ndarray::Array2::<f32>::zeros((4, hidden));
        for f in 0..4 {
            gate0[[f, f]] = 1.0;
        }
        let down0 = vec![
            Some(meta("Paris")),
            Some(meta("Tokyo")),
            Some(meta("French")),
            Some(meta("German")),
        ];
        VectorIndex::new(vec![Some(gate0)], vec![Some(down0)], 1, hidden)
    }

    /// A subject's last-token residual is RELATION-PROMPT-SPECIFIC: France's
    /// residual under "The capital of France is" differs from its residual
    /// under "The official language of France is". Keying residuals by
    /// `(relation, subject)` lets each relation label off its own residual.
    ///
    /// Here France routes to the "Paris" feature under `(capital, France)` and
    /// to a DIFFERENT "French" feature under `(language, France)`. Subject-only
    /// keying would store just one residual for France, so the second capture
    /// would clobber the first and only one of the two assertions below could
    /// ever hold — this test is impossible to satisfy under subject-only keying.
    #[test]
    fn label_catalog_keys_residuals_by_relation_for_shared_subject() {
        let index = synth_index_two_relations();
        let e = |i: usize| {
            let mut v = vec![0.0f32; 4];
            v[i] = 1.0;
            ndarray::Array1::from_vec(v)
        };
        let mut residuals: HashMap<(String, String), HashMap<usize, Array1<f32>>> = HashMap::new();
        // capital: France → Paris(feat 0), Japan → Tokyo(feat 1).
        residuals.insert(
            ("capital".to_string(), "France".to_string()),
            [(0usize, e(0))].into_iter().collect(),
        );
        residuals.insert(
            ("capital".to_string(), "Japan".to_string()),
            [(0usize, e(1))].into_iter().collect(),
        );
        // language: France → French(feat 2), Germany → German(feat 3).
        // Note France's residual here (e_2) DIFFERS from capital's (e_0).
        residuals.insert(
            ("language".to_string(), "France".to_string()),
            [(0usize, e(2))].into_iter().collect(),
        );
        residuals.insert(
            ("language".to_string(), "Germany".to_string()),
            [(0usize, e(3))].into_iter().collect(),
        );

        let json = r#"{
            "capital":{"pid":"P36","template":"The capital of {entity} is","pairs":[["France","Paris"],["Japan","Tokyo"]]},
            "language":{"pid":"P37","template":"The official language of {entity} is","pairs":[["France","French"],["Germany","German"]]}
        }"#;
        let catalog = Catalog::from_json_str(json).unwrap();

        let labels = label_catalog(&index, &catalog, &residuals, 1, 0.5);
        assert!(
            labels.contains(&((0, 0), "capital".to_string())),
            "capital labels France's Paris feature off its capital residual: {labels:?}"
        );
        assert!(
            labels.contains(&((0, 2), "language".to_string())),
            "language labels France's French feature off its language residual: {labels:?}"
        );
    }
}
